"""Recording a live session against an OpenAI-compatible chat completions API."""

from __future__ import annotations

from types import TracebackType
from typing import Any

import httpx

from agentslice.errors import AdapterError
from agentslice.recording.jsonl import TraceWriter
from agentslice.recording.openai_adapter import from_openai_messages


class LiveSession:
    """Records one HTTP chat-completions turn at a time.

    Synchronous, non-streaming, no retries: this is a thin recording
    wrapper around a single request/response pair, not a general-purpose
    agent runtime. The caller owns the conversation loop (executing tool
    calls, appending their results as ``tool`` messages, deciding when to
    stop); each call to :meth:`run_turn` sends the messages so far, records
    the model's reply, and returns the updated message list for the next
    turn.

    ``transport`` exists so tests can inject an ``httpx.MockTransport``
    instead of hitting a real API. ``timeout`` defaults to a generous
    120s: a reasoning model can easily take longer than ``httpx``'s 5s
    default before returning anything.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        *,
        model: str,
        transport: httpx.BaseTransport | None = None,
        side_effect_tools: set[str] | None = None,
        timeout: float = 120.0,
    ) -> None:
        self._model = model
        self._side_effect_tools = side_effect_tools
        self._events_written = 0
        self._client = httpx.Client(
            base_url=base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            transport=transport,
            timeout=timeout,
        )

    def run_turn(
        self,
        messages: list[dict[str, Any]],
        writer: TraceWriter,
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """Request one completion, write any new events, and return the updated history.

        ``messages`` must be exactly what a previous call to ``run_turn``
        returned (optionally with ``tool`` result messages appended by the
        caller); the whole history is re-converted through
        :func:`~agentslice.recording.openai_adapter.from_openai_messages` on
        every call so that fact tracking (used to infer ``reads``) stays
        correct across turns, but only newly produced events are written.

        Raises:
            AdapterError: The request failed (network error, timeout, or a
                non-2xx response), or the response body wasn't shaped like
                a chat completion.
        """
        payload: dict[str, Any] = {"model": self._model, "messages": messages}
        if tools:
            payload["tools"] = tools

        try:
            response = self._client.post("/chat/completions", json=payload)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise AdapterError(f"chat completion request failed: {exc}") from exc

        try:
            data = response.json()
        except ValueError as exc:
            raise AdapterError(f"invalid JSON response: {exc}") from exc

        try:
            assistant_message = data["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise AdapterError(f"unexpected chat completion response shape: {exc}") from exc

        updated_messages = [*messages, assistant_message]
        events = from_openai_messages(updated_messages, side_effect_tools=self._side_effect_tools)
        for event in events[self._events_written :]:
            writer.write(event)
        self._events_written = len(events)

        return updated_messages

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> LiveSession:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
