"""Replaying a compiled or forked context against a real model to capture its next action."""

from __future__ import annotations

from collections.abc import Sequence
from types import TracebackType
from typing import Any

import httpx

from agentslice.compiler.base import ToolSchema
from agentslice.errors import AdapterError
from agentslice.ir.events import TraceEvent
from agentslice.recording.openai_adapter import to_openai_messages
from agentslice.replay.tool_stubs import fill_pending_tool_results


class ReplaySession:
    """Sends a reconstructed message history to a real model and captures its reply.

    Mirrors :class:`agentslice.recording.live.LiveSession`'s
    transport-injection pattern (tests use ``httpx.MockTransport``), but
    unlike that class this session never records a trace of what happens —
    it exists to observe one next action for comparison, not to build a
    new one. Synchronous, non-streaming, no retries. ``timeout`` defaults
    to a generous 120s: a reasoning model can easily take longer than
    ``httpx``'s 5s default before returning anything.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        *,
        model: str,
        transport: httpx.BaseTransport | None = None,
        timeout: float = 120.0,
    ) -> None:
        self._model = model
        self._client = httpx.Client(
            base_url=base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            transport=transport,
            timeout=timeout,
        )

    def next_action(
        self, messages: list[dict[str, Any]], *, tools: list[dict[str, Any]] | None = None
    ) -> dict[str, Any]:
        """Request one completion and return the raw assistant message.

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
            return dict(data["choices"][0]["message"])
        except (KeyError, IndexError, TypeError) as exc:
            raise AdapterError(f"unexpected chat completion response shape: {exc}") from exc

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> ReplaySession:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


def _to_openai_tools(tool_catalog: dict[str, ToolSchema]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": schema.name,
                "description": schema.description,
                "parameters": schema.parameters,
            },
        }
        for schema in tool_catalog.values()
    ]


def replay_compiled_context(
    events: Sequence[TraceEvent],
    *,
    original_events: Sequence[TraceEvent],
    session: ReplaySession,
    tool_catalog: dict[str, ToolSchema] | None = None,
) -> dict[str, Any]:
    """Resend a compiled or forked context to a real model and return its proposed next action.

    Any ``tool_call`` in ``events`` with no matching ``tool_result`` gets
    one filled in from ``original_events`` (see
    :func:`~agentslice.replay.tool_stubs.fill_pending_tool_results`) before
    conversion to messages, so the request is always a valid, fully
    answered OpenAI message array. ``tool_catalog`` is normally
    ``compiled.tool_catalog`` — deliberately whatever the pipeline's
    ``schema_pruning`` pass narrowed it down to, since testing whether a
    model still behaves equivalently under that narrower catalog is the
    point of replay.
    """
    filled = fill_pending_tool_results(events, original_events)
    messages = to_openai_messages(filled)
    tools = _to_openai_tools(tool_catalog) if tool_catalog else None
    return session.next_action(messages, tools=tools)
