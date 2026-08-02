"""Conversion of a Codex CLI session rollout into a TraceEvent sequence.

A Codex rollout (``~/.codex/sessions/**/*.jsonl``) wraps OpenAI Responses
API items in a ``{"type": "response_item", "payload": {...}}`` envelope,
alongside session/turn metadata records this adapter has no use for. The
item shapes are already close to chat-completions messages (``message``
items with ``role``/``content`` blocks, standalone ``function_call`` /
``function_call_output`` items rather than an embedded ``tool_calls``
list) — this module reshapes them and delegates causal inference to
:func:`~agentslice.recording.openai_adapter.from_openai_messages`, rather
than re-implementing it.

Deliberate scope decisions:

- Only ``response_item`` records are read; ``session_meta``, ``turn_context``,
  and ``event_msg`` records carry no conversational content.
- A ``developer``-role message item is mapped to ``system``: Codex uses
  ``developer`` for what OpenAI chat completions calls the system role.
- ``reasoning`` items are dropped: no chat-completions equivalent.
- Consecutive ``function_call`` items (a model requesting several tool
  calls before any of them is answered) are grouped into a single
  synthetic assistant message with multiple ``tool_calls`` entries,
  mirroring how a real chat-completions response would have shaped them.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from agentslice.ir.events import TraceEvent
from agentslice.recording.openai_adapter import from_openai_messages

_MESSAGE_ROLES = {"system", "user", "assistant", "developer"}


def from_codex_rollout(
    records: Sequence[Mapping[str, Any]],
    *,
    side_effect_tools: set[str] | None = None,
) -> list[TraceEvent]:
    """Convert parsed Codex rollout records into ``TraceEvent`` objects.

    Args:
        records: Each line of a ``~/.codex/sessions/**/*.jsonl`` file,
            already parsed as JSON.
        side_effect_tools: See :func:`~agentslice.recording.openai_adapter.from_openai_messages`.

    Raises:
        UnsupportedMessageFormatError: A reshaped message is malformed in
            a way ``from_openai_messages`` rejects.
    """
    messages = _to_openai_messages(records)
    return from_openai_messages(messages, side_effect_tools=side_effect_tools)


def _text_of(content: list[dict[str, Any]]) -> str:
    return "\n".join(block.get("text", "") for block in content if block.get("text"))


def _to_openai_messages(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    pending_tool_calls: list[dict[str, Any]] = []

    def flush_tool_calls() -> None:
        if pending_tool_calls:
            messages.append(
                {"role": "assistant", "content": None, "tool_calls": list(pending_tool_calls)}
            )
            pending_tool_calls.clear()

    for record in records:
        if record.get("type") != "response_item":
            continue
        item = record.get("payload") or {}
        item_type = item.get("type")

        if item_type == "function_call":
            pending_tool_calls.append(
                {
                    "id": item["call_id"],
                    "type": "function",
                    "function": {
                        "name": item["name"],
                        "arguments": item.get("arguments") or "{}",
                    },
                }
            )
            continue

        flush_tool_calls()

        if item_type == "message":
            role = item.get("role")
            if role == "developer":
                role = "system"
            if role not in _MESSAGE_ROLES:
                continue
            text = _text_of(item.get("content") or [])
            if text:
                messages.append({"role": role, "content": text})
        elif item_type == "function_call_output":
            output = item.get("output")
            if isinstance(output, dict):
                output = output.get("output", "")
            messages.append(
                {"role": "tool", "tool_call_id": item["call_id"], "content": str(output or "")}
            )
        # reasoning and anything else: no chat-completions equivalent.

    flush_tool_calls()
    return messages
