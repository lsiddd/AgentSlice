"""Conversion of a Claude Code session transcript into a TraceEvent sequence.

A Claude Code project session (``~/.claude/projects/*/*.jsonl``) is not
chat-completions shaped: an assistant turn's content blocks (``thinking``,
``text``, ``tool_use``) each arrive as a *separate* top-level JSONL record
sharing one ``message.id``, and tool results travel back as ``tool_result``
blocks inside a ``user``-role record rather than a dedicated role. This
module reshapes that into OpenAI-compatible messages and delegates the
actual causal inference to :func:`~agentslice.recording.openai_adapter.from_openai_messages`,
rather than re-implementing it.

Deliberate scope decisions, unlike the stricter ``from_openai_messages``:

- Unrecognized top-level ``type`` values (the session format has many:
  ``mode``, ``attachment``, ``file-history-snapshot``, ``bridge-session``,
  ``system`` telemetry subtypes, and others observed to keep growing) are
  silently skipped rather than rejected. This is an internal CLI log
  format, not a documented, stable API contract; failing fast on every
  unfamiliar bookkeeping record would make the adapter brittle against
  routine schema growth in Claude Code itself.
- ``thinking`` and ``image`` content blocks are dropped: neither has an
  OpenAI chat-completions equivalent.
- Sidechains (``isSidechain: true``, i.e. sub-agent invocations such as
  the ``Task`` tool) are dropped entirely. They are a separate, nested
  conversation; folding them into the main thread would misrepresent
  turn structure.
- No system/constraint message is ever produced: unlike an OpenAI chat
  log, a Claude Code session transcript does not record the system
  prompt as a turn.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from agentslice.ir.events import TraceEvent
from agentslice.recording.openai_adapter import from_openai_messages


def from_claude_code_transcript(
    records: Sequence[Mapping[str, Any]],
    *,
    side_effect_tools: set[str] | None = None,
) -> list[TraceEvent]:
    """Convert parsed Claude Code session records into ``TraceEvent`` objects.

    Args:
        records: Each line of a ``~/.claude/projects/*/*.jsonl`` file,
            already parsed as JSON.
        side_effect_tools: See :func:`~agentslice.recording.openai_adapter.from_openai_messages`.

    Raises:
        UnsupportedMessageFormatError: A reshaped message is malformed in
            a way ``from_openai_messages`` rejects (e.g. a ``tool_result``
            block referencing a ``tool_use_id`` with no matching call).
    """
    messages = _to_openai_messages(records)
    return from_openai_messages(messages, side_effect_tools=side_effect_tools)


def _flatten_tool_result_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    return ""


def _to_openai_messages(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    group_id: str | None = None
    group_text: list[str] = []
    group_tool_calls: list[dict[str, Any]] = []

    def flush_group() -> None:
        nonlocal group_id, group_text, group_tool_calls
        if group_tool_calls:
            messages.append(
                {"role": "assistant", "content": None, "tool_calls": list(group_tool_calls)}
            )
        elif group_text:
            messages.append({"role": "assistant", "content": "".join(group_text)})
        group_id = None
        group_text = []
        group_tool_calls = []

    for record in records:
        if record.get("isSidechain"):
            continue
        record_type = record.get("type")

        if record_type == "assistant":
            message = record.get("message") or {}
            message_id = message.get("id")
            if message_id is None or message_id != group_id:
                flush_group()
                group_id = message_id
            for block in message.get("content") or []:
                block_type = block.get("type")
                if block_type == "text" and block.get("text"):
                    group_text.append(block["text"])
                elif block_type == "tool_use":
                    group_tool_calls.append(
                        {
                            "id": block["id"],
                            "type": "function",
                            "function": {
                                "name": block.get("name", ""),
                                "arguments": json.dumps(block.get("input") or {}),
                            },
                        }
                    )
                # thinking, image, and anything else: no OpenAI equivalent.
            continue

        flush_group()

        if record_type != "user":
            continue

        content = (record.get("message") or {}).get("content")
        if isinstance(content, str):
            if content:
                messages.append({"role": "user", "content": content})
        elif isinstance(content, list):
            for block in content:
                block_type = block.get("type")
                if block_type == "tool_result":
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": block["tool_use_id"],
                            "content": _flatten_tool_result_content(block.get("content")),
                        }
                    )
                elif block_type == "text" and block.get("text"):
                    messages.append({"role": "user", "content": block["text"]})

    flush_group()
    return messages
