"""Comparing a replayed next action to what the original trace actually did next.

v0.2 scope, matching the roadmap's stated limitation: equivalence here is
exact, not semantic. Two tool calls match only if every call's tool name
and JSON-normalized arguments match exactly (order-independent, since
parallel calls carry no meaningful order); two non-call messages match
only if both are, or both aren't, a final text answer. Differently-phrased
but semantically equivalent text, or a call with functionally-equivalent
but textually different arguments (e.g. a reordered list), are *not*
considered equal. A real semantic comparator (embeddings or an LLM judge)
is out of scope here and left to a future milestone.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from agentslice.errors import UnknownAnchorError
from agentslice.ir.events import EventType, TraceEvent
from agentslice.recording.openai_adapter import to_openai_messages


def extract_next_recorded_action(
    events: Sequence[TraceEvent], anchor_id: str
) -> dict[str, Any] | None:
    """Return the OpenAI-style assistant message for what the agent did right after ``anchor_id``.

    This is the ground truth :func:`next_action_equivalence` compares a
    replayed action against: whatever the original, uncompiled trace
    recorded as the very next agent action (one or more consecutive
    ``tool_call`` events, or a single ``model_message``) following the
    anchor. Returns ``None`` when there is nothing comparable — the
    anchor was the last event, or the next event is a new ``user_goal``
    rather than an agent action.

    Raises:
        UnknownAnchorError: No event in ``events`` has id ``anchor_id``.
    """
    ordered = sorted(events, key=lambda event: event.seq)
    anchor_index = next((i for i, event in enumerate(ordered) if event.id == anchor_id), None)
    if anchor_index is None:
        raise UnknownAnchorError(f"no event with id {anchor_id!r}")

    after = ordered[anchor_index + 1 :]
    if not after:
        return None

    first = after[0]
    if first.type is EventType.MODEL_MESSAGE:
        group = [first]
    elif first.type is EventType.TOOL_CALL:
        group = []
        for event in after:
            if event.type is not EventType.TOOL_CALL:
                break
            group.append(event)
    else:
        return None

    messages = to_openai_messages(group)
    return messages[0] if messages else None


def next_action_equivalence(original: dict[str, Any], replayed: dict[str, Any]) -> bool:
    """Compare two OpenAI-style assistant messages for exact next-action equivalence."""
    original_calls = _normalize_tool_calls(original.get("tool_calls") or [])
    replayed_calls = _normalize_tool_calls(replayed.get("tool_calls") or [])
    if original_calls or replayed_calls:
        return original_calls == replayed_calls
    return bool(original.get("content")) == bool(replayed.get("content"))


def _normalize_tool_calls(tool_calls: list[dict[str, Any]]) -> frozenset[tuple[str, str]]:
    normalized: set[tuple[str, str]] = set()
    for call in tool_calls:
        function = call.get("function", {})
        name = function.get("name", "")
        raw_arguments = function.get("arguments") or "{}"
        try:
            arguments = json.dumps(json.loads(raw_arguments), sort_keys=True)
        except json.JSONDecodeError:
            arguments = raw_arguments
        normalized.add((name, arguments))
    return frozenset(normalized)
