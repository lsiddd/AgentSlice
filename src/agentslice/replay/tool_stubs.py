"""Filling in missing tool results during replay, without executing anything for real."""

from __future__ import annotations

from collections.abc import Sequence

from agentslice.errors import MissingToolResultError
from agentslice.ir.events import EventType, TraceEvent
from agentslice.recording.openai_adapter import tool_call_id_of


def fill_pending_tool_results(
    events: Sequence[TraceEvent], original_events: Sequence[TraceEvent]
) -> list[TraceEvent]:
    """Return ``events`` plus a ``tool_result`` for every ``tool_call`` in it that has none.

    Replay never executes a tool for real: a ``tool_call`` that survived
    compilation or forking without its answer (its own result was pruned,
    or the fork point falls right after the call and before its result)
    gets that answer pulled verbatim from ``original_events`` — the full
    trace ``events`` was derived from — instead. This is what lets
    :func:`~agentslice.recording.openai_adapter.to_openai_messages` always
    produce a message array with every ``tool_calls`` entry answered.

    Returned events are sorted by ``seq``, since a filled-in result did
    not necessarily appear at the end of ``events``.

    Raises:
        MissingToolResultError: A pending ``tool_call`` has no answer in
            ``original_events`` either — it was genuinely never resolved,
            even in the source trace.
    """
    answered_ids = {
        tool_call_id_of(event) for event in events if event.type is EventType.TOOL_RESULT
    }
    pending_ids = {event.id for event in events if event.type is EventType.TOOL_CALL} - answered_ids
    if not pending_ids:
        return list(events)

    results_by_call_id = {
        tool_call_id_of(event): event
        for event in original_events
        if event.type is EventType.TOOL_RESULT
    }

    filled = list(events)
    for call_id in pending_ids:
        result = results_by_call_id.get(call_id)
        if result is None:
            raise MissingToolResultError(
                f"tool_call {call_id!r} has no recorded tool_result in the original trace"
            )
        filled.append(result)

    return sorted(filled, key=lambda event: event.seq)
