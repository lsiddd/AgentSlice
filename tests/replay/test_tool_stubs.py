import pytest

from agentslice.errors import MissingToolResultError
from agentslice.ir.events import EventType, TraceEvent
from agentslice.replay.tool_stubs import fill_pending_tool_results


def _tool_call(call_id: str) -> TraceEvent:
    return TraceEvent(
        id=call_id,
        seq=0,
        type=EventType.TOOL_CALL,
        tool_name="get_weather",
        inputs={"city": "nyc"},
        writes=frozenset({f"tool_call:{call_id}"}),
    )


def _tool_result(call_id: str, seq: int) -> TraceEvent:
    return TraceEvent(
        id=f"result_{call_id}",
        seq=seq,
        type=EventType.TOOL_RESULT,
        outputs={"temp": 72},
        reads=frozenset({f"tool_call:{call_id}"}),
        writes=frozenset({f"tool_result:{call_id}.temp"}),
    )


def test_returns_events_unchanged_when_every_call_is_answered() -> None:
    events = [_tool_call("call_1"), _tool_result("call_1", seq=1)]
    result = fill_pending_tool_results(events, original_events=[])
    assert result == events


def test_pulls_missing_result_from_original_trace() -> None:
    call = _tool_call("call_1")
    original_result = _tool_result("call_1", seq=1)
    result = fill_pending_tool_results([call], original_events=[call, original_result])
    assert result == [call, original_result]


def test_raises_when_original_trace_also_lacks_the_result() -> None:
    call = _tool_call("call_1")
    with pytest.raises(MissingToolResultError):
        fill_pending_tool_results([call], original_events=[call])


def test_result_is_inserted_in_seq_order() -> None:
    call_1 = _tool_call("call_1")
    call_2 = TraceEvent(
        id="call_2",
        seq=1,
        type=EventType.TOOL_CALL,
        tool_name="get_time",
        writes=frozenset({"tool_call:call_2"}),
    )
    result_1 = _tool_result("call_1", seq=2)
    result_2 = TraceEvent(
        id="result_call_2",
        seq=3,
        type=EventType.TOOL_RESULT,
        outputs={"utc": "12:00"},
        reads=frozenset({"tool_call:call_2"}),
        writes=frozenset({"tool_result:call_2.utc"}),
    )
    original = [call_1, call_2, result_1, result_2]

    result = fill_pending_tool_results([call_1, result_1, call_2], original_events=original)

    assert [e.id for e in result] == ["call_1", "call_2", "result_call_1", "result_call_2"]
