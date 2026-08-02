import pytest
from pydantic import ValidationError

from agentslice.ir.events import EventType, TraceEvent


def test_minimal_construction_has_sane_defaults() -> None:
    event = TraceEvent(id="e1", seq=0, type=EventType.USER_GOAL)
    assert event.reads == frozenset()
    assert event.writes == frozenset()
    assert event.side_effects is False
    assert event.pinned is False
    assert event.metadata == {}
    assert event.timestamp is None


def test_extra_fields_are_rejected() -> None:
    with pytest.raises(ValidationError):
        TraceEvent(id="e1", seq=0, type=EventType.USER_GOAL, unexpected="nope")


def test_event_type_accepts_string_value() -> None:
    event = TraceEvent(id="e1", seq=0, type="tool_call")
    assert event.type is EventType.TOOL_CALL
