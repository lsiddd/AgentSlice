from agentslice.compiler.tokens import estimate_event_tokens, estimate_graph_tokens, estimate_tokens
from agentslice.ir.events import EventType, TraceEvent


def test_empty_text_costs_zero_tokens() -> None:
    assert estimate_tokens("") == 0


def test_short_text_costs_at_least_one_token() -> None:
    assert estimate_tokens("hi") == 1


def test_longer_text_scales_roughly_by_four_chars_per_token() -> None:
    assert estimate_tokens("a" * 40) == 10


def test_event_tokens_ignore_bookkeeping_fields() -> None:
    minimal = TraceEvent(id="e1", seq=0, type=EventType.STATE_UPDATE)
    with_metadata = TraceEvent(
        id="e1",
        seq=0,
        type=EventType.STATE_UPDATE,
        metadata={"debug": "x" * 1000},
    )
    assert estimate_event_tokens(minimal) == estimate_event_tokens(with_metadata)


def test_event_tokens_grow_with_outputs() -> None:
    small = TraceEvent(id="e1", seq=0, type=EventType.TOOL_RESULT, outputs={"a": "x"})
    large = TraceEvent(id="e1", seq=0, type=EventType.TOOL_RESULT, outputs={"a": "x" * 1000})
    assert estimate_event_tokens(large) > estimate_event_tokens(small)


def test_graph_tokens_sum_over_events() -> None:
    events = [
        TraceEvent(id="a", seq=0, type=EventType.STATE_UPDATE, outputs={"x": "abcd"}),
        TraceEvent(id="b", seq=1, type=EventType.STATE_UPDATE, outputs={"x": "abcd"}),
    ]
    assert estimate_graph_tokens(events) == 2 * estimate_event_tokens(events[0])


def test_empty_graph_costs_zero_tokens() -> None:
    assert estimate_graph_tokens([]) == 0
