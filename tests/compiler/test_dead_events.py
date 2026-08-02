from agentslice.compiler.base import CompileContext
from agentslice.compiler.dead_events import DeadEventsPass
from agentslice.ir.events import EventType, TraceEvent
from agentslice.ir.graph import build_causal_graph


def test_fully_connected_graph_is_a_no_op() -> None:
    writer = TraceEvent(
        id="w", seq=0, type=EventType.TOOL_RESULT, writes=frozenset({"x"}), outputs={"x": 1}
    )
    reader = TraceEvent(
        id="r", seq=1, type=EventType.TOOL_CALL, reads=frozenset({"x"}), inputs={"x": 1}
    )
    graph = build_causal_graph([writer, reader])
    outcome = DeadEventsPass().apply(graph, CompileContext())
    assert set(outcome.graph.events) == {"w", "r"}
    assert outcome.report.removed_event_ids == []


def test_disconnected_island_is_removed() -> None:
    writer = TraceEvent(
        id="w", seq=0, type=EventType.TOOL_RESULT, writes=frozenset({"x"}), outputs={"x": 1}
    )
    island = TraceEvent(id="island", seq=1, type=EventType.STATE_UPDATE)
    anchor = TraceEvent(
        id="anchor", seq=2, type=EventType.TOOL_CALL, reads=frozenset({"x"}), inputs={"x": 1}
    )
    graph = build_causal_graph([writer, island, anchor])
    outcome = DeadEventsPass().apply(graph, CompileContext())
    assert set(outcome.graph.events) == {"w", "anchor"}
    assert outcome.report.removed_event_ids == ["island"]


def test_pinned_but_unreachable_event_is_kept() -> None:
    pinned = TraceEvent(id="pinned", seq=0, type=EventType.CONSTRAINT, pinned=True)
    anchor = TraceEvent(id="anchor", seq=1, type=EventType.STATE_UPDATE)
    graph = build_causal_graph([pinned, anchor])
    outcome = DeadEventsPass().apply(graph, CompileContext())
    assert set(outcome.graph.events) == {"pinned", "anchor"}


def test_pinned_event_after_the_anchor_is_dropped() -> None:
    past_constraint = TraceEvent(id="past", seq=0, type=EventType.CONSTRAINT, pinned=True)
    anchor = TraceEvent(id="anchor", seq=1, type=EventType.STATE_UPDATE)
    future_constraint = TraceEvent(id="future", seq=2, type=EventType.CONSTRAINT, pinned=True)
    graph = build_causal_graph([past_constraint, anchor, future_constraint])
    outcome = DeadEventsPass().apply(graph, CompileContext(anchor_event_id="anchor"))
    assert set(outcome.graph.events) == {"past", "anchor"}
    assert outcome.report.removed_event_ids == ["future"]


def test_single_event_graph_is_kept() -> None:
    only = TraceEvent(id="only", seq=0, type=EventType.USER_GOAL)
    graph = build_causal_graph([only])
    outcome = DeadEventsPass().apply(graph, CompileContext())
    assert set(outcome.graph.events) == {"only"}


def test_empty_graph_is_a_no_op() -> None:
    graph = build_causal_graph([])
    outcome = DeadEventsPass().apply(graph, CompileContext())
    assert outcome.graph.events == {}
    assert outcome.report.events_before == 0
