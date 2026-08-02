from agentslice.compiler.base import CompileContext
from agentslice.compiler.superseded_state import SupersededStatePass
from agentslice.ir.events import EventType, TraceEvent
from agentslice.ir.graph import build_causal_graph


def make_writer(id: str, seq: int, value: str, *, side_effects: bool = False) -> TraceEvent:
    return TraceEvent(
        id=id,
        seq=seq,
        type=EventType.STATE_UPDATE,
        writes=frozenset({"status"}),
        outputs={"status": value},
        side_effects=side_effects,
    )


def test_three_writes_to_the_same_key_keeps_only_the_last() -> None:
    events = [
        make_writer("e1", 0, "pending"),
        make_writer("e2", 1, "running"),
        make_writer("e3", 2, "done"),
    ]
    graph = build_causal_graph(events)
    outcome = SupersededStatePass().apply(graph, CompileContext())
    assert set(outcome.graph.events) == {"e3"}
    assert outcome.report.removed_event_ids == ["e1", "e2"]


def test_side_effect_event_is_kept_but_redacted() -> None:
    events = [
        make_writer("e1", 0, "pending", side_effects=True),
        make_writer("e2", 1, "done"),
    ]
    graph = build_causal_graph(events)
    outcome = SupersededStatePass().apply(graph, CompileContext())
    assert set(outcome.graph.events) == {"e1", "e2"}
    assert outcome.graph.events["e1"].outputs == {"_redacted": "superseded value omitted"}
    assert outcome.report.modified_event_ids == ["e1"]


def test_superseded_pinned_constraint_is_kept_unchanged() -> None:
    old_constraint = TraceEvent(
        id="c1",
        seq=0,
        type=EventType.CONSTRAINT,
        writes=frozenset({"policy"}),
        outputs={"policy": "read-only"},
        pinned=True,
        metadata={"source": "system"},
    )
    new_constraint = TraceEvent(
        id="c2",
        seq=1,
        type=EventType.CONSTRAINT,
        writes=frozenset({"policy"}),
        outputs={"policy": "read-write"},
        pinned=True,
    )
    graph = build_causal_graph([old_constraint, new_constraint])
    outcome = SupersededStatePass().apply(graph, CompileContext())
    assert set(outcome.graph.events) == {"c1", "c2"}
    assert outcome.graph.events["c1"] == old_constraint
    assert outcome.report.removed_event_ids == []
    assert outcome.report.modified_event_ids == []


def test_superseded_anchor_is_kept_unchanged() -> None:
    anchor = make_writer("anchor", 0, "pending")
    latest = make_writer("latest", 1, "done")
    graph = build_causal_graph([anchor, latest])

    outcome = SupersededStatePass().apply(
        graph,
        CompileContext(anchor_event_id=anchor.id),
    )

    assert outcome.graph.events["anchor"] == anchor
    assert outcome.report.removed_event_ids == []
    assert outcome.report.modified_event_ids == []


def test_writer_with_a_surviving_reader_is_kept_despite_being_superseded() -> None:
    a = make_writer("a", 0, "1")
    b = TraceEvent(
        id="b",
        seq=1,
        type=EventType.STATE_UPDATE,
        reads=frozenset({"status"}),
        writes=frozenset({"derived"}),
        outputs={"derived": "1"},
    )
    c = make_writer("c", 2, "2")
    d = TraceEvent(
        id="d",
        seq=3,
        type=EventType.STATE_UPDATE,
        reads=frozenset({"status", "derived"}),
    )
    graph = build_causal_graph([a, b, c, d])
    outcome = SupersededStatePass().apply(graph, CompileContext())
    assert set(outcome.graph.events) == {"a", "b", "c", "d"}
    assert outcome.report.removed_event_ids == []


def test_event_that_never_writes_a_fact_is_untouched() -> None:
    event = TraceEvent(id="e1", seq=0, type=EventType.USER_GOAL, outputs={"content": "hi"})
    graph = build_causal_graph([event])
    outcome = SupersededStatePass().apply(graph, CompileContext())
    assert set(outcome.graph.events) == {"e1"}
    assert outcome.report.removed_event_ids == []
    assert outcome.report.modified_event_ids == []
