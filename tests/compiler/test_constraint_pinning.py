from agentslice.compiler.base import CompileContext
from agentslice.compiler.constraint_pinning import ConstraintPinningPass
from agentslice.ir.events import EventType, TraceEvent
from agentslice.ir.graph import build_causal_graph


def test_constraint_event_gets_pinned() -> None:
    constraint = TraceEvent(
        id="c1", seq=0, type=EventType.CONSTRAINT, outputs={"content": "never delete prod"}
    )
    graph = build_causal_graph([constraint])
    outcome = ConstraintPinningPass().apply(graph, CompileContext())
    assert outcome.graph.events["c1"].pinned is True
    assert outcome.report.pinned_event_ids == ["c1"]


def test_manually_pinned_non_constraint_event_is_respected() -> None:
    event = TraceEvent(id="e1", seq=0, type=EventType.STATE_UPDATE, pinned=True)
    graph = build_causal_graph([event])
    outcome = ConstraintPinningPass().apply(graph, CompileContext())
    assert outcome.graph.events["e1"].pinned is True
    assert outcome.report.pinned_event_ids == ["e1"]


def test_no_constraints_is_a_no_op() -> None:
    event = TraceEvent(id="e1", seq=0, type=EventType.STATE_UPDATE)
    graph = build_causal_graph([event])
    outcome = ConstraintPinningPass().apply(graph, CompileContext())
    assert outcome.graph.events["e1"].pinned is False
    assert outcome.report.pinned_event_ids == []
    assert outcome.report.events_before == outcome.report.events_after == 1
