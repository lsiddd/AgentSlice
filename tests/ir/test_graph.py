import pytest

from agentslice.errors import TraceValidationError
from agentslice.ir.events import EventType, TraceEvent
from agentslice.ir.graph import CausalEdge, CausalGraph, build_causal_graph


def make_event(
    id: str,
    seq: int,
    type: EventType = EventType.STATE_UPDATE,
    **kwargs: object,
) -> TraceEvent:
    return TraceEvent(id=id, seq=seq, type=type, **kwargs)  # type: ignore[arg-type]


def test_empty_events_produce_an_empty_graph() -> None:
    graph = build_causal_graph([])
    assert graph.events == {}
    assert graph.order == []
    assert graph.edges == []
    assert graph.facts == {}
    assert graph.unresolved_reads == set()


def test_events_are_reordered_by_seq_regardless_of_input_order() -> None:
    later = make_event("a", seq=2)
    earlier = make_event("b", seq=1)
    graph = build_causal_graph([later, earlier])
    assert graph.order == ["b", "a"]


def test_duplicate_seq_raises_trace_validation_error() -> None:
    first = make_event("a", seq=1)
    second = make_event("b", seq=1)
    with pytest.raises(TraceValidationError):
        build_causal_graph([first, second])


def test_write_then_read_creates_a_causal_edge() -> None:
    writer = make_event("w", seq=1, writes=frozenset({"x"}), outputs={"x": 1})
    reader = make_event("r", seq=2, reads=frozenset({"x"}))
    graph = build_causal_graph([writer, reader])
    assert graph.edges == [CausalEdge(from_event_id="w", to_event_id="r", fact_key="x")]
    assert graph.ancestors("r") == {"w"}


def test_fact_never_read_has_no_outgoing_edge() -> None:
    writer = make_event("w", seq=1, writes=frozenset({"x"}), outputs={"x": 1})
    graph = build_causal_graph([writer])
    assert graph.edges == []
    latest = graph.latest_fact("x")
    assert latest is not None
    assert latest.origin_event_id == "w"


def test_read_of_never_written_key_is_unresolved_not_an_error() -> None:
    reader = make_event("r", seq=1, reads=frozenset({"missing"}))
    graph = build_causal_graph([reader])
    assert graph.edges == []
    assert ("r", "missing") in graph.unresolved_reads


def test_latest_fact_of_unknown_key_is_none() -> None:
    graph = build_causal_graph([])
    assert graph.latest_fact("nope") is None


def test_supersession_chain_tracks_every_version() -> None:
    e1 = make_event("e1", seq=1, writes=frozenset({"status"}), outputs={"status": "pending"})
    e2 = make_event("e2", seq=2, writes=frozenset({"status"}), outputs={"status": "running"})
    e3 = make_event("e3", seq=3, writes=frozenset({"status"}), outputs={"status": "done"})
    graph = build_causal_graph([e1, e2, e3])
    versions = graph.facts["status"]
    assert [f.origin_event_id for f in versions] == ["e1", "e2", "e3"]
    assert versions[0].supersedes is None
    assert versions[1].supersedes == "e1"
    assert versions[2].supersedes == "e2"
    latest = graph.latest_fact("status")
    assert latest is not None
    assert latest.origin_event_id == "e3"


def test_ancestors_terminates_on_a_manually_built_cycle() -> None:
    graph = CausalGraph(
        events={"a": make_event("a", seq=1), "b": make_event("b", seq=2)},
        order=["a", "b"],
        edges=[
            CausalEdge(from_event_id="a", to_event_id="b", fact_key="x"),
            CausalEdge(from_event_id="b", to_event_id="a", fact_key="y"),
        ],
    )
    assert graph.ancestors("a") == {"a", "b"}


def test_multiple_user_goals_are_all_preserved() -> None:
    first_goal = make_event("g1", seq=1, type=EventType.USER_GOAL)
    second_goal = make_event("g2", seq=2, type=EventType.USER_GOAL)
    graph = build_causal_graph([first_goal, second_goal])
    assert graph.order == ["g1", "g2"]
    assert graph.events_in_order() == [first_goal, second_goal]
