from agentslice.compiler.base import CompileContext
from agentslice.compiler.duplicate_result_elimination import DuplicateResultEliminationPass
from agentslice.ir.events import EventType, TraceEvent
from agentslice.ir.graph import build_causal_graph


def make_call(id: str, seq: int, tool_name: str = "get_weather", **kwargs: object) -> TraceEvent:
    return TraceEvent(
        id=id,
        seq=seq,
        type=EventType.TOOL_CALL,
        tool_name=tool_name,
        inputs={"city": "nyc"},
        writes=frozenset({f"tool_call:{id}"}),
        **kwargs,  # type: ignore[arg-type]
    )


def make_result(id: str, seq: int, call_id: str, **kwargs: object) -> TraceEvent:
    return TraceEvent(
        id=id,
        seq=seq,
        type=EventType.TOOL_RESULT,
        outputs={"temp": 70},
        reads=frozenset({f"tool_call:{call_id}"}),
        **kwargs,  # type: ignore[arg-type]
    )


def _anchor(seq: int) -> TraceEvent:
    return TraceEvent(id="anchor", seq=seq, type=EventType.STATE_UPDATE)


def test_later_duplicate_with_no_side_effects_is_dropped() -> None:
    events = [
        make_call("c1", 0),
        make_result("r1", 1, "c1"),
        make_call("c2", 2),
        make_result("r2", 3, "c2"),
        _anchor(4),
    ]
    graph = build_causal_graph(events)
    outcome = DuplicateResultEliminationPass().apply(graph, CompileContext())
    assert set(outcome.graph.events) == {"c1", "r1", "anchor"}
    assert set(outcome.report.removed_event_ids) == {"c2", "r2"}


def test_duplicate_with_side_effects_is_redacted_not_dropped() -> None:
    events = [
        make_call("c1", 0),
        make_result("r1", 1, "c1", side_effects=True),
        make_call("c2", 2),
        make_result("r2", 3, "c2", side_effects=True),
        _anchor(4),
    ]
    graph = build_causal_graph(events)
    outcome = DuplicateResultEliminationPass().apply(graph, CompileContext())
    assert set(outcome.graph.events) == {"c1", "r1", "c2", "r2", "anchor"}
    assert outcome.report.modified_event_ids == ["r2"]
    assert outcome.graph.events["r2"].outputs == {
        "_redacted": "duplicate of an earlier identical tool result"
    }


def test_calls_with_different_arguments_are_not_treated_as_duplicates() -> None:
    call1 = TraceEvent(
        id="c1",
        seq=0,
        type=EventType.TOOL_CALL,
        tool_name="get_weather",
        inputs={"city": "nyc"},
        writes=frozenset({"tool_call:c1"}),
    )
    call2 = TraceEvent(
        id="c2",
        seq=2,
        type=EventType.TOOL_CALL,
        tool_name="get_weather",
        inputs={"city": "sf"},
        writes=frozenset({"tool_call:c2"}),
    )
    events = [call1, make_result("r1", 1, "c1"), call2, make_result("r2", 3, "c2")]
    graph = build_causal_graph(events)
    outcome = DuplicateResultEliminationPass().apply(graph, CompileContext())
    assert set(outcome.graph.events) == {"c1", "r1", "c2", "r2"}
    assert outcome.report.removed_event_ids == []


def test_same_inputs_but_different_results_are_not_treated_as_duplicates() -> None:
    events = [
        make_call("c1", 0),
        make_result("r1", 1, "c1"),
        make_call("c2", 2),
        TraceEvent(
            id="r2",
            seq=3,
            type=EventType.TOOL_RESULT,
            outputs={"temp": 71},
            reads=frozenset({"tool_call:c2"}),
        ),
    ]
    graph = build_causal_graph(events)
    outcome = DuplicateResultEliminationPass().apply(graph, CompileContext())
    assert set(outcome.graph.events) == {"c1", "r1", "c2", "r2"}
    assert outcome.report.removed_event_ids == []


def test_duplicate_that_is_the_anchor_is_left_untouched() -> None:
    events = [
        make_call("c1", 0),
        make_result("r1", 1, "c1"),
        make_call("c2", 2),
        make_result("r2", 3, "c2"),
    ]
    graph = build_causal_graph(events)
    outcome = DuplicateResultEliminationPass().apply(graph, CompileContext(anchor_event_id="r2"))
    assert set(outcome.graph.events) == {"c1", "r1", "c2", "r2"}
    assert outcome.report.removed_event_ids == []


def test_pinned_duplicate_is_left_untouched() -> None:
    events = [
        make_call("c1", 0),
        make_result("r1", 1, "c1"),
        make_call("c2", 2, pinned=True),
        make_result("r2", 3, "c2"),
        _anchor(4),
    ]
    graph = build_causal_graph(events)
    outcome = DuplicateResultEliminationPass().apply(graph, CompileContext())
    assert set(outcome.graph.events) == {"c1", "r1", "c2", "r2", "anchor"}
    assert outcome.report.removed_event_ids == []


def test_duplicate_result_with_an_external_reader_is_left_untouched() -> None:
    events = [
        make_call("c1", 0),
        make_result("r1", 1, "c1"),
        make_call("c2", 2),
        TraceEvent(
            id="r2",
            seq=3,
            type=EventType.TOOL_RESULT,
            outputs={"temp": 70},
            reads=frozenset({"tool_call:c2"}),
            writes=frozenset({"tool_result:c2"}),
        ),
        TraceEvent(
            id="reader",
            seq=4,
            type=EventType.MODEL_MESSAGE,
            reads=frozenset({"tool_result:c2"}),
            outputs={"content": "..."},
        ),
    ]
    graph = build_causal_graph(events)
    outcome = DuplicateResultEliminationPass().apply(graph, CompileContext())
    assert set(outcome.graph.events) == {"c1", "r1", "c2", "r2", "reader"}
    assert outcome.report.removed_event_ids == []


def test_duplicate_is_dropped_when_its_reader_also_depends_on_the_baseline() -> None:
    events = [
        make_call("c1", 0),
        make_result("r1", 1, "c1", writes=frozenset({"tool_result:c1"})),
        make_call("c2", 2),
        make_result("r2", 3, "c2", writes=frozenset({"tool_result:c2"})),
        TraceEvent(
            id="c3",
            seq=4,
            type=EventType.TOOL_CALL,
            tool_name="log_check",
            inputs={"note": 70},
            reads=frozenset({"tool_result:c1", "tool_result:c2"}),
        ),
    ]
    graph = build_causal_graph(events)
    outcome = DuplicateResultEliminationPass().apply(graph, CompileContext())
    assert set(outcome.graph.events) == {"c1", "r1", "c3"}
    assert set(outcome.report.removed_event_ids) == {"c2", "r2"}


def test_three_identical_calls_keep_only_the_first() -> None:
    events = [
        make_call("c1", 0),
        make_result("r1", 1, "c1"),
        make_call("c2", 2),
        make_result("r2", 3, "c2"),
        make_call("c3", 4),
        make_result("r3", 5, "c3"),
        _anchor(6),
    ]
    graph = build_causal_graph(events)
    outcome = DuplicateResultEliminationPass().apply(graph, CompileContext())
    assert set(outcome.graph.events) == {"c1", "r1", "anchor"}


def test_call_with_no_result_is_ignored_without_crashing() -> None:
    events = [make_call("c1", 0), make_call("c2", 1)]
    graph = build_causal_graph(events)
    outcome = DuplicateResultEliminationPass().apply(graph, CompileContext())
    assert set(outcome.graph.events) == {"c1", "c2"}
    assert outcome.report.removed_event_ids == []


def test_single_call_is_a_no_op() -> None:
    events = [make_call("c1", 0), make_result("r1", 1, "c1")]
    graph = build_causal_graph(events)
    outcome = DuplicateResultEliminationPass().apply(graph, CompileContext())
    assert set(outcome.graph.events) == {"c1", "r1"}
    assert outcome.report.removed_event_ids == []
