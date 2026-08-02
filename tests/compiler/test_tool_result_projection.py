from agentslice.compiler.base import CompileContext
from agentslice.compiler.tool_result_projection import ToolResultProjectionPass
from agentslice.ir.events import EventType, TraceEvent
from agentslice.ir.graph import build_causal_graph


def test_projects_down_to_only_read_fields() -> None:
    result = TraceEvent(
        id="call_1",
        seq=0,
        type=EventType.TOOL_RESULT,
        outputs={"title": "bug", "body": "long text", "labels": "bug"},
        writes=frozenset(
            {"tool_result:call_1.title", "tool_result:call_1.body", "tool_result:call_1.labels"}
        ),
    )
    reader = TraceEvent(
        id="reader",
        seq=1,
        type=EventType.MODEL_MESSAGE,
        reads=frozenset({"tool_result:call_1.title"}),
        outputs={"content": "..."},
    )
    graph = build_causal_graph([result, reader])
    outcome = ToolResultProjectionPass().apply(graph, CompileContext())
    projected = outcome.graph.events["call_1"]
    assert projected.outputs == {"title": "bug"}
    assert projected.writes == frozenset({"tool_result:call_1.title"})
    assert outcome.report.modified_event_ids == ["call_1"]


def test_no_op_when_all_fields_are_read() -> None:
    result = TraceEvent(
        id="call_1",
        seq=0,
        type=EventType.TOOL_RESULT,
        outputs={"a": 1, "b": 2},
        writes=frozenset({"tool_result:call_1.a", "tool_result:call_1.b"}),
    )
    reader = TraceEvent(
        id="reader",
        seq=1,
        type=EventType.MODEL_MESSAGE,
        reads=frozenset({"tool_result:call_1.a", "tool_result:call_1.b"}),
    )
    graph = build_causal_graph([result, reader])
    outcome = ToolResultProjectionPass().apply(graph, CompileContext())
    assert outcome.graph.events["call_1"].outputs == {"a": 1, "b": 2}
    assert outcome.report.modified_event_ids == []


def test_anchor_event_is_kept_whole_even_with_no_readers() -> None:
    result = TraceEvent(
        id="call_1",
        seq=0,
        type=EventType.TOOL_RESULT,
        outputs={"a": 1, "b": 2},
        writes=frozenset({"tool_result:call_1.a", "tool_result:call_1.b"}),
    )
    graph = build_causal_graph([result])
    outcome = ToolResultProjectionPass().apply(graph, CompileContext())
    assert outcome.graph.events["call_1"].outputs == {"a": 1, "b": 2}


def test_pinned_tool_result_is_kept_whole() -> None:
    result = TraceEvent(
        id="call_1",
        seq=0,
        type=EventType.TOOL_RESULT,
        outputs={"a": 1, "b": 2},
        writes=frozenset({"tool_result:call_1.a", "tool_result:call_1.b"}),
        pinned=True,
    )
    other = TraceEvent(id="other", seq=1, type=EventType.STATE_UPDATE)
    graph = build_causal_graph([result, other])
    outcome = ToolResultProjectionPass().apply(graph, CompileContext())
    assert outcome.graph.events["call_1"].outputs == {"a": 1, "b": 2}


def test_opaque_tool_result_is_untouched() -> None:
    result = TraceEvent(
        id="call_1",
        seq=0,
        type=EventType.TOOL_RESULT,
        outputs={"nested": {"a": 1}},
        writes=frozenset({"tool_result:call_1"}),
    )
    other = TraceEvent(id="other", seq=1, type=EventType.STATE_UPDATE)
    graph = build_causal_graph([result, other])
    outcome = ToolResultProjectionPass().apply(graph, CompileContext())
    assert outcome.graph.events["call_1"].outputs == {"nested": {"a": 1}}
    assert outcome.report.modified_event_ids == []
