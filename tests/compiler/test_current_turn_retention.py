from agentslice.compiler.base import CompileContext
from agentslice.compiler.current_turn_retention import CurrentTurnRetentionPass
from agentslice.ir.events import EventType, TraceEvent
from agentslice.ir.graph import build_causal_graph


def test_unreferenced_call_in_the_current_turn_is_pinned() -> None:
    goal = TraceEvent(id="goal", seq=0, type=EventType.USER_GOAL)
    orient_call = TraceEvent(id="pwd_call", seq=1, type=EventType.TOOL_CALL, tool_name="pwd")
    orient_result = TraceEvent(
        id="pwd_result", seq=2, type=EventType.TOOL_RESULT, outputs={"cwd": "/home"}
    )
    anchor = TraceEvent(id="anchor", seq=3, type=EventType.TOOL_CALL, tool_name="ls")
    graph = build_causal_graph([goal, orient_call, orient_result, anchor])
    outcome = CurrentTurnRetentionPass().apply(graph, CompileContext(anchor_event_id="anchor"))
    assert set(outcome.report.pinned_event_ids) == {"goal", "pwd_call", "pwd_result", "anchor"}
    for event in outcome.graph.events.values():
        assert event.pinned is True


def test_earlier_turn_is_left_unpinned() -> None:
    first_goal = TraceEvent(id="goal1", seq=0, type=EventType.USER_GOAL)
    stray_call = TraceEvent(id="stray", seq=1, type=EventType.TOOL_CALL, tool_name="pwd")
    second_goal = TraceEvent(id="goal2", seq=2, type=EventType.USER_GOAL)
    anchor = TraceEvent(id="anchor", seq=3, type=EventType.TOOL_CALL, tool_name="ls")
    graph = build_causal_graph([first_goal, stray_call, second_goal, anchor])
    outcome = CurrentTurnRetentionPass().apply(graph, CompileContext(anchor_event_id="anchor"))
    assert set(outcome.report.pinned_event_ids) == {"goal2", "anchor"}
    assert outcome.graph.events["stray"].pinned is False
    assert outcome.graph.events["goal1"].pinned is False


def test_already_pinned_event_is_reported_but_untouched() -> None:
    constraint = TraceEvent(
        id="c1", seq=0, type=EventType.CONSTRAINT, pinned=True, outputs={"content": "policy"}
    )
    goal = TraceEvent(id="goal", seq=1, type=EventType.USER_GOAL)
    anchor = TraceEvent(id="anchor", seq=2, type=EventType.TOOL_CALL, tool_name="ls")
    graph = build_causal_graph([constraint, goal, anchor])
    outcome = CurrentTurnRetentionPass().apply(graph, CompileContext(anchor_event_id="anchor"))
    assert set(outcome.report.pinned_event_ids) == {"c1", "goal", "anchor"}


def test_no_anchor_and_empty_graph_is_a_no_op() -> None:
    graph = build_causal_graph([])
    outcome = CurrentTurnRetentionPass().apply(graph, CompileContext())
    assert outcome.graph.events == {}
    assert outcome.report.pinned_event_ids == []


def test_falls_back_to_the_most_recent_event_as_anchor() -> None:
    goal = TraceEvent(id="goal", seq=0, type=EventType.USER_GOAL)
    call = TraceEvent(id="call", seq=1, type=EventType.TOOL_CALL, tool_name="pwd")
    graph = build_causal_graph([goal, call])
    outcome = CurrentTurnRetentionPass().apply(graph, CompileContext())
    assert set(outcome.report.pinned_event_ids) == {"goal", "call"}
