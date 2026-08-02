import pytest

from agentslice.compiler.base import CompilationReport, CompileContext, Pass, PassOutcome
from agentslice.compiler.pipeline import DEFAULT_PASSES, Pipeline, compile_graph
from agentslice.errors import BudgetNotSatisfiableError
from agentslice.ir.events import EventType, TraceEvent
from agentslice.ir.graph import CausalGraph, build_causal_graph
from agentslice.recording.claude_code_adapter import from_claude_code_transcript
from agentslice.recording.codex_adapter import from_codex_rollout
from agentslice.recording.openai_adapter import from_openai_messages


class _NoopPass(Pass):
    def __init__(self, name: str) -> None:
        self.name = name

    def apply(self, graph: CausalGraph, ctx: CompileContext) -> PassOutcome:
        events = graph.events_in_order()
        report = CompilationReport(
            pass_name=self.name,
            events_before=len(events),
            events_after=len(events),
            tokens_before=0,
            tokens_after=0,
        )
        return PassOutcome(graph=graph, ctx=ctx, report=report)


def test_custom_pass_order_is_respected() -> None:
    graph = build_causal_graph([TraceEvent(id="a", seq=0, type=EventType.STATE_UPDATE)])
    pipeline = Pipeline([_NoopPass("second"), _NoopPass("first")])
    result = pipeline.run(graph, CompileContext())
    assert [r.pass_name for r in result.reports] == ["second", "first"]


def test_strict_budget_not_satisfied_raises() -> None:
    events = [TraceEvent(id="a", seq=0, type=EventType.TOOL_RESULT, outputs={"x": "y" * 1000})]
    graph = build_causal_graph(events)
    with pytest.raises(BudgetNotSatisfiableError):
        compile_graph(graph, budget_tokens=1, strict=True)


def test_non_strict_budget_not_satisfied_sets_flag_without_raising() -> None:
    events = [TraceEvent(id="a", seq=0, type=EventType.TOOL_RESULT, outputs={"x": "y" * 1000})]
    graph = build_causal_graph(events)
    result = compile_graph(graph, budget_tokens=1, strict=False)
    assert result.budget_satisfied is False


def test_all_default_passes_run_even_if_budget_already_satisfied() -> None:
    events = [TraceEvent(id="a", seq=0, type=EventType.STATE_UPDATE)]
    graph = build_causal_graph(events)
    result = compile_graph(graph, budget_tokens=10_000)
    assert [r.pass_name for r in result.reports] == [p.name for p in DEFAULT_PASSES]


def test_no_budget_leaves_budget_satisfied_none() -> None:
    graph = build_causal_graph([TraceEvent(id="a", seq=0, type=EventType.STATE_UPDATE)])
    result = compile_graph(graph)
    assert result.budget_satisfied is None


def test_default_compile_keeps_goal_and_originating_call_for_a_realistic_turn() -> None:
    messages = [
        {"role": "system", "content": "be helpful"},
        {"role": "user", "content": "what's the weather in nyc?"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city": "nyc"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": '{"temp": 72}'},
        {"role": "assistant", "content": "it's 72F in nyc"},
    ]
    graph = build_causal_graph(from_openai_messages(messages))
    result = compile_graph(graph)
    assert {e.type for e in result.events} == {
        EventType.CONSTRAINT,
        EventType.USER_GOAL,
        EventType.TOOL_CALL,
        EventType.TOOL_RESULT,
        EventType.MODEL_MESSAGE,
    }


def test_default_compile_keeps_all_constraints_when_their_state_is_superseded() -> None:
    events = [
        TraceEvent(
            id="constraint-a",
            seq=0,
            type=EventType.CONSTRAINT,
            outputs={"content": "constraint A"},
            writes=frozenset({"conversation:current"}),
        ),
        TraceEvent(
            id="constraint-b",
            seq=1,
            type=EventType.CONSTRAINT,
            outputs={"content": "constraint B"},
            writes=frozenset({"conversation:current"}),
        ),
        TraceEvent(
            id="goal",
            seq=2,
            type=EventType.USER_GOAL,
            outputs={"content": "go"},
            reads=frozenset({"conversation:current"}),
            writes=frozenset({"conversation:current", "user_goal:current"}),
        ),
        TraceEvent(
            id="answer",
            seq=3,
            type=EventType.MODEL_MESSAGE,
            outputs={"content": "done"},
            reads=frozenset({"user_goal:current"}),
            writes=frozenset({"conversation:current"}),
        ),
    ]

    result = compile_graph(build_causal_graph(events))

    constraints = [event for event in result.events if event.type is EventType.CONSTRAINT]
    assert [event.id for event in constraints] == ["constraint-a", "constraint-b"]
    assert all(event.pinned for event in constraints)


def test_default_compile_keeps_goal_for_a_claude_code_transcript() -> None:
    records = [
        {"type": "user", "message": {"role": "user", "content": "what's the weather in nyc?"}},
        {
            "type": "assistant",
            "message": {
                "id": "msg_1",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "get_weather",
                        "input": {"city": "nyc"},
                    }
                ],
            },
        },
        {
            "type": "user",
            "message": {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "72F"}],
            },
        },
        {
            "type": "assistant",
            "message": {"id": "msg_2", "content": [{"type": "text", "text": "it's 72F in nyc"}]},
        },
    ]
    graph = build_causal_graph(from_claude_code_transcript(records))
    result = compile_graph(graph)
    assert {e.type for e in result.events} == {
        EventType.USER_GOAL,
        EventType.TOOL_CALL,
        EventType.TOOL_RESULT,
        EventType.MODEL_MESSAGE,
    }


def test_default_compile_keeps_goal_for_a_codex_rollout() -> None:
    def item(payload: dict[str, object]) -> dict[str, object]:
        return {"type": "response_item", "payload": payload}

    records = [
        item(
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "what's the weather in nyc?"}],
            }
        ),
        item(
            {
                "type": "function_call",
                "name": "get_weather",
                "arguments": '{"city": "nyc"}',
                "call_id": "call_1",
            }
        ),
        item({"type": "function_call_output", "call_id": "call_1", "output": "72F"}),
        item(
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "it's 72F in nyc"}],
            }
        ),
    ]
    graph = build_causal_graph(from_codex_rollout(records))
    result = compile_graph(graph)
    assert {e.type for e in result.events} == {
        EventType.USER_GOAL,
        EventType.TOOL_CALL,
        EventType.TOOL_RESULT,
        EventType.MODEL_MESSAGE,
    }
