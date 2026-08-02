import pytest

from agentslice.compiler.base import CompilationReport, CompileContext, Pass, PassOutcome
from agentslice.compiler.pipeline import DEFAULT_PASSES, Pipeline, compile_graph
from agentslice.errors import BudgetNotSatisfiableError
from agentslice.ir.events import EventType, TraceEvent
from agentslice.ir.graph import CausalGraph, build_causal_graph


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
