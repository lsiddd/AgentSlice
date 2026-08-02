"""Orchestration of compiler passes into a single compilation run."""

from __future__ import annotations

from collections.abc import Sequence

from agentslice.compiler.base import (
    CompilationReport,
    CompileContext,
    CompiledContext,
    Pass,
    ToolSchema,
)
from agentslice.compiler.constraint_pinning import ConstraintPinningPass
from agentslice.compiler.current_turn_retention import CurrentTurnRetentionPass
from agentslice.compiler.dead_events import DeadEventsPass
from agentslice.compiler.duplicate_result_elimination import DuplicateResultEliminationPass
from agentslice.compiler.failed_hypothesis_folding import (
    FailedHypothesisFoldingPass,
    FoldAnnotationResolutionPass,
)
from agentslice.compiler.schema_pruning import SchemaPruningPass
from agentslice.compiler.superseded_state import SupersededStatePass
from agentslice.compiler.tokens import estimate_graph_tokens
from agentslice.compiler.tool_result_projection import ToolResultProjectionPass
from agentslice.errors import BudgetNotSatisfiableError
from agentslice.ir.graph import CausalGraph

DEFAULT_PASSES: tuple[Pass, ...] = (
    ConstraintPinningPass(),
    CurrentTurnRetentionPass(),
    DeadEventsPass(),
    SupersededStatePass(),
    DuplicateResultEliminationPass(),
    ToolResultProjectionPass(),
    SchemaPruningPass(),
)

EXPERIMENTAL_FAILED_HYPOTHESIS_FOLDING_PASSES: tuple[Pass, ...] = (
    DEFAULT_PASSES[0],
    DEFAULT_PASSES[1],
    FoldAnnotationResolutionPass(),
    FailedHypothesisFoldingPass(),
    *DEFAULT_PASSES[2:],
)


class Pipeline:
    """Runs a sequence of compiler passes over a causal graph, in order.

    Always runs every configured pass, even once the token budget is
    already satisfied: simplicity over an early-exit optimization that
    would make results depend on pass order in confusing ways.
    """

    def __init__(self, passes: Sequence[Pass] = DEFAULT_PASSES) -> None:
        self._passes = tuple(passes)

    def run(self, graph: CausalGraph, ctx: CompileContext) -> CompiledContext:
        tokens_before = estimate_graph_tokens(graph.events_in_order())
        reports: list[CompilationReport] = []

        current_graph = graph
        current_ctx = ctx
        for compiler_pass in self._passes:
            outcome = compiler_pass.apply(current_graph, current_ctx)
            current_graph = outcome.graph
            current_ctx = outcome.ctx
            reports.append(outcome.report)

        tokens_after = estimate_graph_tokens(current_graph.events_in_order())
        budget_satisfied: bool | None = None
        if ctx.budget_tokens is not None:
            budget_satisfied = tokens_after <= ctx.budget_tokens
            if ctx.strict and not budget_satisfied:
                raise BudgetNotSatisfiableError(
                    f"budget of {ctx.budget_tokens} tokens not met: "
                    f"{tokens_after} tokens remain after running all passes"
                )

        return CompiledContext(
            events=current_graph.events_in_order(),
            tool_catalog=current_ctx.tool_catalog,
            reports=reports,
            budget_tokens=ctx.budget_tokens,
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            budget_satisfied=budget_satisfied,
        )


def compile_graph(
    graph: CausalGraph,
    *,
    budget_tokens: int | None = None,
    tool_catalog: dict[str, ToolSchema] | None = None,
    passes: Sequence[Pass] = DEFAULT_PASSES,
    strict: bool = False,
    strict_schema: bool = False,
    anchor_event_id: str | None = None,
    accepted_fold_annotators: frozenset[str] | None = None,
) -> CompiledContext:
    """Compile ``graph`` into a :class:`~agentslice.compiler.base.CompiledContext`.

    Convenience wrapper around :class:`Pipeline` for the common case of a
    one-off compilation with a fresh :class:`~agentslice.compiler.base.CompileContext`.
    """
    ctx = CompileContext(
        budget_tokens=budget_tokens,
        tool_catalog=tool_catalog or {},
        strict=strict,
        strict_schema=strict_schema,
        anchor_event_id=anchor_event_id,
        accepted_fold_annotators=(
            accepted_fold_annotators
            if accepted_fold_annotators is not None
            else frozenset({"runtime", "human"})
        ),
    )
    return Pipeline(passes).run(graph, ctx)
