"""Pass: mark constraints as pinned before anything else can prune them."""

from __future__ import annotations

from agentslice.compiler.base import CompilationReport, CompileContext, Pass, PassOutcome
from agentslice.compiler.tokens import estimate_graph_tokens
from agentslice.ir.events import EventType
from agentslice.ir.graph import CausalGraph, build_causal_graph


class ConstraintPinningPass(Pass):
    """Marks every ``constraint`` event as pinned.

    Runs first in :data:`~agentslice.compiler.pipeline.DEFAULT_PASSES` so
    that dead-event and superseded-state pruning cannot drop a constraint
    for lacking causal edges, which is normal for a standing instruction
    that nothing "reads" in the usual sense. Only ever adds protection:
    events that are already pinned, or aren't constraints, are left as-is.
    """

    name = "constraint_pinning"

    def apply(self, graph: CausalGraph, ctx: CompileContext) -> PassOutcome:
        events = graph.events_in_order()
        tokens_before = estimate_graph_tokens(events)

        pinned_ids: list[str] = []
        new_events = []
        for event in events:
            if event.type is EventType.CONSTRAINT and not event.pinned:
                event = event.model_copy(update={"pinned": True})
                pinned_ids.append(event.id)
            elif event.pinned:
                pinned_ids.append(event.id)
            new_events.append(event)

        new_graph = build_causal_graph(new_events)
        tokens_after = estimate_graph_tokens(new_events)
        report = CompilationReport(
            pass_name=self.name,
            events_before=len(events),
            events_after=len(new_events),
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            pinned_event_ids=pinned_ids,
        )
        return PassOutcome(graph=new_graph, ctx=ctx, report=report)
