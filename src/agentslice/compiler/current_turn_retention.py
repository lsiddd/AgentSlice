"""Pass: pin every event in the anchor's own turn before anything else can prune it."""

from __future__ import annotations

from agentslice.compiler.base import CompilationReport, CompileContext, Pass, PassOutcome
from agentslice.compiler.tokens import estimate_graph_tokens
from agentslice.ir.events import EventType
from agentslice.ir.graph import CausalGraph, build_causal_graph


class CurrentTurnRetentionPass(Pass):
    """Pins every event from the anchor's own turn, regardless of causal reach.

    A turn starts at the most recent ``user_goal`` at or before the anchor
    and runs through the anchor itself. ``dead_events`` only keeps an event
    that is a causal ancestor of the anchor — connected through a fact some
    later event actually reads. An exploratory ``tool_call`` the agent made
    earlier *in the same turn it is still completing* (a ``pwd`` or ``ls``
    run to orient itself, whose result was never reused as a literal
    argument downstream) has no such edge, so it gets pruned the moment the
    turn's next iteration recompiles the context. The agent then has no
    record of having just done that and repeats it — this is a live loop
    observed running ``causal_compile`` against the BFCL benchmark, not a
    hypothetical.

    Compaction is meant to summarize *history*, not erase an agent's
    short-term memory of what it is still in the middle of doing. Runs
    before ``dead_events`` (mirrors ``constraint_pinning``'s ordering
    rule), so every event pinned here is protected by the same mechanism
    a constraint is, including the seq-bounded pinned-event allowance in
    ``dead_events`` and the ancestor-of-pinned-event protection it also
    provides.
    """

    name = "current_turn_retention"

    def apply(self, graph: CausalGraph, ctx: CompileContext) -> PassOutcome:
        events = graph.events_in_order()
        tokens_before = estimate_graph_tokens(events)

        anchor_id = ctx.anchor_event_id or (events[-1].id if events else None)
        anchor = graph.events.get(anchor_id) if anchor_id is not None else None

        if anchor is None:
            report = CompilationReport(
                pass_name=self.name,
                events_before=len(events),
                events_after=len(events),
                tokens_before=tokens_before,
                tokens_after=tokens_before,
            )
            return PassOutcome(graph=graph, ctx=ctx, report=report)

        turn_start_seq = 0
        for event in events:
            if event.type is EventType.USER_GOAL and event.seq <= anchor.seq:
                turn_start_seq = event.seq

        pinned_ids: list[str] = []
        new_events = []
        for event in events:
            if event.pinned:
                pinned_ids.append(event.id)
                new_events.append(event)
            elif turn_start_seq <= event.seq <= anchor.seq:
                event = event.model_copy(update={"pinned": True})
                pinned_ids.append(event.id)
                new_events.append(event)
            else:
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
