"""Pass: drop events with no causal path to the anchor and that aren't pinned."""

from __future__ import annotations

from agentslice.compiler.base import CompilationReport, CompileContext, Pass, PassOutcome
from agentslice.compiler.tokens import estimate_graph_tokens
from agentslice.ir.graph import CausalGraph, build_causal_graph


class DeadEventsPass(Pass):
    """Removes events that neither lead to the anchor event nor are pinned.

    The anchor is ``ctx.anchor_event_id`` if set, otherwise the event with
    the highest ``seq`` (the most recent one, i.e. "now"). An event
    survives if it is the anchor itself, a causal ancestor of the anchor
    (per :meth:`~agentslice.ir.graph.CausalGraph.ancestors`), a pinned
    event with a ``seq`` no later than the anchor's, or a causal ancestor
    of one of those pinned events. That last clause matters: a pinned
    event is kept for reasons unrelated to the anchor (it is a standing
    constraint, say), but it can still ``reads`` a fact written by some
    earlier, unpinned, non-ancestor-of-anchor event. Without also keeping
    that writer alive, rebuilding the graph from the survivors would leave
    the pinned event's read unresolved, silently — a protected event would
    survive without the data it depends on. The anchor-seq bound on
    pinned events matters too: without it, a constraint (or any other
    pinned event) recorded *after* the anchor would still survive, leaking
    future information into what is supposed to be a snapshot of
    "everything known up to this point" — exactly the scenario ``fork``
    relies on not happening.
    """

    name = "dead_events"

    def apply(self, graph: CausalGraph, ctx: CompileContext) -> PassOutcome:
        events = graph.events_in_order()
        tokens_before = estimate_graph_tokens(events)

        if not events:
            report = CompilationReport(
                pass_name=self.name,
                events_before=0,
                events_after=0,
                tokens_before=0,
                tokens_after=0,
            )
            return PassOutcome(graph=graph, ctx=ctx, report=report)

        anchor_id = ctx.anchor_event_id or events[-1].id
        anchor = graph.events.get(anchor_id)
        anchor_seq = anchor.seq if anchor is not None else None

        alive = {anchor_id} | graph.ancestors(anchor_id)
        pinned_alive = {
            event.id
            for event in events
            if event.pinned and (anchor_seq is None or event.seq <= anchor_seq)
        }
        for pinned_id in pinned_alive:
            alive |= graph.ancestors(pinned_id)
        alive |= pinned_alive

        removed_ids = [event.id for event in events if event.id not in alive]
        surviving = [event for event in events if event.id in alive]

        new_graph = build_causal_graph(surviving)
        tokens_after = estimate_graph_tokens(surviving)
        report = CompilationReport(
            pass_name=self.name,
            events_before=len(events),
            events_after=len(surviving),
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            removed_event_ids=removed_ids,
        )
        return PassOutcome(graph=new_graph, ctx=ctx, report=report)
