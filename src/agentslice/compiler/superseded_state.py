"""Pass: collapse events whose only role was writing since-overwritten state."""

from __future__ import annotations

from agentslice.compiler.base import CompilationReport, CompileContext, Pass, PassOutcome
from agentslice.compiler.tokens import estimate_graph_tokens
from agentslice.ir.graph import CausalGraph, build_causal_graph

_REDACTED_OUTPUTS = {"_redacted": "superseded value omitted"}


class SupersededStatePass(Pass):
    """Removes or redacts events whose written facts have all been overwritten.

    An event is "fully superseded" when every key it wrote no longer points
    to it as the latest version in ``graph.facts``. A fully superseded
    event with ``side_effects=False`` is dropped entirely: its only role
    was producing a value nothing downstream still uses. One with
    ``side_effects=True`` is kept, since its execution still happened and
    may matter for audit, but its ``outputs`` are redacted and its
    ``pinned`` flag is cleared: a pin protects a fact's *current* value,
    not a value it has since moved past.

    Events that never write a fact, or whose writes are still current, are
    left untouched.
    """

    name = "superseded_state"

    def apply(self, graph: CausalGraph, ctx: CompileContext) -> PassOutcome:
        events = graph.events_in_order()
        tokens_before = estimate_graph_tokens(events)

        current_writer_of = {
            key: versions[-1].origin_event_id for key, versions in graph.facts.items() if versions
        }

        removed_ids: list[str] = []
        modified_ids: list[str] = []
        new_events = []

        for event in events:
            if not event.writes:
                new_events.append(event)
                continue

            is_current = any(current_writer_of.get(key) == event.id for key in event.writes)
            if is_current:
                new_events.append(event)
                continue

            if event.side_effects:
                new_events.append(
                    event.model_copy(update={"outputs": dict(_REDACTED_OUTPUTS), "pinned": False})
                )
                modified_ids.append(event.id)
            else:
                removed_ids.append(event.id)

        new_graph = build_causal_graph(new_events)
        tokens_after = estimate_graph_tokens(new_events)
        report = CompilationReport(
            pass_name=self.name,
            events_before=len(events),
            events_after=len(new_events),
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            removed_event_ids=removed_ids,
            modified_event_ids=modified_ids,
        )
        return PassOutcome(graph=new_graph, ctx=ctx, report=report)
