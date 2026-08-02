"""Pass: shrink tool results down to only the fields something reads."""

from __future__ import annotations

from agentslice.compiler.base import CompilationReport, CompileContext, Pass, PassOutcome
from agentslice.compiler.tokens import estimate_graph_tokens
from agentslice.ir.events import EventType
from agentslice.ir.graph import CausalGraph, build_causal_graph


class ToolResultProjectionPass(Pass):
    """Projects a ``tool_result``'s ``outputs`` down to the fields something reads.

    Only applies to tool results whose ``writes`` were broken out per field
    (keys of the form ``tool_result:{id}.{field}``, produced when the
    adapter judged the result a shallow dict); a result written as a
    single opaque key has nothing to project down to and is left alone.
    The anchor event and any pinned event are always kept whole, since
    "the current state" and "protected content" must stay fully available
    regardless of who reads them.
    """

    name = "tool_result_projection"

    def apply(self, graph: CausalGraph, ctx: CompileContext) -> PassOutcome:
        events = graph.events_in_order()
        tokens_before = estimate_graph_tokens(events)
        anchor_id = ctx.anchor_event_id or (events[-1].id if events else None)

        all_reads: set[str] = set()
        for event in events:
            all_reads |= event.reads

        modified_ids: list[str] = []
        notes: list[str] = []
        new_events = []

        for event in events:
            if event.type is not EventType.TOOL_RESULT or event.outputs is None:
                new_events.append(event)
                continue
            if event.id == anchor_id or event.pinned:
                new_events.append(event)
                continue

            field_keys = {key: key.rsplit(".", 1)[-1] for key in event.writes if "." in key}
            if not field_keys:
                new_events.append(event)
                continue

            kept_keys = {key for key in field_keys if key in all_reads}
            if kept_keys == set(field_keys):
                new_events.append(event)
                continue

            kept_fields = {field_keys[key] for key in kept_keys}
            dropped_fields = sorted(set(event.outputs) - kept_fields)
            projected_outputs = {k: v for k, v in event.outputs.items() if k in kept_fields}
            update = {"outputs": projected_outputs, "writes": frozenset(kept_keys)}
            new_events.append(event.model_copy(update=update))
            modified_ids.append(event.id)
            notes.append(f"{event.id}: dropped unread fields {dropped_fields}")

        new_graph = build_causal_graph(new_events)
        tokens_after = estimate_graph_tokens(new_events)
        report = CompilationReport(
            pass_name=self.name,
            events_before=len(events),
            events_after=len(new_events),
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            modified_event_ids=modified_ids,
            notes=notes,
        )
        return PassOutcome(graph=new_graph, ctx=ctx, report=report)
