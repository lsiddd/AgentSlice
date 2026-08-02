"""Pass: shrink tool results down to only the fields something reads."""

from __future__ import annotations

from agentslice.compiler.base import CompilationReport, CompileContext, Pass, PassOutcome
from agentslice.compiler.tokens import estimate_graph_tokens
from agentslice.ir.events import EventType, TraceEvent
from agentslice.ir.graph import CausalGraph, build_causal_graph


def _matching_field_keys(event: TraceEvent) -> dict[str, str] | None:
    """Map each per-field write key of ``event`` back to its output field name.

    Matches by suffix (``write_key.endswith(f".{field_name}")``) rather
    than splitting the key from the right, so a field name that itself
    contains a dot (``"user.name"``) is not truncated to the wrong,
    shorter name. Every output field must resolve to exactly one write
    key, and every resolved write key must agree on the same prefix before
    the field name — anything ambiguous (an opaque result whose call id
    happens to contain a dot and so superficially looks field-shaped, two
    field names that are suffixes of one another) returns ``None`` rather
    than risk projecting the wrong field away.
    """
    if not event.outputs:
        return None

    field_keys: dict[str, str] = {}
    prefix: str | None = None
    for field_name in event.outputs:
        suffix = f".{field_name}"
        matches = [key for key in event.writes if key.endswith(suffix)]
        if len(matches) != 1:
            return None
        write_key = matches[0]
        this_prefix = write_key[: -len(suffix)]
        if prefix is None:
            prefix = this_prefix
        elif prefix != this_prefix:
            return None
        field_keys[write_key] = field_name
    return field_keys


class ToolResultProjectionPass(Pass):
    """Projects a ``tool_result``'s ``outputs`` down to the fields something reads.

    Only applies to tool results whose ``writes`` were broken out per field
    (see :func:`_matching_field_keys`); a result written as a single opaque
    key has nothing to project down to and is left alone. Any write key
    that isn't part of that per-field mapping (e.g. a bookkeeping key
    unrelated to a specific output field) survives projection untouched,
    alongside whichever field keys are still read. The anchor event and
    any pinned event are always kept whole, since "the current state" and
    "protected content" must stay fully available regardless of who reads
    them.
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

            field_keys = _matching_field_keys(event)
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
            other_writes = event.writes - set(field_keys)
            update = {"outputs": projected_outputs, "writes": frozenset(kept_keys) | other_writes}
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
