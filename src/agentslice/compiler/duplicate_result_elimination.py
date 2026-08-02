"""Pass: collapse a tool_call/tool_result pair that exactly repeats an earlier one."""

from __future__ import annotations

import json
from typing import Any

from agentslice.compiler.base import CompilationReport, CompileContext, Pass, PassOutcome
from agentslice.compiler.tokens import estimate_graph_tokens
from agentslice.ir.events import EventType, TraceEvent
from agentslice.ir.graph import CausalGraph, build_causal_graph

_REDACTED_OUTPUTS = {"_redacted": "duplicate of an earlier identical tool result"}


def _result_of(call: TraceEvent, graph: CausalGraph) -> TraceEvent | None:
    """Return the sole `tool_result` causally downstream of `call`, or None if ambiguous.

    Deliberately structural (edges only) rather than relying on any
    particular fact-key naming convention, since this is compiler code and
    must not depend on how `recording/` chose to link the two together.
    Zero or more than one candidate means the link can't be established
    safely, so the call is left out of duplicate consideration entirely.
    """
    results = [
        graph.events[edge.to_event_id]
        for edge in graph.edges
        if edge.from_event_id == call.id
        and graph.events[edge.to_event_id].type is EventType.TOOL_RESULT
    ]
    return results[0] if len(results) == 1 else None


def _inputs_key(inputs: dict[str, Any] | None) -> str:
    return json.dumps(inputs or {}, sort_keys=True, default=str)


class DuplicateResultEliminationPass(Pass):
    """Collapses a later `tool_call`/`tool_result` pair that exactly repeats an earlier one.

    Two `tool_call` events with the same `tool_name` and `inputs` whose
    linked `tool_result` events have identical `outputs` are duplicates:
    the second call told the model nothing the first one didn't already.
    The earliest occurrence is always kept. A later duplicate is dropped
    entirely when its result has `side_effects=False` — the pair added
    nothing and executing it (if replayed) had no consequence either.
    One with `side_effects=True` is kept but redacted, the same
    remove-vs-redact split `superseded_state` uses: the call still
    happened and may matter for audit, but its (redundant) value doesn't
    need to be shown again in full.

    A candidate is left untouched — not dropped, not redacted — if the
    anchor or a pinned event is one half of the pair, or if some other
    surviving event has a causal edge reading from the call or the result
    (beyond the edge the result itself has back to its own call): removing
    it would then leave that other event's read unresolved, the same class
    of problem `superseded_state` guards against for superseded facts. A
    reader that depends on *both* the duplicate and the baseline (which
    happens routinely: the adapter's value-matching links a later call to
    every prior fact sharing that value, not just the newest) does not
    block elimination — the baseline alone already covers it.
    """

    name = "duplicate_result_elimination"

    def apply(self, graph: CausalGraph, ctx: CompileContext) -> PassOutcome:
        events = graph.events_in_order()
        tokens_before = estimate_graph_tokens(events)
        anchor_id = ctx.anchor_event_id or (events[-1].id if events else None)

        readers_of: dict[str, set[str]] = {}
        for edge in graph.edges:
            readers_of.setdefault(edge.from_event_id, set()).add(edge.to_event_id)

        groups: dict[tuple[str, str], list[TraceEvent]] = {}
        for event in events:
            if event.type is not EventType.TOOL_CALL or not event.tool_name:
                continue
            key = (event.tool_name, _inputs_key(event.inputs))
            groups.setdefault(key, []).append(event)

        to_drop: set[str] = set()
        to_redact: set[str] = set()
        notes: list[str] = []

        for calls in groups.values():
            if len(calls) < 2:
                continue

            baseline_call: TraceEvent | None = None
            baseline_result: TraceEvent | None = None
            baseline_dependents: set[str] = set()
            for call in sorted(calls, key=lambda e: e.seq):
                result = _result_of(call, graph)
                if result is None:
                    continue
                if baseline_result is None:
                    baseline_call, baseline_result = call, result
                    baseline_dependents = readers_of.get(call.id, set()) | readers_of.get(
                        result.id, set()
                    )
                    continue
                if result.outputs != baseline_result.outputs:
                    continue

                external_readers = (readers_of.get(call.id, set()) - {result.id}) | readers_of.get(
                    result.id, set()
                )
                # A reader that also depends on the baseline pair is redundantly
                # covered: dropping this duplicate doesn't strand it, since the
                # identical value it needed is still available from the baseline.
                blocking_readers = external_readers - baseline_dependents
                if blocking_readers:
                    continue
                if anchor_id in (call.id, result.id):
                    continue
                if call.pinned or result.pinned:
                    continue

                assert baseline_call is not None
                if result.side_effects:
                    to_redact.add(result.id)
                    notes.append(
                        f"{result.id}: redacted as a side-effecting duplicate of "
                        f"{baseline_call.id}/{baseline_result.id}"
                    )
                else:
                    to_drop.add(call.id)
                    to_drop.add(result.id)
                    notes.append(
                        f"{call.id}/{result.id}: dropped as a duplicate of "
                        f"{baseline_call.id}/{baseline_result.id}"
                    )

        removed_ids: list[str] = []
        modified_ids: list[str] = []
        new_events = []
        for event in events:
            if event.id in to_drop:
                removed_ids.append(event.id)
                continue
            if event.id in to_redact:
                new_events.append(
                    event.model_copy(update={"outputs": dict(_REDACTED_OUTPUTS), "metadata": {}})
                )
                modified_ids.append(event.id)
                continue
            new_events.append(event)

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
            notes=notes,
        )
        return PassOutcome(graph=new_graph, ctx=ctx, report=report)
