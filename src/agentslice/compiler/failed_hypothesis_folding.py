"""Experimental passes for folding explicitly annotated failed hypotheses."""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from agentslice.compiler.base import (
    CompilationReport,
    CompileContext,
    Pass,
    PassOutcome,
    ToolEffect,
)
from agentslice.compiler.fold_plans import EvidenceRef, FoldPlan
from agentslice.compiler.tokens import estimate_event_tokens, estimate_graph_tokens
from agentslice.ir.epistemic import EpistemicEvidence, EpistemicStateV1, RuledOutHypothesis
from agentslice.ir.events import EventType, TraceEvent
from agentslice.ir.graph import CausalGraph, build_causal_graph
from agentslice.ir.keys import CONVERSATION_KEY

_FOLD_METADATA_NAMESPACE = "agentslice"
_FOLD_METADATA_KEY = "fold"
_EPISTEMIC_KEY_PREFIX = "epistemic:ruled_out:"
_INVALID_POINTER_ESCAPE = re.compile(r"~(?![01])")


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class _HypothesisAnnotation(_StrictModel):
    text: str = Field(min_length=1)
    source_event_id: str = Field(min_length=1)


class _EvidenceAnnotation(_StrictModel):
    event_id: str = Field(min_length=1)
    json_pointer: str
    operator: Literal["=="]
    value: Any


class _AnnotatorAnnotation(_StrictModel):
    kind: str = Field(min_length=1)
    name: str = Field(min_length=1)
    version: str = Field(min_length=1)


class _FoldAnnotationV1(_StrictModel):
    schema_version: Literal[1]
    fold_id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    kind: Literal["ruled_out_hypothesis"]
    hypothesis: _HypothesisAnnotation
    evidence: list[_EvidenceAnnotation] = Field(min_length=1)
    remove_event_ids: list[str] = Field(min_length=1)
    conclusion_event_ids: list[str] = Field(min_length=1)
    dedicated_conclusion: bool
    annotator: _AnnotatorAnnotation


@dataclass(frozen=True)
class _Candidate:
    annotation_event_id: str
    annotation: _FoldAnnotationV1


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _format_validation_error(exc: ValidationError) -> str:
    first = exc.errors()[0]
    location = ".".join(str(part) for part in first["loc"])
    return f"{location}: {first['msg']}"


def _json_pointer_get(value: Any, pointer: str) -> Any:
    if pointer == "":
        return value
    if not pointer.startswith("/"):
        raise ValueError("JSON pointer must be empty or start with '/'")

    current = value
    for raw_token in pointer[1:].split("/"):
        if _INVALID_POINTER_ESCAPE.search(raw_token):
            raise ValueError(f"invalid JSON pointer escape in {raw_token!r}")
        token = raw_token.replace("~1", "/").replace("~0", "~")
        if isinstance(current, Mapping):
            if token not in current:
                raise ValueError(f"JSON pointer component {token!r} does not exist")
            current = current[token]
        elif isinstance(current, list):
            if not token.isdigit():
                raise ValueError(f"JSON pointer list index {token!r} is invalid")
            index = int(token)
            if index >= len(current):
                raise ValueError(f"JSON pointer list index {index} is out of range")
            current = current[index]
        else:
            raise ValueError(f"JSON pointer cannot descend through {type(current).__name__}")
    return current


def _annotation_candidates(
    events: list[TraceEvent],
    accepted_annotators: frozenset[str],
) -> tuple[list[_Candidate], list[str]]:
    candidates: list[_Candidate] = []
    notes: list[str] = []

    for event in events:
        if _FOLD_METADATA_NAMESPACE not in event.metadata:
            continue
        namespace = event.metadata[_FOLD_METADATA_NAMESPACE]
        if not isinstance(namespace, Mapping):
            notes.append(f"{event.id}: agentslice metadata namespace is not an object")
            continue
        if _FOLD_METADATA_KEY not in namespace:
            continue
        try:
            annotation = _FoldAnnotationV1.model_validate(namespace[_FOLD_METADATA_KEY])
        except ValidationError as exc:
            notes.append(f"{event.id}: invalid fold annotation: {_format_validation_error(exc)}")
            continue
        if annotation.annotator.kind not in accepted_annotators:
            notes.append(
                f"{event.id}: annotator kind {annotation.annotator.kind!r} is not accepted"
            )
            continue
        candidates.append(_Candidate(annotation_event_id=event.id, annotation=annotation))

    duplicate_ids = {
        fold_id
        for fold_id, count in Counter(
            candidate.annotation.fold_id for candidate in candidates
        ).items()
        if count > 1
    }
    if duplicate_ids:
        candidates = [
            candidate
            for candidate in candidates
            if candidate.annotation.fold_id not in duplicate_ids
        ]
        notes.extend(
            f"fold {fold_id!r}: duplicate fold_id, all matching annotations ignored"
            for fold_id in sorted(duplicate_ids)
        )

    return candidates, notes


def _resolve_candidate(
    candidate: _Candidate,
    graph: CausalGraph,
    ctx: CompileContext,
) -> tuple[FoldPlan | None, str | None]:
    annotation = candidate.annotation
    prefix = f"fold {annotation.fold_id!r}"
    remove_ids = tuple(annotation.remove_event_ids)

    if len(set(remove_ids)) != len(remove_ids):
        return None, f"{prefix}: remove_event_ids contains duplicates"
    if any(event_id not in graph.events for event_id in remove_ids):
        missing = sorted(event_id for event_id in remove_ids if event_id not in graph.events)
        return None, f"{prefix}: remove_event_ids contains unknown events {missing}"

    positions = [graph.order.index(event_id) for event_id in remove_ids]
    if positions != sorted(positions) or positions != list(
        range(min(positions), max(positions) + 1)
    ):
        return None, f"{prefix}: remove_event_ids is not contiguous or not ordered"

    region = [graph.events[event_id] for event_id in remove_ids]
    region_ids = set(remove_ids)
    allowed_types = {EventType.MODEL_MESSAGE, EventType.TOOL_CALL, EventType.TOOL_RESULT}
    invalid_types = sorted(
        f"{event.id}:{event.type.value}" for event in region if event.type not in allowed_types
    )
    if invalid_types:
        return None, f"{prefix}: region contains unsupported event types {invalid_types}"

    anchor_id = ctx.anchor_event_id or graph.order[-1]
    anchor = graph.events.get(anchor_id)
    if anchor is None:
        return None, f"{prefix}: anchor {anchor_id!r} does not exist"
    if anchor_id in region_ids:
        return None, f"{prefix}: region contains the anchor {anchor_id!r}"
    if region[-1].seq > anchor.seq:
        return None, f"{prefix}: region occurs after the anchor {anchor_id!r}"
    pinned_ids = sorted(event.id for event in region if event.pinned)
    if pinned_ids:
        return None, f"{prefix}: region contains pinned event(s) {pinned_ids}"
    side_effect_ids = sorted(event.id for event in region if event.side_effects)
    if side_effect_ids:
        return (
            None,
            f"{prefix}: region contains event(s) marked with side effects {side_effect_ids}",
        )

    if not annotation.dedicated_conclusion:
        return None, f"{prefix}: conclusion is not explicitly dedicated to this hypothesis"

    hypothesis_id = annotation.hypothesis.source_event_id
    if hypothesis_id not in region_ids:
        return None, f"{prefix}: hypothesis source event is outside the region"
    if graph.events[hypothesis_id].type is not EventType.MODEL_MESSAGE:
        return None, f"{prefix}: hypothesis source event is not a model_message"

    conclusion_ids = tuple(annotation.conclusion_event_ids)
    if len(set(conclusion_ids)) != len(conclusion_ids):
        return None, f"{prefix}: conclusion_event_ids contains duplicates"
    if any(event_id not in region_ids for event_id in conclusion_ids):
        return None, f"{prefix}: conclusion event is outside the region"
    if any(
        graph.events[event_id].type is not EventType.MODEL_MESSAGE for event_id in conclusion_ids
    ):
        return None, f"{prefix}: every conclusion event must be a model_message"
    conclusion_positions = [graph.order.index(event_id) for event_id in conclusion_ids]
    if conclusion_positions != sorted(conclusion_positions):
        return None, f"{prefix}: conclusion_event_ids is not ordered"
    if conclusion_ids[-1] != remove_ids[-1]:
        return None, f"{prefix}: final region event is not the final conclusion"
    if candidate.annotation_event_id != conclusion_ids[-1]:
        return None, f"{prefix}: annotation must be on the final conclusion event"

    calls = [event for event in region if event.type is EventType.TOOL_CALL]
    results = [event for event in region if event.type is EventType.TOOL_RESULT]
    if not calls or not results:
        return None, f"{prefix}: region must contain at least one tool call and result"

    for call in calls:
        if not call.tool_name:
            return None, f"{prefix}: tool call {call.id!r} has no tool_name"
        schema = ctx.tool_catalog.get(call.tool_name)
        if schema is None:
            return None, f"{prefix}: tool {call.tool_name!r} has no tool schema"
        if schema.effects is not ToolEffect.PURE:
            return None, f"{prefix}: tool {call.tool_name!r} is not explicitly pure"
        linked_results = {
            edge.to_event_id
            for edge in graph.edges
            if edge.from_event_id == call.id
            and edge.to_event_id in region_ids
            and graph.events[edge.to_event_id].type is EventType.TOOL_RESULT
        }
        if not linked_results:
            return None, f"{prefix}: tool call {call.id!r} has no linked result in the region"

    call_ids = {call.id for call in calls}
    for result in results:
        linked_calls = {
            edge.from_event_id
            for edge in graph.edges
            if edge.to_event_id == result.id and edge.from_event_id in call_ids
        }
        if not linked_calls:
            return None, f"{prefix}: tool result {result.id!r} has no linked call in the region"

    evidence_refs: list[EvidenceRef] = []
    evidence_keys: set[tuple[str, str]] = set()
    for evidence in annotation.evidence:
        evidence_key = (evidence.event_id, evidence.json_pointer)
        if evidence_key in evidence_keys:
            return None, f"{prefix}: evidence contains duplicate references"
        evidence_keys.add(evidence_key)
        if evidence.event_id not in region_ids:
            return None, f"{prefix}: evidence event {evidence.event_id!r} is outside the region"
        event = graph.events[evidence.event_id]
        if event.type is not EventType.TOOL_RESULT:
            return None, f"{prefix}: evidence event {event.id!r} is not a tool_result"
        try:
            actual_value = _json_pointer_get(event.outputs, evidence.json_pointer)
            expected_json = _canonical_json(evidence.value)
            actual_json = _canonical_json(actual_value)
        except (TypeError, ValueError) as exc:
            return None, f"{prefix}: invalid evidence {event.id}:{evidence.json_pointer}: {exc}"
        if actual_json != expected_json:
            return (
                None,
                f"{prefix}: evidence predicate did not match at {event.id}:{evidence.json_pointer}",
            )
        evidence_refs.append(
            EvidenceRef(
                event_id=event.id,
                json_pointer=evidence.json_pointer,
                operator=evidence.operator,
                value_json=expected_json,
            )
        )

    evidence_event_ids = {item.event_id for item in evidence_refs}
    conclusion_id_set = set(conclusion_ids)
    if (
        hypothesis_id in evidence_event_ids
        or hypothesis_id in conclusion_id_set
        or evidence_event_ids & conclusion_id_set
    ):
        return None, f"{prefix}: hypothesis, evidence, and conclusion roles overlap"
    hypothesis_seq = graph.events[hypothesis_id].seq
    evidence_seqs = [graph.events[event_id].seq for event_id in evidence_event_ids]
    conclusion_seqs = [graph.events[event_id].seq for event_id in conclusion_ids]
    if not (hypothesis_seq < min(evidence_seqs) and max(evidence_seqs) < min(conclusion_seqs)):
        return None, f"{prefix}: hypothesis, evidence, and conclusion roles are out of order"

    substantive_outgoing = sorted(
        {
            f"{edge.from_event_id}->{edge.to_event_id}:{edge.fact_key}"
            for edge in graph.edges
            if edge.from_event_id in region_ids
            and edge.to_event_id not in region_ids
            and edge.fact_key != CONVERSATION_KEY
        }
    )
    if substantive_outgoing:
        return None, f"{prefix}: region has outgoing dependency {substantive_outgoing}"

    external_reads = {
        edge.fact_key
        for edge in graph.edges
        if edge.from_event_id not in region_ids and edge.to_event_id in region_ids
    }
    external_reads |= {
        fact_key for event_id, fact_key in graph.unresolved_reads if event_id in region_ids
    }

    synthetic_id = f"fold_{annotation.fold_id}"
    if synthetic_id in graph.events:
        return None, f"{prefix}: synthetic event id {synthetic_id!r} already exists"

    return (
        FoldPlan(
            fold_id=annotation.fold_id,
            annotation_event_id=candidate.annotation_event_id,
            hypothesis_text=annotation.hypothesis.text,
            hypothesis_event_id=hypothesis_id,
            evidence=tuple(evidence_refs),
            remove_event_ids=remove_ids,
            conclusion_event_ids=conclusion_ids,
            annotator_kind=annotation.annotator.kind,
            annotator_name=annotation.annotator.name,
            annotator_version=annotation.annotator.version,
            external_reads=frozenset(external_reads),
            preserve_conversation_write=any(CONVERSATION_KEY in event.writes for event in region),
        ),
        None,
    )


class FoldAnnotationResolutionPass(Pass):
    """Resolve trusted v1 fold metadata into immutable, validated plans."""

    name = "fold_annotation_resolution"

    def apply(self, graph: CausalGraph, ctx: CompileContext) -> PassOutcome:
        events = graph.events_in_order()
        tokens = estimate_graph_tokens(events)
        candidates, notes = _annotation_candidates(events, ctx.accepted_fold_annotators)

        plans: list[FoldPlan] = []
        for candidate in candidates:
            plan, reason = _resolve_candidate(candidate, graph, ctx)
            if plan is None:
                assert reason is not None
                notes.append(reason)
                continue
            plans.append(plan)

        conflicted_fold_ids: set[str] = set()
        for index, plan in enumerate(plans):
            region = set(plan.remove_event_ids)
            for other in plans[index + 1 :]:
                if region & set(other.remove_event_ids):
                    conflicted_fold_ids.update({plan.fold_id, other.fold_id})
        if conflicted_fold_ids:
            plans = [plan for plan in plans if plan.fold_id not in conflicted_fold_ids]
            notes.extend(
                f"fold {fold_id!r}: region overlaps another fold, annotation ignored"
                for fold_id in sorted(conflicted_fold_ids)
            )

        plans.sort(key=lambda plan: graph.events[plan.remove_event_ids[0]].seq)
        notes.extend(f"fold {plan.fold_id!r}: annotation resolved" for plan in plans)
        report = CompilationReport(
            pass_name=self.name,
            events_before=len(events),
            events_after=len(events),
            tokens_before=tokens,
            tokens_after=tokens,
            notes=notes,
        )
        return PassOutcome(
            graph=graph,
            ctx=replace(ctx, fold_plans=tuple(plans)),
            report=report,
        )


def _synthetic_event(plan: FoldPlan, last_source: TraceEvent) -> TraceEvent:
    evidence = [
        EpistemicEvidence(
            event_id=item.event_id,
            json_pointer=item.json_pointer,
            operator=item.operator,
            value=item.value(),
        )
        for item in plan.evidence
    ]
    state = EpistemicStateV1(
        kind="epistemic_state",
        schema_version=1,
        ruled_out=[
            RuledOutHypothesis(
                fold_id=plan.fold_id,
                hypothesis=plan.hypothesis_text,
                evidence=evidence,
            )
        ],
    )
    writes = {f"{_EPISTEMIC_KEY_PREFIX}{plan.fold_id}"}
    if plan.preserve_conversation_write:
        writes.add(CONVERSATION_KEY)
    return TraceEvent(
        id=plan.synthetic_event_id,
        seq=last_source.seq,
        type=EventType.STATE_UPDATE,
        timestamp=last_source.timestamp,
        outputs=state.model_dump(),
        reads=plan.external_reads,
        writes=frozenset(writes),
        pinned=True,
        metadata={
            "agentslice": {
                "synthetic": True,
                "generated_by": "failed_hypothesis_folding",
                "source_annotation_schema_version": 1,
                "fold_id": plan.fold_id,
                "source_event_ids": list(plan.remove_event_ids),
                "hypothesis_event_ids": [plan.hypothesis_event_id],
                "evidence_event_ids": [item.event_id for item in plan.evidence],
                "conclusion_event_ids": list(plan.conclusion_event_ids),
                "annotator": {
                    "kind": plan.annotator_kind,
                    "name": plan.annotator_name,
                    "version": plan.annotator_version,
                },
            }
        },
    )


class FailedHypothesisFoldingPass(Pass):
    """Replace each resolved failed-hypothesis region with epistemic state."""

    name = "failed_hypothesis_folding"

    def apply(self, graph: CausalGraph, ctx: CompileContext) -> PassOutcome:
        events = graph.events_in_order()
        tokens_before = estimate_graph_tokens(events)
        notes: list[str] = []

        applicable: list[FoldPlan] = []
        synthetic_by_fold_id: dict[str, TraceEvent] = {}
        for plan in ctx.fold_plans:
            missing = [
                event_id for event_id in plan.remove_event_ids if event_id not in graph.events
            ]
            if missing:
                notes.append(
                    f"fold {plan.fold_id!r}: source events disappeared before folding: {missing}"
                )
                continue
            source_events = [graph.events[event_id] for event_id in plan.remove_event_ids]
            synthetic = _synthetic_event(plan, source_events[-1])
            source_tokens = estimate_graph_tokens(source_events)
            synthetic_tokens = estimate_event_tokens(synthetic)
            if synthetic_tokens >= source_tokens:
                notes.append(
                    f"fold {plan.fold_id!r}: skipped because it would not reduce tokens "
                    f"({source_tokens} -> {synthetic_tokens})"
                )
                continue
            applicable.append(plan)
            synthetic_by_fold_id[plan.fold_id] = synthetic

        plan_by_removed_id = {
            event_id: plan for plan in applicable for event_id in plan.remove_event_ids
        }
        plan_by_last_id = {plan.remove_event_ids[-1]: plan for plan in applicable}
        removed_ids: list[str] = []
        added_ids: list[str] = []
        new_events: list[TraceEvent] = []

        for event in events:
            current_plan = plan_by_removed_id.get(event.id)
            if current_plan is None:
                new_events.append(event)
                continue
            removed_ids.append(event.id)
            if event.id == current_plan.remove_event_ids[-1]:
                synthetic = synthetic_by_fold_id[plan_by_last_id[event.id].fold_id]
                new_events.append(synthetic)
                added_ids.append(synthetic.id)
                notes.append(
                    f"fold {current_plan.fold_id!r}: "
                    f"replaced {list(current_plan.remove_event_ids)} "
                    f"with {synthetic.id!r}"
                )

        new_graph = build_causal_graph(new_events)
        tokens_after = estimate_graph_tokens(new_events)
        report = CompilationReport(
            pass_name=self.name,
            events_before=len(events),
            events_after=len(new_events),
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            removed_event_ids=removed_ids,
            added_event_ids=added_ids,
            pinned_event_ids=added_ids,
            notes=notes,
        )
        return PassOutcome(
            graph=new_graph,
            ctx=replace(ctx, fold_plans=()),
            report=report,
        )
