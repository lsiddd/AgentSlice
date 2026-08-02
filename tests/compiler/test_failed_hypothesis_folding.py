from __future__ import annotations

import json
from typing import Any

import pytest

from agentslice.compiler.base import CompileContext, ToolEffect, ToolSchema
from agentslice.compiler.failed_hypothesis_folding import (
    FailedHypothesisFoldingPass,
    FoldAnnotationResolutionPass,
)
from agentslice.compiler.pipeline import (
    EXPERIMENTAL_FAILED_HYPOTHESIS_FOLDING_PASSES,
    Pipeline,
    compile_graph,
)
from agentslice.ir.events import EventType, TraceEvent
from agentslice.ir.graph import build_causal_graph
from agentslice.recording.openai_adapter import to_openai_messages


def _annotation(
    *,
    schema_version: int = 1,
    annotator_kind: str = "runtime",
    evidence_value: Any = True,
    dedicated_conclusion: bool = True,
) -> dict[str, Any]:
    return {
        "agentslice": {
            "fold": {
                "schema_version": schema_version,
                "fold_id": "fh_token_expired",
                "kind": "ruled_out_hypothesis",
                "hypothesis": {
                    "text": "The token expired",
                    "source_event_id": "m1",
                },
                "evidence": [
                    {
                        "event_id": "r1",
                        "json_pointer": "/valid",
                        "operator": "==",
                        "value": evidence_value,
                    }
                ],
                "remove_event_ids": ["m1", "c1", "r1", "m2"],
                "conclusion_event_ids": ["m2"],
                "dedicated_conclusion": dedicated_conclusion,
                "annotator": {
                    "kind": annotator_kind,
                    "name": "test-runtime",
                    "version": "1.0",
                },
            }
        }
    }


def _events(*, annotation: dict[str, Any] | None = None) -> list[TraceEvent]:
    return [
        TraceEvent(
            id="m1",
            seq=0,
            type=EventType.MODEL_MESSAGE,
            outputs={"content": "Maybe the token expired"},
            reads=frozenset({"goal:current"}),
            writes=frozenset({"conversation:current"}),
        ),
        TraceEvent(
            id="c1",
            seq=1,
            type=EventType.TOOL_CALL,
            tool_name="check_token",
            inputs={"token": "abc"},
            writes=frozenset({"tool_call:c1", "conversation:current"}),
        ),
        TraceEvent(
            id="r1",
            seq=2,
            type=EventType.TOOL_RESULT,
            tool_name="check_token",
            outputs={"valid": True, "subject": {"id": 7}},
            reads=frozenset({"tool_call:c1"}),
            writes=frozenset({"tool_result:c1", "conversation:current"}),
        ),
        TraceEvent(
            id="m2",
            seq=3,
            type=EventType.MODEL_MESSAGE,
            outputs={"content": "The token is valid; hypothesis ruled out"},
            reads=frozenset({"tool_result:c1"}),
            writes=frozenset({"conversation:current"}),
            metadata=annotation or _annotation(),
        ),
        TraceEvent(
            id="anchor",
            seq=4,
            type=EventType.USER_GOAL,
            outputs={"content": "What should we try next?"},
            reads=frozenset({"conversation:current"}),
            writes=frozenset({"conversation:current", "user_goal:current"}),
        ),
    ]


def _context(
    *,
    effect: ToolEffect = ToolEffect.PURE,
    anchor_event_id: str | None = None,
) -> CompileContext:
    return CompileContext(
        tool_catalog={
            "check_token": ToolSchema(name="check_token", effects=effect),
        },
        anchor_event_id=anchor_event_id,
    )


def _resolve_and_fold(
    events: list[TraceEvent], ctx: CompileContext | None = None
) -> tuple[object, object]:
    graph = build_causal_graph(events)
    resolved = FoldAnnotationResolutionPass().apply(graph, ctx or _context())
    folded = FailedHypothesisFoldingPass().apply(resolved.graph, resolved.ctx)
    return resolved, folded


def test_valid_annotation_resolves_to_an_immutable_plan_and_folds() -> None:
    resolved, folded = _resolve_and_fold(_events())

    assert len(resolved.ctx.fold_plans) == 1
    plan = resolved.ctx.fold_plans[0]
    assert plan.fold_id == "fh_token_expired"
    assert plan.remove_event_ids == ("m1", "c1", "r1", "m2")
    assert plan.external_reads == frozenset({"goal:current"})

    synthetic = folded.graph.events["fold_fh_token_expired"]
    assert synthetic.type is EventType.STATE_UPDATE
    assert synthetic.seq == 3
    assert synthetic.pinned is True
    assert synthetic.reads == frozenset({"goal:current"})
    assert synthetic.writes == frozenset(
        {"conversation:current", "epistemic:ruled_out:fh_token_expired"}
    )
    assert synthetic.outputs == {
        "kind": "epistemic_state",
        "schema_version": 1,
        "ruled_out": [
            {
                "fold_id": "fh_token_expired",
                "hypothesis": "The token expired",
                "evidence": [
                    {
                        "event_id": "r1",
                        "json_pointer": "/valid",
                        "operator": "==",
                        "value": True,
                    }
                ],
            }
        ],
    }
    assert synthetic.metadata["agentslice"] == {
        "synthetic": True,
        "generated_by": "failed_hypothesis_folding",
        "source_annotation_schema_version": 1,
        "fold_id": "fh_token_expired",
        "source_event_ids": ["m1", "c1", "r1", "m2"],
        "hypothesis_event_ids": ["m1"],
        "evidence_event_ids": ["r1"],
        "conclusion_event_ids": ["m2"],
        "annotator": {"kind": "runtime", "name": "test-runtime", "version": "1.0"},
    }
    assert folded.report.removed_event_ids == ["m1", "c1", "r1", "m2"]
    assert folded.report.added_event_ids == ["fold_fh_token_expired"]
    assert folded.report.tokens_after < folded.report.tokens_before


def test_nested_json_pointer_evidence_is_verified() -> None:
    annotation = _annotation()
    annotation["agentslice"]["fold"]["evidence"][0].update(
        {"json_pointer": "/subject/id", "value": 7}
    )
    resolved, folded = _resolve_and_fold(_events(annotation=annotation))
    assert len(resolved.ctx.fold_plans) == 1
    assert "fold_fh_token_expired" in folded.graph.events


@pytest.mark.parametrize(
    ("effect", "expected_note"),
    [
        (ToolEffect.UNKNOWN, "not explicitly pure"),
        (ToolEffect.EFFECTFUL, "not explicitly pure"),
    ],
)
def test_unknown_or_effectful_tool_blocks_fold(effect: ToolEffect, expected_note: str) -> None:
    resolved, folded = _resolve_and_fold(_events(), _context(effect=effect))
    assert resolved.ctx.fold_plans == ()
    assert any(expected_note in note for note in resolved.report.notes)
    assert "fold_fh_token_expired" not in folded.graph.events


def test_missing_tool_schema_blocks_fold() -> None:
    resolved, _ = _resolve_and_fold(_events(), CompileContext())
    assert resolved.ctx.fold_plans == ()
    assert any("has no tool schema" in note for note in resolved.report.notes)


def test_side_effect_flag_blocks_fold_even_for_a_pure_tool() -> None:
    events = _events()
    events[2] = events[2].model_copy(update={"side_effects": True})
    resolved, _ = _resolve_and_fold(events)
    assert resolved.ctx.fold_plans == ()
    assert any("side effects" in note for note in resolved.report.notes)


def test_untrusted_annotator_is_ignored_by_default() -> None:
    resolved, _ = _resolve_and_fold(_events(annotation=_annotation(annotator_kind="llm")))
    assert resolved.ctx.fold_plans == ()
    assert any("annotator kind 'llm' is not accepted" in note for note in resolved.report.notes)


def test_untrusted_annotator_can_be_explicitly_accepted() -> None:
    ctx = _context()
    ctx = CompileContext(
        tool_catalog=ctx.tool_catalog,
        accepted_fold_annotators=frozenset({"llm"}),
    )
    resolved, folded = _resolve_and_fold(_events(annotation=_annotation(annotator_kind="llm")), ctx)
    assert len(resolved.ctx.fold_plans) == 1
    assert "fold_fh_token_expired" in folded.graph.events


def test_unknown_annotation_version_is_a_no_op_with_a_report_note() -> None:
    resolved, _ = _resolve_and_fold(_events(annotation=_annotation(schema_version=2)))
    assert resolved.ctx.fold_plans == ()
    assert any("schema_version" in note for note in resolved.report.notes)


def test_malformed_annotation_namespace_is_a_no_op() -> None:
    events = _events()
    events[3] = events[3].model_copy(update={"metadata": {"agentslice": "not-an-object"}})
    resolved, _ = _resolve_and_fold(events)
    assert resolved.ctx.fold_plans == ()
    assert any("namespace is not an object" in note for note in resolved.report.notes)


def test_duplicate_fold_ids_are_all_ignored() -> None:
    events = _events()
    events[0] = events[0].model_copy(update={"metadata": _annotation()})
    resolved, _ = _resolve_and_fold(events)
    assert resolved.ctx.fold_plans == ()
    assert any("duplicate fold_id" in note for note in resolved.report.notes)


@pytest.mark.parametrize(
    ("remove_ids", "expected_note"),
    [
        (["m1", "c1", "r1", "r1"], "contains duplicates"),
        (["m1", "c1", "r1", "missing"], "unknown events"),
    ],
)
def test_invalid_remove_event_ids_fail_closed(remove_ids: list[str], expected_note: str) -> None:
    annotation = _annotation()
    annotation["agentslice"]["fold"]["remove_event_ids"] = remove_ids
    resolved, _ = _resolve_and_fold(_events(annotation=annotation))
    assert resolved.ctx.fold_plans == ()
    assert any(expected_note in note for note in resolved.report.notes)


def test_region_with_non_operational_event_type_is_rejected() -> None:
    events = _events()
    events[0] = events[0].model_copy(update={"type": EventType.USER_GOAL})
    resolved, _ = _resolve_and_fold(events)
    assert resolved.ctx.fold_plans == ()
    assert any("unsupported event types" in note for note in resolved.report.notes)


def test_missing_explicit_anchor_is_reported_without_crashing() -> None:
    resolved, _ = _resolve_and_fold(_events(), _context(anchor_event_id="missing"))
    assert resolved.ctx.fold_plans == ()
    assert any("anchor 'missing' does not exist" in note for note in resolved.report.notes)


def test_hypothesis_source_must_be_a_model_message_inside_the_region() -> None:
    outside = _annotation()
    outside["agentslice"]["fold"]["hypothesis"]["source_event_id"] = "anchor"
    resolved_outside, _ = _resolve_and_fold(_events(annotation=outside))
    assert any(
        "hypothesis source event is outside" in note for note in resolved_outside.report.notes
    )

    wrong_type = _annotation()
    wrong_type["agentslice"]["fold"]["hypothesis"]["source_event_id"] = "c1"
    resolved_type, _ = _resolve_and_fold(_events(annotation=wrong_type))
    assert any(
        "hypothesis source event is not a model_message" in note
        for note in resolved_type.report.notes
    )


@pytest.mark.parametrize(
    ("conclusion_ids", "expected_note"),
    [
        (["m2", "m2"], "contains duplicates"),
        (["m1"], "final region event is not the final conclusion"),
        (["m2", "anchor"], "conclusion event is outside"),
        (["m2", "c1"], "must be a model_message"),
    ],
)
def test_conclusion_roles_are_validated(conclusion_ids: list[str], expected_note: str) -> None:
    annotation = _annotation()
    annotation["agentslice"]["fold"]["conclusion_event_ids"] = conclusion_ids
    resolved, _ = _resolve_and_fold(_events(annotation=annotation))
    assert resolved.ctx.fold_plans == ()
    assert any(expected_note in note for note in resolved.report.notes)


def test_annotation_must_be_on_the_ordered_final_conclusion() -> None:
    annotation = _annotation()
    annotation["agentslice"]["fold"]["conclusion_event_ids"] = ["m2", "m1"]
    resolved_order, _ = _resolve_and_fold(_events(annotation=annotation))
    assert any(
        "conclusion_event_ids is not ordered" in note for note in resolved_order.report.notes
    )

    events = _events()
    events[0] = events[0].model_copy(update={"metadata": _annotation()})
    events[3] = events[3].model_copy(update={"metadata": {}})
    resolved_host, _ = _resolve_and_fold(events)
    assert any(
        "annotation must be on the final conclusion" in note for note in resolved_host.report.notes
    )


def test_tool_call_and_result_must_be_structurally_linked() -> None:
    events = _events()
    events[1] = events[1].model_copy(update={"writes": frozenset({"conversation:current"})})
    resolved, _ = _resolve_and_fold(events)
    assert resolved.ctx.fold_plans == ()
    assert any("has no linked result" in note for note in resolved.report.notes)


def test_region_requires_a_tool_call_and_result() -> None:
    annotation = _annotation()
    annotation["agentslice"]["fold"]["remove_event_ids"] = ["m1", "m2"]
    events = [
        TraceEvent(id="m1", seq=0, type=EventType.MODEL_MESSAGE),
        TraceEvent(
            id="m2",
            seq=1,
            type=EventType.MODEL_MESSAGE,
            metadata=annotation,
        ),
        TraceEvent(id="anchor", seq=2, type=EventType.STATE_UPDATE),
    ]
    resolved, _ = _resolve_and_fold(events)
    assert resolved.ctx.fold_plans == ()
    assert any("at least one tool call and result" in note for note in resolved.report.notes)


def test_tool_call_requires_a_name() -> None:
    events = _events()
    events[1] = events[1].model_copy(update={"tool_name": None})
    resolved, _ = _resolve_and_fold(events)
    assert resolved.ctx.fold_plans == ()
    assert any("has no tool_name" in note for note in resolved.report.notes)


def test_every_tool_result_requires_a_linked_call() -> None:
    events = _events()
    events.insert(
        3,
        TraceEvent(
            id="orphan_result",
            seq=3,
            type=EventType.TOOL_RESULT,
            outputs={"ignored": True},
        ),
    )
    events = [event.model_copy(update={"seq": index}) for index, event in enumerate(events)]
    annotation = _annotation()
    annotation["agentslice"]["fold"]["remove_event_ids"] = [
        "m1",
        "c1",
        "r1",
        "orphan_result",
        "m2",
    ]
    events[4] = events[4].model_copy(update={"metadata": annotation})
    resolved, _ = _resolve_and_fold(events)
    assert resolved.ctx.fold_plans == ()
    assert any("has no linked call" in note for note in resolved.report.notes)


@pytest.mark.parametrize(
    ("pointer", "expected_note"),
    [
        ("valid", "must be empty or start"),
        ("/missing", "does not exist"),
        ("/valid/nope", "cannot descend"),
        ("/bad~2escape", "invalid JSON pointer escape"),
    ],
)
def test_invalid_evidence_pointer_fails_closed(pointer: str, expected_note: str) -> None:
    annotation = _annotation()
    annotation["agentslice"]["fold"]["evidence"][0]["json_pointer"] = pointer
    resolved, _ = _resolve_and_fold(_events(annotation=annotation))
    assert resolved.ctx.fold_plans == ()
    assert any(expected_note in note for note in resolved.report.notes)


def test_json_pointer_supports_array_indices_and_root_values() -> None:
    events = _events()
    events[2] = events[2].model_copy(update={"outputs": {"checks": [{"valid": True}]}})
    annotation = _annotation()
    annotation["agentslice"]["fold"]["evidence"][0].update(
        {"json_pointer": "/checks/0/valid", "value": True}
    )
    resolved_array, _ = _resolve_and_fold(
        events[:-2]
        + [
            events[3].model_copy(update={"metadata": annotation}),
            events[4],
        ]
    )
    assert len(resolved_array.ctx.fold_plans) == 1

    root_annotation = _annotation()
    root_annotation["agentslice"]["fold"]["evidence"][0].update(
        {
            "json_pointer": "",
            "value": {"valid": True, "subject": {"id": 7}},
        }
    )
    resolved_root, _ = _resolve_and_fold(_events(annotation=root_annotation))
    assert len(resolved_root.ctx.fold_plans) == 1


@pytest.mark.parametrize(
    ("pointer", "expected_note"),
    [
        ("/checks/nope", "list index 'nope' is invalid"),
        ("/checks/2", "list index 2 is out of range"),
    ],
)
def test_invalid_json_pointer_array_index_fails_closed(pointer: str, expected_note: str) -> None:
    events = _events()
    events[2] = events[2].model_copy(update={"outputs": {"checks": [True]}})
    annotation = _annotation()
    annotation["agentslice"]["fold"]["evidence"][0].update({"json_pointer": pointer, "value": True})
    events[3] = events[3].model_copy(update={"metadata": annotation})
    resolved, _ = _resolve_and_fold(events)
    assert resolved.ctx.fold_plans == ()
    assert any(expected_note in note for note in resolved.report.notes)


def test_evidence_must_be_a_tool_result_inside_the_region() -> None:
    outside = _annotation()
    outside["agentslice"]["fold"]["evidence"][0]["event_id"] = "anchor"
    resolved_outside, _ = _resolve_and_fold(_events(annotation=outside))
    assert any(
        "evidence event 'anchor' is outside" in note for note in resolved_outside.report.notes
    )

    wrong_type = _annotation()
    wrong_type["agentslice"]["fold"]["evidence"][0].update(
        {"event_id": "m1", "json_pointer": "/content", "value": "Maybe the token expired"}
    )
    resolved_type, _ = _resolve_and_fold(_events(annotation=wrong_type))
    assert any(
        "evidence event 'm1' is not a tool_result" in note for note in resolved_type.report.notes
    )


def test_duplicate_or_overlapping_semantic_roles_fail_closed() -> None:
    duplicate = _annotation()
    duplicate["agentslice"]["fold"]["evidence"].append(
        dict(duplicate["agentslice"]["fold"]["evidence"][0])
    )
    resolved_duplicate, _ = _resolve_and_fold(_events(annotation=duplicate))
    assert any(
        "evidence contains duplicate references" in note for note in resolved_duplicate.report.notes
    )

    overlap = _annotation()
    overlap["agentslice"]["fold"]["hypothesis"]["source_event_id"] = "m2"
    resolved_overlap, _ = _resolve_and_fold(_events(annotation=overlap))
    assert any(
        "hypothesis, evidence, and conclusion roles overlap" in note
        for note in resolved_overlap.report.notes
    )

    misordered = _annotation()
    misordered["agentslice"]["fold"]["remove_event_ids"] = ["c1", "r1", "m1", "m2"]
    events = _events(annotation=misordered)
    events = [
        events[1].model_copy(update={"seq": 0}),
        events[2].model_copy(update={"seq": 1}),
        events[0].model_copy(update={"seq": 2}),
        events[3].model_copy(update={"seq": 3}),
        events[4],
    ]
    resolved_order, _ = _resolve_and_fold(events)
    assert any(
        "hypothesis, evidence, and conclusion roles are out of order" in note
        for note in resolved_order.report.notes
    )


def test_evidence_mismatch_is_a_no_op() -> None:
    resolved, _ = _resolve_and_fold(_events(annotation=_annotation(evidence_value=False)))
    assert resolved.ctx.fold_plans == ()
    assert any("evidence predicate did not match" in note for note in resolved.report.notes)


def test_non_contiguous_region_is_a_no_op() -> None:
    events = _events()
    events.insert(
        2,
        TraceEvent(
            id="independent",
            seq=2,
            type=EventType.MODEL_MESSAGE,
            outputs={"content": "keep me"},
        ),
    )
    events = [event.model_copy(update={"seq": index}) for index, event in enumerate(events)]
    resolved, _ = _resolve_and_fold(events)
    assert resolved.ctx.fold_plans == ()
    assert any("not contiguous" in note for note in resolved.report.notes)


def test_anchor_or_pinned_event_in_region_blocks_fold() -> None:
    resolved_anchor, _ = _resolve_and_fold(_events(), _context(anchor_event_id="m2"))
    assert resolved_anchor.ctx.fold_plans == ()
    assert any("contains the anchor" in note for note in resolved_anchor.report.notes)

    events = _events()
    events[1] = events[1].model_copy(update={"pinned": True})
    resolved_pinned, _ = _resolve_and_fold(events)
    assert resolved_pinned.ctx.fold_plans == ()
    assert any("pinned event" in note for note in resolved_pinned.report.notes)


def test_region_after_an_explicit_anchor_blocks_fold() -> None:
    events = _events()
    events.insert(
        0,
        TraceEvent(id="early_anchor", seq=-1, type=EventType.STATE_UPDATE),
    )
    resolved, _ = _resolve_and_fold(events, _context(anchor_event_id="early_anchor"))
    assert resolved.ctx.fold_plans == ()
    assert any("occurs after the anchor" in note for note in resolved.report.notes)


def test_synthetic_event_id_collision_blocks_fold() -> None:
    events = _events()
    events.append(
        TraceEvent(
            id="fold_fh_token_expired",
            seq=5,
            type=EventType.STATE_UPDATE,
        )
    )
    resolved, _ = _resolve_and_fold(events)
    assert resolved.ctx.fold_plans == ()
    assert any("synthetic event id" in note for note in resolved.report.notes)


def test_overlapping_fold_regions_are_both_ignored() -> None:
    first = _events()
    first[-1] = first[-1].model_copy(update={"seq": 7})
    second_annotation = _annotation()
    second_fold = second_annotation["agentslice"]["fold"]
    second_fold.update(
        {
            "fold_id": "fh_second",
            "hypothesis": {
                "text": "A second hypothesis",
                "source_event_id": "m2",
            },
            "evidence": [
                {
                    "event_id": "r2",
                    "json_pointer": "/ok",
                    "operator": "==",
                    "value": True,
                }
            ],
            "remove_event_ids": ["m2", "c2", "r2", "m3"],
            "conclusion_event_ids": ["m3"],
        }
    )
    events = [
        *first[:-1],
        TraceEvent(
            id="c2",
            seq=4,
            type=EventType.TOOL_CALL,
            tool_name="check_token",
            writes=frozenset({"tool_call:c2", "conversation:current"}),
        ),
        TraceEvent(
            id="r2",
            seq=5,
            type=EventType.TOOL_RESULT,
            outputs={"ok": True},
            reads=frozenset({"tool_call:c2"}),
            writes=frozenset({"conversation:current"}),
        ),
        TraceEvent(
            id="m3",
            seq=6,
            type=EventType.MODEL_MESSAGE,
            writes=frozenset({"conversation:current"}),
            metadata=second_annotation,
        ),
        first[-1],
    ]
    resolved = FoldAnnotationResolutionPass().apply(build_causal_graph(events), _context())
    assert resolved.ctx.fold_plans == ()
    assert sum("region overlaps another fold" in note for note in resolved.report.notes) == 2


def test_substantive_outgoing_dependency_blocks_fold() -> None:
    events = _events()
    events[-1] = events[-1].model_copy(
        update={"reads": frozenset({"conversation:current", "tool_result:c1"})}
    )
    resolved, _ = _resolve_and_fold(events)
    assert resolved.ctx.fold_plans == ()
    assert any("outgoing dependency" in note for note in resolved.report.notes)


def test_mixed_conclusion_blocks_fold_without_text_inference() -> None:
    resolved, _ = _resolve_and_fold(_events(annotation=_annotation(dedicated_conclusion=False)))
    assert resolved.ctx.fold_plans == ()
    assert any("not explicitly dedicated" in note for note in resolved.report.notes)


def test_experimental_pipeline_is_idempotent() -> None:
    graph = build_causal_graph(_events())
    first = Pipeline(EXPERIMENTAL_FAILED_HYPOTHESIS_FOLDING_PASSES).run(graph, _context())
    second = Pipeline(EXPERIMENTAL_FAILED_HYPOTHESIS_FOLDING_PASSES).run(
        build_causal_graph(first.events), _context()
    )
    assert first.events == second.events
    assert [event.id for event in second.events].count("fold_fh_token_expired") == 1


def test_experimental_pipeline_output_is_replayable() -> None:
    compiled = Pipeline(EXPERIMENTAL_FAILED_HYPOTHESIS_FOLDING_PASSES).run(
        build_causal_graph(_events()), _context()
    )
    messages = to_openai_messages(compiled.events)
    assert messages[0]["role"] == "assistant"
    assert json.loads(messages[0]["content"])["_agentslice"] == {
        "kind": "epistemic_state",
        "version": 1,
    }
    assert messages[-1] == {
        "role": "user",
        "content": "What should we try next?",
    }


def test_fold_is_skipped_when_structured_state_would_not_reduce_tokens() -> None:
    annotation = _annotation()
    annotation["agentslice"]["fold"]["hypothesis"]["text"] = "x" * 10_000
    _, folded = _resolve_and_fold(_events(annotation=annotation))
    assert "fold_fh_token_expired" not in folded.graph.events
    assert folded.report.removed_event_ids == []
    assert any("would not reduce tokens" in note for note in folded.report.notes)


def test_fold_plan_is_ignored_if_a_source_disappears_before_application() -> None:
    graph = build_causal_graph(_events())
    resolved = FoldAnnotationResolutionPass().apply(graph, _context())
    graph_without_source = build_causal_graph([event for event in _events() if event.id != "r1"])
    folded = FailedHypothesisFoldingPass().apply(graph_without_source, resolved.ctx)
    assert folded.graph.events == graph_without_source.events
    assert any("source events disappeared" in note for note in folded.report.notes)


def test_default_pipeline_does_not_apply_experimental_fold() -> None:
    compiled = compile_graph(
        build_causal_graph(_events()),
        tool_catalog={"check_token": ToolSchema(name="check_token", effects=ToolEffect.PURE)},
    )
    assert all(
        report.pass_name not in {"fold_annotation_resolution", "failed_hypothesis_folding"}
        for report in compiled.reports
    )
    assert "fold_fh_token_expired" not in {event.id for event in compiled.events}
