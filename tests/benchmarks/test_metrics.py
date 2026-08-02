from benchmarks.ground_truth import ParsedCall
from benchmarks.metrics import (
    AggregateReport,
    PriceTable,
    TaskOutcome,
    TurnOutcome,
    aggregate,
    constraint_retention,
    estimate_cost_usd,
)


def _turn(
    model_calls: list[ParsedCall],
    ground_truth_calls: list[ParsedCall],
    *,
    invalid: int = 0,
    context_tokens: int = 10,
    full_trace_tokens: int = 20,
) -> TurnOutcome:
    return TurnOutcome(
        turn_index=0,
        model_calls=model_calls,
        ground_truth_calls=ground_truth_calls,
        invalid_call_count=invalid,
        context_tokens=context_tokens,
        full_trace_tokens=full_trace_tokens,
    )


def test_next_action_equivalent_true_on_exact_multiset_match() -> None:
    turn = _turn(
        [ParsedCall("cd", {"folder": "a"})],
        [ParsedCall("cd", {"folder": "a"})],
    )
    assert turn.next_action_equivalent is True


def test_next_action_equivalent_false_on_different_arguments() -> None:
    turn = _turn(
        [ParsedCall("cd", {"folder": "b"})],
        [ParsedCall("cd", {"folder": "a"})],
    )
    assert turn.next_action_equivalent is False


def test_next_action_equivalent_is_multiplicity_sensitive() -> None:
    call = ParsedCall("cd", {"folder": "a"})
    turn = _turn([call, call], [call])
    assert turn.next_action_equivalent is False


def test_argument_equivalence_is_none_with_no_name_overlap() -> None:
    turn = _turn([ParsedCall("ls", {})], [ParsedCall("cd", {"folder": "a"})])
    assert turn.argument_equivalence is None


def test_argument_equivalence_partial_when_name_matches_but_args_differ() -> None:
    turn = _turn(
        [ParsedCall("cd", {"folder": "wrong"})],
        [ParsedCall("cd", {"folder": "right"})],
    )
    assert turn.argument_equivalence == 0.0


def test_argument_equivalence_full_when_everything_matches() -> None:
    call = ParsedCall("cd", {"folder": "a"})
    turn = _turn([call], [call])
    assert turn.argument_equivalence == 1.0


def test_extra_call_count_counts_unmatched_model_calls() -> None:
    turn = _turn(
        [ParsedCall("cd", {"folder": "a"}), ParsedCall("ls", {})],
        [ParsedCall("cd", {"folder": "a"})],
    )
    assert turn.extra_call_count == 1


def test_context_reduction_zero_when_full_trace_is_empty() -> None:
    turn = _turn([], [], context_tokens=0, full_trace_tokens=0)
    assert turn.context_reduction == 0.0


def test_context_reduction_positive_when_smaller_than_baseline() -> None:
    turn = _turn([], [], context_tokens=5, full_trace_tokens=20)
    assert turn.context_reduction == 0.75


def test_constraint_retention_is_none_without_a_system_message() -> None:
    assert constraint_retention([{"role": "user", "content": "hi"}], []) is None


def test_constraint_retention_full_when_system_message_survives() -> None:
    baseline = [{"role": "system", "content": "be nice"}, {"role": "user", "content": "hi"}]
    policy = [{"role": "system", "content": "be nice"}]
    assert constraint_retention(baseline, policy) == 1.0


def test_constraint_retention_zero_when_system_message_dropped() -> None:
    baseline = [{"role": "system", "content": "be nice"}]
    policy = [{"role": "user", "content": "hi"}]
    assert constraint_retention(baseline, policy) == 0.0


def test_estimate_cost_usd() -> None:
    price = PriceTable(prompt_usd_per_1k=1.0, completion_usd_per_1k=2.0)
    assert estimate_cost_usd(1000, 500, price) == 1.0 + 1.0


def test_aggregate_groups_by_policy_and_model_and_computes_rates() -> None:
    success = TaskOutcome(
        task_id="t1",
        policy_name="full_trace",
        model="m",
        turns=[_turn([ParsedCall("cd", {"folder": "a"})], [ParsedCall("cd", {"folder": "a"})])],
        end_to_end_success=True,
        estimated_cost_usd=0.5,
    )
    failure = TaskOutcome(
        task_id="t2",
        policy_name="full_trace",
        model="m",
        turns=[_turn([ParsedCall("cd", {"folder": "b"})], [ParsedCall("cd", {"folder": "a"})])],
        end_to_end_success=False,
        estimated_cost_usd=0.5,
    )
    reports = aggregate([success, failure])
    assert len(reports) == 1
    report: AggregateReport = reports[0]
    assert report.policy_name == "full_trace"
    assert report.task_count == 2
    assert report.end_to_end_success_rate == 0.5
    assert report.cost_per_successful_task_usd == 0.5


def test_aggregate_reports_none_cost_when_nothing_succeeded() -> None:
    failure = TaskOutcome(
        task_id="t1",
        policy_name="p",
        model="m",
        turns=[],
        end_to_end_success=False,
        estimated_cost_usd=1.0,
    )
    report = aggregate([failure])[0]
    assert report.cost_per_successful_task_usd is None
