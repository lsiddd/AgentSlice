"""Metrics computed from a run of the benchmark harness.

Names follow the list in ``NEXT_STEPS.md``'s Marco 10. Two of them need a
definition beyond what that list states, so they're pinned down here:

- **next-action equivalence** reuses ``agentslice.replay.comparator``'s
  definition: a turn's *entire* multiset of (name, JSON-normalized
  arguments) pairs must match the ground truth's, exactly.
- **argument equivalence** is finer-grained and only defined where next-
  action equivalence already failed to be all-or-nothing: of the calls
  whose *name* had a same-named counterpart in the ground truth (so the
  model at least attempted the right kind of operation), what fraction
  also had matching arguments? This isolates "picked the right tool" from
  "filled it in correctly."

**constraint retention** reports ``None`` (not a rate of 0) when the
underlying trace carries no ``system`` message to retain in the first
place — the honest answer for the bundled BFCL fixture, whose tasks never
include one, rather than a manufactured 100%.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from typing import Any

from benchmarks.ground_truth import ParsedCall


def _name_multiset(calls: list[ParsedCall]) -> Counter[str]:
    return Counter(call.name for call in calls)


def _full_multiset(calls: list[ParsedCall]) -> Counter[tuple[str, str]]:
    return Counter(
        (call.name, json.dumps(call.kwargs, sort_keys=True, default=str)) for call in calls
    )


@dataclass(frozen=True)
class TurnOutcome:
    turn_index: int
    model_calls: list[ParsedCall]
    ground_truth_calls: list[ParsedCall]
    invalid_call_count: int
    context_tokens: int
    full_trace_tokens: int

    @property
    def next_action_equivalent(self) -> bool:
        return _full_multiset(self.model_calls) == _full_multiset(self.ground_truth_calls)

    @property
    def argument_equivalence(self) -> float | None:
        model_names = _name_multiset(self.model_calls)
        gt_names = _name_multiset(self.ground_truth_calls)
        name_overlap = sum((model_names & gt_names).values())
        if name_overlap == 0:
            return None
        model_full = _full_multiset(self.model_calls)
        gt_full = _full_multiset(self.ground_truth_calls)
        full_overlap = sum((model_full & gt_full).values())
        return full_overlap / name_overlap

    @property
    def extra_call_count(self) -> int:
        model_full = _full_multiset(self.model_calls)
        gt_full = _full_multiset(self.ground_truth_calls)
        matched = sum((model_full & gt_full).values())
        return max(0, len(self.model_calls) - matched)

    @property
    def context_reduction(self) -> float:
        if self.full_trace_tokens == 0:
            return 0.0
        return 1 - (self.context_tokens / self.full_trace_tokens)


@dataclass(frozen=True)
class TaskOutcome:
    task_id: str
    policy_name: str
    model: str
    turns: list[TurnOutcome]
    end_to_end_success: bool
    estimated_cost_usd: float
    determinism_rate: float | None = None
    constraint_retention: float | None = None

    @property
    def total_model_calls(self) -> int:
        return sum(len(turn.model_calls) for turn in self.turns)

    @property
    def total_invalid_calls(self) -> int:
        return sum(turn.invalid_call_count for turn in self.turns)

    @property
    def total_extra_calls(self) -> int:
        return sum(turn.extra_call_count for turn in self.turns)

    @property
    def next_action_equivalence_rate(self) -> float:
        if not self.turns:
            return 0.0
        return sum(1 for turn in self.turns if turn.next_action_equivalent) / len(self.turns)


@dataclass(frozen=True)
class PriceTable:
    """Estimated USD cost per 1,000 tokens. See :func:`estimate_cost_usd`."""

    prompt_usd_per_1k: float
    completion_usd_per_1k: float


def estimate_cost_usd(prompt_tokens: int, completion_tokens: int, price: PriceTable) -> float:
    """Approximate a call's cost from AgentSlice's own token *estimate*, not real billing.

    Same heuristic-first philosophy as ``agentslice.compiler.tokens``: good
    enough to compare policies against each other, not to reconcile against
    an actual invoice.
    """
    return (prompt_tokens / 1000) * price.prompt_usd_per_1k + (
        completion_tokens / 1000
    ) * price.completion_usd_per_1k


@dataclass(frozen=True)
class AggregateReport:
    policy_name: str
    model: str
    task_count: int
    end_to_end_success_rate: float
    next_action_equivalence_rate: float
    argument_equivalence_rate: float | None
    context_reduction: float
    invalid_call_rate: float
    extra_call_rate: float
    cost_per_successful_task_usd: float | None
    constraint_retention: float | None


def aggregate(outcomes: list[TaskOutcome]) -> list[AggregateReport]:
    """Group task outcomes by (policy, model) and compute per-group rates."""
    groups: dict[tuple[str, str], list[TaskOutcome]] = {}
    for outcome in outcomes:
        groups.setdefault((outcome.policy_name, outcome.model), []).append(outcome)

    reports: list[AggregateReport] = []
    for (policy_name, model), group in groups.items():
        all_turns = [turn for task in group for turn in task.turns]
        total_calls = sum(task.total_model_calls for task in group)
        total_gt_calls = sum(len(turn.ground_truth_calls) for turn in all_turns)
        total_context_tokens = sum(turn.context_tokens for turn in all_turns)
        total_full_trace_tokens = sum(turn.full_trace_tokens for turn in all_turns)
        argument_equivalences = [
            turn.argument_equivalence for turn in all_turns if turn.argument_equivalence is not None
        ]
        successful = [task for task in group if task.end_to_end_success]
        constraint_retentions = [
            task.constraint_retention for task in group if task.constraint_retention is not None
        ]

        reports.append(
            AggregateReport(
                policy_name=policy_name,
                model=model,
                task_count=len(group),
                end_to_end_success_rate=len(successful) / len(group),
                next_action_equivalence_rate=(
                    sum(task.next_action_equivalence_rate for task in group) / len(group)
                ),
                argument_equivalence_rate=(
                    sum(argument_equivalences) / len(argument_equivalences)
                    if argument_equivalences
                    else None
                ),
                context_reduction=(
                    1 - (total_context_tokens / total_full_trace_tokens)
                    if total_full_trace_tokens
                    else 0.0
                ),
                invalid_call_rate=(
                    sum(task.total_invalid_calls for task in group) / total_calls
                    if total_calls
                    else 0.0
                ),
                extra_call_rate=(
                    sum(task.total_extra_calls for task in group) / total_gt_calls
                    if total_gt_calls
                    else 0.0
                ),
                cost_per_successful_task_usd=(
                    sum(task.estimated_cost_usd for task in successful) / len(successful)
                    if successful
                    else None
                ),
                constraint_retention=(
                    sum(constraint_retentions) / len(constraint_retentions)
                    if constraint_retentions
                    else None
                ),
            )
        )
    return reports


def constraint_retention(
    baseline_messages: list[dict[str, Any]], policy_messages: list[dict[str, Any]]
) -> float | None:
    """Fraction of `system`-role messages in `baseline_messages` also present in `policy_messages`.

    Returns `None`, not `0.0`, when `baseline_messages` has no `system`
    message to begin with: the metric doesn't apply, and reporting a rate
    would imply a constraint was dropped when none ever existed.
    """
    baseline_system = [
        message.get("content") for message in baseline_messages if message.get("role") == "system"
    ]
    if not baseline_system:
        return None
    policy_system = {
        message.get("content") for message in policy_messages if message.get("role") == "system"
    }
    retained = sum(1 for content in baseline_system if content in policy_system)
    return retained / len(baseline_system)
