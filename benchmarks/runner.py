"""Live orchestration of one BFCL task against a real model, under one context policy.

Unlike ``agentslice.replay``, which resends an *already recorded* trace and
compares one next action, this drives the whole multi-turn conversation
live: for each turn, ask the policy what to send, call the model, execute
any tool calls against a simulated environment, feed the results back, and
repeat until the model gives a plain-text answer for that turn. The trace
is rebuilt fresh from scratch every iteration via ``from_openai_messages``
— the same "simple and correct over incremental patching" choice the
compiler pipeline itself makes.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from agentslice.compiler.base import ToolEffect, ToolSchema
from agentslice.compiler.tokens import estimate_tokens
from agentslice.recording.openai_adapter import from_openai_messages
from agentslice.replay.comparator import next_action_equivalence
from agentslice.replay.runtime import ReplaySession
from benchmarks.bfcl.schema import BFCLTask
from benchmarks.cache import ResponseCache
from benchmarks.environments.base import Environment, safe_call
from benchmarks.ground_truth import ParsedCall, param_names_from_tool_catalog, parse_turn
from benchmarks.metrics import (
    PriceTable,
    TaskOutcome,
    TurnOutcome,
    constraint_retention,
    estimate_cost_usd,
)
from benchmarks.policies import ContextPolicy, FullTracePolicy


@dataclass(frozen=True)
class RunnerConfig:
    """Knobs that trade thoroughness for cost.

    ``check_determinism`` re-issues the final turn's request an extra
    ``determinism_samples`` times, uncached, purely to compare responses —
    it multiplies that turn's cost and should stay off by default.
    """

    max_tool_iterations_per_turn: int = 8
    check_determinism: bool = False
    determinism_samples: int = 2
    price: PriceTable = field(
        default_factory=lambda: PriceTable(prompt_usd_per_1k=0.0, completion_usd_per_1k=0.0)
    )


class BenchmarkRunner:
    """Runs one (task, policy, model) combination end to end."""

    def __init__(
        self,
        session: ReplaySession,
        model: str,
        policy: ContextPolicy,
        tool_catalog: dict[str, ToolSchema],
        tools_payload: list[dict[str, Any]],
        environment_factory: Callable[[dict[str, Any]], Environment],
        *,
        cache: ResponseCache | None = None,
        config: RunnerConfig | None = None,
    ) -> None:
        self._session = session
        self._model = model
        self._policy = policy
        self._tool_catalog = tool_catalog
        self._side_effect_tools = {
            name for name, schema in tool_catalog.items() if schema.effects is ToolEffect.EFFECTFUL
        }
        self._param_names = param_names_from_tool_catalog(tools_payload)
        self._environment_factory = environment_factory
        self._cache = cache
        self._config = config or RunnerConfig()

    def run_task(self, task: BFCLTask) -> TaskOutcome:
        model_env = self._environment_factory(task.initial_config)
        ground_truth_env = self._environment_factory(task.initial_config)
        full_trace_policy = FullTracePolicy()

        messages: list[dict[str, Any]] = []
        turns: list[TurnOutcome] = []
        prompt_tokens_total = 0
        completion_tokens_total = 0
        synthetic_id_counter = 0
        last_request_messages: list[dict[str, Any]] = []
        last_request_tools: list[dict[str, Any]] | None = None
        last_response: dict[str, Any] = {}

        for turn_index, turn_messages in enumerate(task.turns):
            messages.extend(turn_messages)
            model_calls: list[ParsedCall] = []
            invalid_call_count = 0
            context_tokens = 0
            full_trace_tokens = 0

            for _ in range(self._config.max_tool_iterations_per_turn):
                events = from_openai_messages(messages, side_effect_tools=self._side_effect_tools)
                full_request = full_trace_policy.build_request(events, self._tool_catalog)
                request = self._policy.build_request(events, self._tool_catalog)
                context_tokens = request.tokens
                full_trace_tokens = full_request.tokens
                prompt_tokens_total += request.tokens

                response = self._next_action(request.messages, request.tools)
                completion_tokens_total += estimate_tokens(json.dumps(response, default=str))
                last_request_messages = request.messages
                last_request_tools = request.tools
                last_response = response

                raw_tool_calls = response.get("tool_calls") or []
                if not raw_tool_calls:
                    messages.append({"role": "assistant", "content": response.get("content") or ""})
                    break

                tool_calls, synthetic_id_counter = _normalize_tool_calls(
                    raw_tool_calls, synthetic_id_counter
                )
                messages.append({"role": "assistant", "content": None, "tool_calls": tool_calls})
                for call in tool_calls:
                    function = call["function"]
                    name = function["name"]
                    try:
                        kwargs = json.loads(function["arguments"] or "{}")
                    except json.JSONDecodeError:
                        kwargs = {}
                    result, is_valid = safe_call(model_env, name, kwargs)
                    if not is_valid:
                        invalid_call_count += 1
                    model_calls.append(ParsedCall(name=name, kwargs=kwargs))
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call["id"],
                            "content": json.dumps(result),
                        }
                    )

            ground_truth_calls = parse_turn(task.ground_truth[turn_index], self._param_names)
            for gt_call in ground_truth_calls:
                ground_truth_env.call(gt_call.name, gt_call.kwargs)

            turns.append(
                TurnOutcome(
                    turn_index=turn_index,
                    model_calls=model_calls,
                    ground_truth_calls=ground_truth_calls,
                    invalid_call_count=invalid_call_count,
                    context_tokens=context_tokens,
                    full_trace_tokens=full_trace_tokens,
                )
            )

        end_to_end_success = model_env.snapshot() == ground_truth_env.snapshot()

        full_trace_events = from_openai_messages(
            messages, side_effect_tools=self._side_effect_tools
        )
        baseline_messages = full_trace_policy.build_request(
            full_trace_events, self._tool_catalog
        ).messages
        retention = constraint_retention(baseline_messages, last_request_messages)

        determinism_rate = None
        if self._config.check_determinism and turns:
            determinism_rate = self._check_determinism(
                last_request_messages, last_request_tools, last_response
            )

        return TaskOutcome(
            task_id=task.id,
            policy_name=self._policy.name,
            model=self._model,
            turns=turns,
            end_to_end_success=end_to_end_success,
            estimated_cost_usd=estimate_cost_usd(
                prompt_tokens_total, completion_tokens_total, self._config.price
            ),
            determinism_rate=determinism_rate,
            constraint_retention=retention,
        )

    def _next_action(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None
    ) -> dict[str, Any]:
        if self._cache is None:
            return self._session.next_action(messages, tools=tools)
        key = ResponseCache.key_for(self._model, messages, tools)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        response = self._session.next_action(messages, tools=tools)
        self._cache.set(key, response)
        return response

    def _check_determinism(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        original_response: dict[str, Any],
    ) -> float:
        matches = 0
        for _ in range(self._config.determinism_samples):
            response = self._session.next_action(messages, tools=tools)
            if next_action_equivalence(original_response, response):
                matches += 1
        return matches / self._config.determinism_samples


def _normalize_tool_calls(
    raw_tool_calls: list[dict[str, Any]], counter: int
) -> tuple[list[dict[str, Any]], int]:
    """Fill in a synthetic ``id``/``type`` for a live response that omits them.

    A real provider's response is expected to be fully OpenAI-shaped, but
    this loop drives external, uncontrolled model output turn after turn —
    a live boundary, unlike a recorded trace already validated on the way
    in. ``from_openai_messages`` requires every tool call to have an
    ``id``; without this, one omission from a lenient provider would abort
    the entire task.
    """
    normalized: list[dict[str, Any]] = []
    for call in raw_tool_calls:
        function = call.get("function", {})
        call_id = call.get("id")
        if not call_id:
            call_id = f"call_{counter}"
            counter += 1
        normalized.append(
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": function.get("name", ""),
                    "arguments": function.get("arguments") or "{}",
                },
            }
        )
    return normalized, counter
