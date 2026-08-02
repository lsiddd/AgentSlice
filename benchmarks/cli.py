"""Command-line entry point for the benchmark harness.

Not part of the installed `agentslice` package (see `NEXT_STEPS.md`, Marco
10) — run it from the repo root with `uv run python -m benchmarks.cli
<command>`, never installed as a console script.
"""

from __future__ import annotations

import json
import os
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any

import typer

from agentslice.compiler.base import ToolEffect, ToolSchema
from agentslice.errors import AgentSliceError
from agentslice.replay.runtime import ReplaySession
from benchmarks.bfcl.loader import (
    DEFAULT_TASKS_PATH,
    load_gorilla_file_system_tool_catalog,
    load_tasks,
)
from benchmarks.cache import ResponseCache
from benchmarks.environments.gorilla_file_system import create as create_gorilla_file_system
from benchmarks.errors import BenchmarkError
from benchmarks.metrics import PriceTable, TaskOutcome, aggregate
from benchmarks.policies import (
    CausalCompilePolicy,
    ContextPolicy,
    FullTracePolicy,
    LastNTurnsPolicy,
    LLMSummaryPolicy,
    RollingStatePolicy,
    Summarizer,
)
from benchmarks.runner import BenchmarkRunner, RunnerConfig

app = typer.Typer(add_completion=False, no_args_is_help=True)

_POLICY_NAMES = ("full_trace", "last_n_turns", "rolling_state", "causal_compile", "llm_summary")
_DEFAULT_POLICIES = ("full_trace", "causal_compile")


def _handle_errors(fn: Callable[[], None], *, verbose: bool) -> None:
    try:
        fn()
    except (AgentSliceError, BenchmarkError) as exc:
        typer.secho(f"error: {exc}", fg=typer.colors.RED, err=True)
        if verbose:
            traceback.print_exc()
        raise typer.Exit(code=1) from None


@app.command("list-tasks")
def list_tasks() -> None:
    """Print every bundled BFCL task's id and turn count."""
    for task in load_tasks():
        typer.echo(f"{task.id}\t{len(task.turns)} turns")


def _load_tool_catalog() -> tuple[dict[str, ToolSchema], list[dict[str, Any]]]:
    tools_payload = load_gorilla_file_system_tool_catalog()
    catalog: dict[str, ToolSchema] = {}
    for tool in tools_payload:
        function = tool["function"]
        catalog[function["name"]] = ToolSchema(
            name=function["name"],
            description=function.get("description", ""),
            parameters=function.get("parameters", {}),
            effects=ToolEffect(tool.get("effects", "unknown")),
        )
    return catalog, tools_payload


def _make_summarizer(session: ReplaySession) -> Summarizer:
    def summarize(messages: list[dict[str, Any]]) -> str:
        prompt = [
            {
                "role": "system",
                "content": (
                    "Summarize the conversation so far in 2-3 sentences, keeping every "
                    "fact a future turn might still need."
                ),
            },
            *messages,
        ]
        response = session.next_action(prompt, tools=None)
        return str(response.get("content") or "")

    return summarize


def _build_policy(name: str, session: ReplaySession) -> ContextPolicy:
    if name == "full_trace":
        return FullTracePolicy()
    if name == "last_n_turns":
        return LastNTurnsPolicy(n_turns=1)
    if name == "rolling_state":
        return RollingStatePolicy()
    if name == "causal_compile":
        return CausalCompilePolicy()
    if name == "llm_summary":
        return LLMSummaryPolicy(summarizer=_make_summarizer(session))
    raise BenchmarkError(f"unknown policy {name!r}, choose from {_POLICY_NAMES}")


def _outcome_to_dict(outcome: TaskOutcome) -> dict[str, Any]:
    return {
        "task_id": outcome.task_id,
        "policy": outcome.policy_name,
        "model": outcome.model,
        "end_to_end_success": outcome.end_to_end_success,
        "estimated_cost_usd": outcome.estimated_cost_usd,
        "determinism_rate": outcome.determinism_rate,
        "constraint_retention": outcome.constraint_retention,
        "next_action_equivalence_rate": outcome.next_action_equivalence_rate,
        "turns": [
            {
                "turn_index": turn.turn_index,
                "model_calls": [{"name": c.name, "kwargs": c.kwargs} for c in turn.model_calls],
                "ground_truth_calls": [
                    {"name": c.name, "kwargs": c.kwargs} for c in turn.ground_truth_calls
                ],
                "invalid_call_count": turn.invalid_call_count,
                "context_tokens": turn.context_tokens,
                "full_trace_tokens": turn.full_trace_tokens,
                "next_action_equivalent": turn.next_action_equivalent,
                "argument_equivalence": turn.argument_equivalence,
            }
            for turn in outcome.turns
        ],
    }


@app.command()
def run(
    model: Annotated[str, typer.Option(help="Model id to send to the API.")],
    policy: Annotated[
        list[str] | None,
        typer.Option("--policy", help=f"One or more of: {', '.join(_POLICY_NAMES)}."),
    ] = None,
    api_key_env: Annotated[
        str, typer.Option(help="Environment variable holding the API key.")
    ] = "OPENROUTER_API_KEY",
    base_url: Annotated[str, typer.Option()] = "https://openrouter.ai/api/v1",
    limit: Annotated[
        int, typer.Option(help=f"Number of bundled tasks to run (there are {13}).")
    ] = 3,
    cache_dir: Annotated[Path, typer.Option()] = Path("benchmarks/.cache/responses"),
    output: Annotated[
        Path | None, typer.Option(help="Write per-task results as JSON here.")
    ] = None,
    check_determinism: Annotated[
        bool, typer.Option(help="Re-issue each task's final turn twice more, uncached, to check.")
    ] = False,
    prompt_price_per_1k: Annotated[
        float, typer.Option(help="USD per 1k prompt tokens, for the cost estimate.")
    ] = 0.0,
    completion_price_per_1k: Annotated[
        float, typer.Option(help="USD per 1k completion tokens, for the cost estimate.")
    ] = 0.0,
    timeout: Annotated[float, typer.Option()] = 120.0,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Run the benchmark against a real model. This spends real API budget.

    Every request is cached under `cache_dir` by exact payload hash, so
    re-running the same (policy, model, task) triple after an interruption
    costs nothing. There is no built-in spending cap beyond `--limit` and
    the number of policies requested — choose both deliberately.
    """

    def do_run() -> None:
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise BenchmarkError(f"environment variable {api_key_env!r} is not set")

        policy_names = policy or list(_DEFAULT_POLICIES)
        unknown = set(policy_names) - set(_POLICY_NAMES)
        if unknown:
            raise BenchmarkError(f"unknown --policy value(s) {sorted(unknown)}: {_POLICY_NAMES}")

        tasks = load_tasks()[:limit]
        tool_catalog, tools_payload = _load_tool_catalog()
        cache = ResponseCache(cache_dir)
        price = PriceTable(
            prompt_usd_per_1k=prompt_price_per_1k, completion_usd_per_1k=completion_price_per_1k
        )

        typer.echo(
            f"Running {len(tasks)} task(s) from {DEFAULT_TASKS_PATH.name} x "
            f"{len(policy_names)} polic{'y' if len(policy_names) == 1 else 'ies'} "
            f"against {model!r}. Responses are cached under {cache_dir}.",
            err=True,
        )

        outcomes: list[TaskOutcome] = []
        with ReplaySession(base_url, api_key, model=model, timeout=timeout) as session:
            for policy_name in policy_names:
                context_policy = _build_policy(policy_name, session)
                runner = BenchmarkRunner(
                    session,
                    model,
                    context_policy,
                    tool_catalog,
                    tools_payload,
                    create_gorilla_file_system,
                    cache=cache,
                    config=RunnerConfig(check_determinism=check_determinism, price=price),
                )
                for task in tasks:
                    outcome = runner.run_task(task)
                    outcomes.append(outcome)
                    status = "OK" if outcome.end_to_end_success else "FAIL"
                    typer.echo(f"[{policy_name}] {task.id}: {status}", err=True)

        for report in aggregate(outcomes):
            typer.echo(
                f"{report.policy_name:15s} success={report.end_to_end_success_rate:.0%} "
                f"next_action={report.next_action_equivalence_rate:.0%} "
                f"context_reduction={report.context_reduction:.0%} "
                f"invalid_calls={report.invalid_call_rate:.0%}"
            )

        if output is not None:
            output.write_text(
                json.dumps(
                    [_outcome_to_dict(outcome) for outcome in outcomes], indent=2, default=str
                )
            )
            typer.echo(f"wrote {len(outcomes)} task result(s) -> {output}", err=True)

    _handle_errors(do_run, verbose=verbose)


if __name__ == "__main__":
    app()
