"""The agentslice command line interface: record, compile, diff."""

from __future__ import annotations

import json
import os
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.table import Table

from agentslice.__about__ import __version__
from agentslice.compiler.base import CompiledContext, ToolSchema
from agentslice.compiler.pipeline import compile_graph
from agentslice.errors import (
    AdapterError,
    AgentSliceError,
    CLIUsageError,
    CompilerError,
    ReplayError,
    TraceError,
)
from agentslice.ir.graph import build_causal_graph
from agentslice.recording.claude_code_adapter import from_claude_code_transcript
from agentslice.recording.codex_adapter import from_codex_rollout
from agentslice.recording.jsonl import TraceReader, TraceWriter
from agentslice.recording.openai_adapter import from_openai_messages
from agentslice.replay import (
    ReplaySession,
    extract_next_recorded_action,
    next_action_equivalence,
    replay_compiled_context,
)

app = typer.Typer(add_completion=False, no_args_is_help=True)

_err_console = Console(stderr=True)
_out_console = Console()


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(__version__)
        raise typer.Exit()


@app.callback()
def main(
    ctx: typer.Context,
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose", "-v", help="Print a full traceback in addition to the error message."
        ),
    ] = False,
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            callback=_version_callback,
            is_eager=True,
            help="Show the version and exit.",
        ),
    ] = False,
) -> None:
    """AgentSlice: a causal context compiler for tool-using agents."""
    ctx.obj = {"verbose": verbose}


def _exit_code_for(exc: AgentSliceError) -> int:
    if isinstance(exc, TraceError | AdapterError | ReplayError):
        return 3
    if isinstance(exc, CompilerError):
        return 4
    return 2


def _handle_errors(fn: Callable[[], None], *, verbose: bool) -> None:
    try:
        fn()
    except AgentSliceError as exc:
        typer.secho(f"error: {exc}", fg=typer.colors.RED, err=True)
        if verbose:
            traceback.print_exc()
        raise typer.Exit(code=_exit_code_for(exc)) from None


_RECORD_FORMATS = ("openai", "claude-code", "codex")


def _read_json_array(path: Path) -> list[Any]:
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise CLIUsageError(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(data, list):
        raise CLIUsageError(f"{path}: expected a JSON array of messages")
    return data


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for lineno, raw_line in enumerate(path.read_text().splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise CLIUsageError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
    return records


@app.command()
def record(
    ctx: typer.Context,
    input_path: Annotated[
        Path,
        typer.Option(
            "--input",
            exists=True,
            readable=True,
            help="Message log to convert. A JSON array for --format openai; JSON Lines "
            "(one record per line) for --format claude-code or codex.",
        ),
    ],
    output_path: Annotated[
        Path, typer.Option("--output", help="Where to write the resulting trace (JSON Lines).")
    ],
    format: Annotated[
        str,
        typer.Option(
            "--format",
            help="Input message format: 'openai', 'claude-code', or 'codex'.",
        ),
    ] = "openai",
) -> None:
    """Convert a message log into a trace."""
    verbose = ctx.obj["verbose"]

    def run() -> None:
        if format == "openai":
            events = from_openai_messages(_read_json_array(input_path))
        elif format == "claude-code":
            events = from_claude_code_transcript(_read_jsonl(input_path))
        elif format == "codex":
            events = from_codex_rollout(_read_jsonl(input_path))
        else:
            raise CLIUsageError(
                f"unsupported --format {format!r}: expected one of {_RECORD_FORMATS}"
            )

        with TraceWriter(output_path) as writer:
            for event in events:
                writer.write(event)
        typer.echo(f"recorded {len(events)} events -> {output_path}")

    _handle_errors(run, verbose=verbose)


def _parse_tool_catalog(raw: Any) -> dict[str, ToolSchema]:
    if not isinstance(raw, list):
        raise CLIUsageError("tool catalog must be a JSON array of OpenAI-style tool definitions")
    catalog: dict[str, ToolSchema] = {}
    for item in raw:
        try:
            function = item["function"]
            schema = ToolSchema(
                name=function["name"],
                description=function.get("description", ""),
                parameters=function.get("parameters", {}),
            )
        except (KeyError, TypeError) as exc:
            raise CLIUsageError(f"malformed tool definition: {exc}") from exc
        catalog[schema.name] = schema
    return catalog


def _print_compile_report(compiled: CompiledContext) -> None:
    table = Table(title="agentslice compile")
    table.add_column("pass")
    table.add_column("events before", justify="right")
    table.add_column("events after", justify="right")
    table.add_column("tokens before", justify="right")
    table.add_column("tokens after", justify="right")
    for r in compiled.reports:
        table.add_row(
            r.pass_name,
            str(r.events_before),
            str(r.events_after),
            str(r.tokens_before),
            str(r.tokens_after),
        )
    _err_console.print(table)

    summary = f"total: {compiled.tokens_before} -> {compiled.tokens_after} tokens"
    if compiled.budget_tokens is not None:
        status = "OK" if compiled.budget_satisfied else "OVER BUDGET"
        summary += f" (budget {compiled.budget_tokens}, {status})"
    _err_console.print(summary)


@app.command(name="compile")
def compile_cmd(
    ctx: typer.Context,
    trace: Annotated[
        Path, typer.Argument(exists=True, readable=True, help="Trace file to compile.")
    ],
    output: Annotated[
        Path | None,
        typer.Option("-o", "--output", help="Where to write the compiled trace (default: stdout)."),
    ] = None,
    budget: Annotated[
        int | None, typer.Option("--budget", help="Token budget for the compiled context.")
    ] = None,
    tools: Annotated[
        Path | None,
        typer.Option("--tools", exists=True, readable=True, help="JSON file with a tools array."),
    ] = None,
    strict: Annotated[
        bool, typer.Option("--strict", help="Fail if the token budget isn't met after every pass.")
    ] = False,
    strict_schema: Annotated[
        bool, typer.Option("--strict-schema", help="Fail if a used tool has no entry in --tools.")
    ] = False,
    report: Annotated[
        Path | None,
        typer.Option("--report", help="Where to write the per-pass compilation report."),
    ] = None,
) -> None:
    """Compile a trace into the smallest context that preserves its causal state."""
    verbose = ctx.obj["verbose"]

    def run() -> None:
        events = TraceReader(trace).read_all()
        graph = build_causal_graph(events)

        tool_catalog = None
        if tools is not None:
            try:
                raw_tools = json.loads(tools.read_text())
            except json.JSONDecodeError as exc:
                raise CLIUsageError(f"{tools}: invalid JSON: {exc}") from exc
            tool_catalog = _parse_tool_catalog(raw_tools)

        compiled = compile_graph(
            graph,
            budget_tokens=budget,
            tool_catalog=tool_catalog,
            strict=strict,
            strict_schema=strict_schema,
        )

        _print_compile_report(compiled)

        report_path = report
        if output is not None:
            with TraceWriter(output) as writer:
                for event in compiled.events:
                    writer.write(event)
            typer.echo(f"compiled {len(compiled.events)} events -> {output}", err=True)
            if report_path is None:
                report_path = Path(f"{output}.report.json")
        else:
            for event in compiled.events:
                typer.echo(event.model_dump_json())

        if report_path is not None:
            report_path.write_text(json.dumps([r.model_dump() for r in compiled.reports], indent=2))

    _handle_errors(run, verbose=verbose)


def _print_diff_table(rows: list[dict[str, str]], token_deltas: list[dict[str, Any]]) -> None:
    table = Table(title="agentslice diff")
    table.add_column("event")
    table.add_column("status")
    table.add_column("pass")
    for row in rows:
        table.add_row(row["id"], row["status"], row.get("pass", ""))
    _out_console.print(table)

    if token_deltas:
        totals = Table(title="token deltas by pass")
        totals.add_column("pass")
        totals.add_column("tokens before", justify="right")
        totals.add_column("tokens after", justify="right")
        for delta in token_deltas:
            totals.add_row(
                str(delta["pass"]), str(delta["tokens_before"]), str(delta["tokens_after"])
            )
        _out_console.print(totals)


@app.command()
def diff(
    ctx: typer.Context,
    original: Annotated[Path, typer.Argument(exists=True, readable=True)],
    compiled: Annotated[Path, typer.Argument(exists=True, readable=True)],
    report: Annotated[
        Path | None,
        typer.Option(
            "--report",
            exists=True,
            readable=True,
            help="Per-pass compilation report produced by `compile --report`.",
        ),
    ] = None,
    format: Annotated[
        str, typer.Option("--format", help="Output format: 'table' or 'json'.")
    ] = "table",
) -> None:
    """Compare an original trace against a compiled one, event by event."""
    verbose = ctx.obj["verbose"]

    def run() -> None:
        if format not in ("table", "json"):
            raise CLIUsageError(f"unsupported --format {format!r}: expected 'table' or 'json'")

        original_events = {event.id: event for event in TraceReader(original).read_all()}
        compiled_events = {event.id: event for event in TraceReader(compiled).read_all()}

        rows: list[dict[str, str]] = []
        for event_id, original_event in original_events.items():
            compiled_event = compiled_events.get(event_id)
            if compiled_event is None:
                status = "removed"
            elif compiled_event.pinned and not original_event.pinned:
                status = "pinned"
            elif compiled_event != original_event:
                status = "modified"
            else:
                status = "kept"
            rows.append({"id": event_id, "status": status})

        token_deltas: list[dict[str, Any]] = []
        if report is not None:
            report_data = json.loads(report.read_text())
            pass_by_event: dict[str, str] = {}
            for entry in report_data:
                changed_ids = [
                    *entry.get("removed_event_ids", []),
                    *entry.get("modified_event_ids", []),
                ]
                for event_id in changed_ids:
                    pass_by_event[event_id] = entry["pass_name"]
                token_deltas.append(
                    {
                        "pass": entry["pass_name"],
                        "tokens_before": entry["tokens_before"],
                        "tokens_after": entry["tokens_after"],
                    }
                )
            for row in rows:
                if row["status"] in ("removed", "modified") and row["id"] in pass_by_event:
                    row["pass"] = pass_by_event[row["id"]]

        if format == "json":
            payload: dict[str, Any] = {"events": rows}
            if token_deltas:
                payload["token_deltas"] = token_deltas
            typer.echo(json.dumps(payload, indent=2))
        else:
            _print_diff_table(rows, token_deltas)

    _handle_errors(run, verbose=verbose)


def _resolve_api_key(env_var: str) -> str:
    api_key = os.environ.get(env_var)
    if not api_key:
        raise CLIUsageError(f"environment variable {env_var!r} is not set")
    return api_key


def _load_tool_catalog(path: Path | None) -> dict[str, ToolSchema] | None:
    if path is None:
        return None
    try:
        raw_tools = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise CLIUsageError(f"{path}: invalid JSON: {exc}") from exc
    return _parse_tool_catalog(raw_tools)


def _print_replay_result(replayed: dict[str, Any], original_action: dict[str, Any] | None) -> None:
    _out_console.print(json.dumps(replayed, indent=2))
    if original_action is None:
        _err_console.print("(no comparable next action recorded in the original trace)")
        return
    equivalent = next_action_equivalence(original_action, replayed)
    status = "EQUIVALENT" if equivalent else "DIFFERENT"
    _err_console.print(f"original next action: {json.dumps(original_action, indent=2)}")
    _err_console.print(f"comparison: {status}")


_BASE_URL_OPTION = typer.Option("--base-url", help="OpenAI-compatible API base URL.")
_API_KEY_ENV_OPTION = typer.Option(
    "--api-key-env", help="Environment variable holding the API key."
)
_TOOLS_OPTION = typer.Option(
    "--tools", exists=True, readable=True, help="JSON file with an OpenAI-style tools array."
)
_TIMEOUT_OPTION = typer.Option("--timeout", help="Seconds to wait for the model's response.")


@app.command()
def replay(
    ctx: typer.Context,
    trace: Annotated[
        Path, typer.Argument(exists=True, readable=True, help="Compiled trace to replay.")
    ],
    compare_with: Annotated[
        Path,
        typer.Option(
            "--compare-with",
            exists=True,
            readable=True,
            help=(
                "The original, uncompiled trace `trace` was derived from: source for filling "
                "in any pending tool results and ground truth for the equivalence comparison."
            ),
        ),
    ],
    model: Annotated[str, typer.Option("--model", help="Model to send the replayed context to.")],
    base_url: Annotated[str, _BASE_URL_OPTION] = "https://openrouter.ai/api/v1",
    api_key_env: Annotated[str, _API_KEY_ENV_OPTION] = "OPENROUTER_API_KEY",
    tools: Annotated[Path | None, _TOOLS_OPTION] = None,
    timeout: Annotated[float, _TIMEOUT_OPTION] = 120.0,
) -> None:
    """Resend a compiled trace to a real model and compare its next action to the original.

    Note: a compiled trace file carries only events, not the tool catalog
    `schema_pruning` narrowed it to — pass the same (or a manually
    narrowed) `--tools` file again if the replayed model should be able to
    call tools.
    """
    verbose = ctx.obj["verbose"]

    def run() -> None:
        api_key = _resolve_api_key(api_key_env)
        events = TraceReader(trace).read_all()
        original_events = TraceReader(compare_with).read_all()
        tool_catalog = _load_tool_catalog(tools)

        with ReplaySession(base_url, api_key, model=model, timeout=timeout) as session:
            replayed = replay_compiled_context(
                events,
                original_events=original_events,
                session=session,
                tool_catalog=tool_catalog,
            )

        anchor_id = max(events, key=lambda e: e.seq).id if events else None
        original_action = (
            extract_next_recorded_action(original_events, anchor_id) if anchor_id else None
        )
        _print_replay_result(replayed, original_action)

    _handle_errors(run, verbose=verbose)


@app.command()
def fork(
    ctx: typer.Context,
    trace: Annotated[Path, typer.Argument(exists=True, readable=True, help="Full trace to fork.")],
    at: Annotated[str, typer.Option("--at", help="Event id to fork at.")],
    model: Annotated[str, typer.Option("--model", help="Model to send the forked context to.")],
    context_policy: Annotated[
        str,
        typer.Option(
            "--context-policy",
            help="How to build the forked context. Only 'causal' is supported today.",
        ),
    ] = "causal",
    budget: Annotated[
        int | None, typer.Option("--budget", help="Token budget for the forked context.")
    ] = None,
    base_url: Annotated[str, _BASE_URL_OPTION] = "https://openrouter.ai/api/v1",
    api_key_env: Annotated[str, _API_KEY_ENV_OPTION] = "OPENROUTER_API_KEY",
    tools: Annotated[Path | None, _TOOLS_OPTION] = None,
    timeout: Annotated[float, _TIMEOUT_OPTION] = 120.0,
) -> None:
    """Fork a trace at an event, compile the causal context up to it, and replay it to a model."""
    verbose = ctx.obj["verbose"]

    def run() -> None:
        if context_policy != "causal":
            raise CLIUsageError(
                f"unsupported --context-policy {context_policy!r}: only 'causal' is supported"
            )
        api_key = _resolve_api_key(api_key_env)
        original_events = TraceReader(trace).read_all()
        if at not in {event.id for event in original_events}:
            raise CLIUsageError(f"no event {at!r} in {trace}")

        graph = build_causal_graph(original_events)
        tool_catalog = _load_tool_catalog(tools)
        compiled = compile_graph(
            graph, budget_tokens=budget, tool_catalog=tool_catalog, anchor_event_id=at
        )

        with ReplaySession(base_url, api_key, model=model, timeout=timeout) as session:
            replayed = replay_compiled_context(
                compiled.events,
                original_events=original_events,
                session=session,
                tool_catalog=compiled.tool_catalog,
            )

        original_action = extract_next_recorded_action(original_events, at)
        _print_replay_result(replayed, original_action)

    _handle_errors(run, verbose=verbose)
