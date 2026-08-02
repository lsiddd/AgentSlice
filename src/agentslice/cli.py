"""The agentslice command line interface: record, compile, diff."""

from __future__ import annotations

import json
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
    TraceError,
)
from agentslice.ir.graph import build_causal_graph
from agentslice.recording.jsonl import TraceReader, TraceWriter
from agentslice.recording.openai_adapter import from_openai_messages

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
    if isinstance(exc, TraceError | AdapterError):
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


@app.command()
def record(
    ctx: typer.Context,
    input_path: Annotated[
        Path,
        typer.Option(
            "--input",
            exists=True,
            readable=True,
            help="JSON file holding an array of OpenAI-compatible chat messages.",
        ),
    ],
    output_path: Annotated[
        Path, typer.Option("--output", help="Where to write the resulting trace (JSON Lines).")
    ],
    format: Annotated[
        str,
        typer.Option("--format", help="Input message format. Only 'openai' is supported."),
    ] = "openai",
) -> None:
    """Convert a message log into a trace."""
    verbose = ctx.obj["verbose"]

    def run() -> None:
        if format != "openai":
            raise CLIUsageError(f"unsupported --format {format!r}: only 'openai' is supported")
        try:
            data = json.loads(input_path.read_text())
        except json.JSONDecodeError as exc:
            raise CLIUsageError(f"{input_path}: invalid JSON: {exc}") from exc
        if not isinstance(data, list):
            raise CLIUsageError(f"{input_path}: expected a JSON array of messages")

        events = from_openai_messages(data)
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
