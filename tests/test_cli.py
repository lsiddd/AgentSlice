import json
from pathlib import Path

from typer.testing import CliRunner

from agentslice.__about__ import __version__
from agentslice.cli import app
from agentslice.ir.events import EventType, TraceEvent
from agentslice.recording.jsonl import TraceWriter

runner = CliRunner()


def test_version_flag() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.output


def test_top_level_help() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0


def test_record_help() -> None:
    result = runner.invoke(app, ["record", "--help"])
    assert result.exit_code == 0


def test_compile_help() -> None:
    result = runner.invoke(app, ["compile", "--help"])
    assert result.exit_code == 0


def test_diff_help() -> None:
    result = runner.invoke(app, ["diff", "--help"])
    assert result.exit_code == 0


def test_record_writes_trace_and_reports_count(tmp_path: Path) -> None:
    messages = tmp_path / "messages.json"
    messages.write_text(json.dumps([{"role": "user", "content": "hi"}]))
    output = tmp_path / "trace.jsonl"

    result = runner.invoke(app, ["record", "--input", str(messages), "--output", str(output)])

    assert result.exit_code == 0
    assert "recorded 1 events" in result.output
    assert output.exists()


def test_record_missing_input_file_exits_2(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "record",
            "--input",
            str(tmp_path / "missing.json"),
            "--output",
            str(tmp_path / "o.jsonl"),
        ],
    )
    assert result.exit_code == 2


def test_record_invalid_format_exits_2(tmp_path: Path) -> None:
    messages = tmp_path / "messages.json"
    messages.write_text("[]")
    result = runner.invoke(
        app,
        [
            "record",
            "--input",
            str(messages),
            "--output",
            str(tmp_path / "o.jsonl"),
            "--format",
            "bogus",
        ],
    )
    assert result.exit_code == 2


def test_record_malformed_messages_exits_3(tmp_path: Path) -> None:
    messages = tmp_path / "messages.json"
    messages.write_text(json.dumps([{"role": "tool", "tool_call_id": "missing", "content": "{}"}]))
    result = runner.invoke(
        app, ["record", "--input", str(messages), "--output", str(tmp_path / "o.jsonl")]
    )
    assert result.exit_code == 3


def test_compile_writes_output_and_default_report(tmp_path: Path) -> None:
    trace = tmp_path / "trace.jsonl"
    with TraceWriter(trace) as writer:
        writer.write(TraceEvent(id="a", seq=0, type=EventType.USER_GOAL, outputs={"content": "hi"}))
    output = tmp_path / "compiled.jsonl"

    result = runner.invoke(app, ["compile", str(trace), "-o", str(output)])

    assert result.exit_code == 0
    assert output.exists()
    assert Path(f"{output}.report.json").exists()


def test_compile_without_output_writes_jsonl_to_stdout(tmp_path: Path) -> None:
    trace = tmp_path / "trace.jsonl"
    with TraceWriter(trace) as writer:
        writer.write(TraceEvent(id="a", seq=0, type=EventType.USER_GOAL, outputs={"content": "hi"}))

    result = runner.invoke(app, ["compile", str(trace)])

    assert result.exit_code == 0
    assert '"id":"a"' in result.output


def test_compile_missing_trace_file_exits_2(tmp_path: Path) -> None:
    result = runner.invoke(app, ["compile", str(tmp_path / "missing.jsonl")])
    assert result.exit_code == 2


def test_compile_malformed_trace_exits_3(tmp_path: Path) -> None:
    trace = tmp_path / "trace.jsonl"
    trace.write_text("not json\n")
    result = runner.invoke(app, ["compile", str(trace)])
    assert result.exit_code == 3


def test_compile_strict_budget_not_satisfiable_exits_4(tmp_path: Path) -> None:
    trace = tmp_path / "trace.jsonl"
    with TraceWriter(trace) as writer:
        writer.write(
            TraceEvent(id="a", seq=0, type=EventType.TOOL_RESULT, outputs={"x": "y" * 1000})
        )

    result = runner.invoke(app, ["compile", str(trace), "--budget", "1", "--strict"])

    assert result.exit_code == 4


def test_compile_strict_schema_unknown_tool_exits_4(tmp_path: Path) -> None:
    trace = tmp_path / "trace.jsonl"
    with TraceWriter(trace) as writer:
        writer.write(TraceEvent(id="a", seq=0, type=EventType.TOOL_CALL, tool_name="mystery"))
    tools = tmp_path / "tools.json"
    tools.write_text(json.dumps([{"type": "function", "function": {"name": "get_weather"}}]))

    result = runner.invoke(app, ["compile", str(trace), "--tools", str(tools), "--strict-schema"])

    assert result.exit_code == 4


def test_diff_reports_removed_and_kept_events(tmp_path: Path) -> None:
    original = tmp_path / "original.jsonl"
    compiled = tmp_path / "compiled.jsonl"
    kept = TraceEvent(id="a", seq=0, type=EventType.USER_GOAL, outputs={"content": "hi"})
    removed = TraceEvent(id="b", seq=1, type=EventType.STATE_UPDATE)
    with TraceWriter(original) as writer:
        writer.write(kept)
        writer.write(removed)
    with TraceWriter(compiled) as writer:
        writer.write(kept)

    result = runner.invoke(app, ["diff", str(original), str(compiled)])

    assert result.exit_code == 0
    assert "removed" in result.output
    assert "kept" in result.output


def test_diff_json_format(tmp_path: Path) -> None:
    original = tmp_path / "original.jsonl"
    compiled = tmp_path / "compiled.jsonl"
    event = TraceEvent(id="a", seq=0, type=EventType.USER_GOAL)
    with TraceWriter(original) as writer:
        writer.write(event)
    with TraceWriter(compiled) as writer:
        writer.write(event)

    result = runner.invoke(app, ["diff", str(original), str(compiled), "--format", "json"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["events"] == [{"id": "a", "status": "kept"}]


def test_diff_invalid_format_exits_2(tmp_path: Path) -> None:
    original = tmp_path / "original.jsonl"
    compiled = tmp_path / "compiled.jsonl"
    event = TraceEvent(id="a", seq=0, type=EventType.USER_GOAL)
    with TraceWriter(original) as writer:
        writer.write(event)
    with TraceWriter(compiled) as writer:
        writer.write(event)

    result = runner.invoke(app, ["diff", str(original), str(compiled), "--format", "bogus"])

    assert result.exit_code == 2


def test_diff_missing_file_exits_2(tmp_path: Path) -> None:
    missing = tmp_path / "missing.jsonl"
    other = tmp_path / "b.jsonl"
    result = runner.invoke(app, ["diff", str(missing), str(other)])
    assert result.exit_code == 2


def test_verbose_flag_prints_traceback_on_error(tmp_path: Path) -> None:
    trace = tmp_path / "trace.jsonl"
    trace.write_text("not json\n")

    result = runner.invoke(app, ["--verbose", "compile", str(trace)])

    assert result.exit_code == 3
    assert "Traceback" in result.output
