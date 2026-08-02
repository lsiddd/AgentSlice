import json
from pathlib import Path

import pytest
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


def test_replay_help() -> None:
    result = runner.invoke(app, ["replay", "--help"])
    assert result.exit_code == 0


def test_fork_help() -> None:
    result = runner.invoke(app, ["fork", "--help"])
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


def test_compile_rejects_invalid_tool_effect_classification(tmp_path: Path) -> None:
    trace = tmp_path / "trace.jsonl"
    with TraceWriter(trace) as writer:
        writer.write(TraceEvent(id="a", seq=0, type=EventType.USER_GOAL))
    tools = tmp_path / "tools.json"
    tools.write_text(
        json.dumps(
            [
                {
                    "type": "function",
                    "effects": "probably-pure",
                    "function": {"name": "get_weather"},
                }
            ]
        )
    )

    result = runner.invoke(app, ["compile", str(trace), "--tools", str(tools)])

    assert result.exit_code == 2
    assert "malformed tool definition" in result.output


def test_compile_rejects_unknown_experimental_pass(tmp_path: Path) -> None:
    trace = tmp_path / "trace.jsonl"
    with TraceWriter(trace) as writer:
        writer.write(TraceEvent(id="a", seq=0, type=EventType.USER_GOAL))

    result = runner.invoke(
        app,
        ["compile", str(trace), "--enable-pass", "made_up"],
    )

    assert result.exit_code == 2
    assert "unsupported --enable-pass" in result.output


def test_compile_can_enable_failed_hypothesis_folding(tmp_path: Path) -> None:
    trace = tmp_path / "trace.jsonl"
    events = [
        TraceEvent(
            id="m1",
            seq=0,
            type=EventType.MODEL_MESSAGE,
            outputs={"content": "Maybe expired"},
            writes=frozenset({"conversation:current"}),
        ),
        TraceEvent(
            id="c1",
            seq=1,
            type=EventType.TOOL_CALL,
            tool_name="check_token",
            writes=frozenset({"tool_call:c1", "conversation:current"}),
        ),
        TraceEvent(
            id="r1",
            seq=2,
            type=EventType.TOOL_RESULT,
            tool_name="check_token",
            outputs={"valid": True},
            reads=frozenset({"tool_call:c1"}),
            writes=frozenset({"tool_result:c1", "conversation:current"}),
        ),
        TraceEvent(
            id="m2",
            seq=3,
            type=EventType.MODEL_MESSAGE,
            outputs={"content": "Ruled out"},
            reads=frozenset({"tool_result:c1"}),
            writes=frozenset({"conversation:current"}),
            metadata={
                "agentslice": {
                    "fold": {
                        "schema_version": 1,
                        "fold_id": "fh_token",
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
                                "value": True,
                            }
                        ],
                        "remove_event_ids": ["m1", "c1", "r1", "m2"],
                        "conclusion_event_ids": ["m2"],
                        "dedicated_conclusion": True,
                        "annotator": {
                            "kind": "runtime",
                            "name": "test",
                            "version": "1",
                        },
                    }
                }
            },
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
    with TraceWriter(trace) as writer:
        for event in events:
            writer.write(event)
    tools = tmp_path / "tools.json"
    tools.write_text(
        json.dumps(
            [
                {
                    "type": "function",
                    "effects": "pure",
                    "function": {"name": "check_token"},
                }
            ]
        )
    )
    output = tmp_path / "compiled.jsonl"

    result = runner.invoke(
        app,
        [
            "compile",
            str(trace),
            "--tools",
            str(tools),
            "--enable-pass",
            "failed_hypothesis_folding",
            "-o",
            str(output),
        ],
    )

    assert result.exit_code == 0
    compiled = output.read_text()
    assert '"id":"fold_fh_token"' in compiled
    assert '"type":"state_update"' in compiled
    assert '"id":"c1"' not in compiled
    report = json.loads(Path(f"{output}.report.json").read_text())
    assert [entry["pass_name"] for entry in report[:4]] == [
        "constraint_pinning",
        "current_turn_retention",
        "fold_annotation_resolution",
        "failed_hypothesis_folding",
    ]
    assert report[3]["added_event_ids"] == ["fold_fh_token"]


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


def test_diff_reports_synthetic_events_as_added(tmp_path: Path) -> None:
    original = tmp_path / "original.jsonl"
    compiled = tmp_path / "compiled.jsonl"
    report = tmp_path / "report.json"
    original_event = TraceEvent(id="a", seq=0, type=EventType.MODEL_MESSAGE)
    synthetic = TraceEvent(id="fold_x", seq=1, type=EventType.STATE_UPDATE)
    with TraceWriter(original) as writer:
        writer.write(original_event)
    with TraceWriter(compiled) as writer:
        writer.write(original_event)
        writer.write(synthetic)
    report.write_text(
        json.dumps(
            [
                {
                    "pass_name": "failed_hypothesis_folding",
                    "events_before": 1,
                    "events_after": 2,
                    "tokens_before": 1,
                    "tokens_after": 2,
                    "added_event_ids": ["fold_x"],
                }
            ]
        )
    )

    result = runner.invoke(
        app,
        [
            "diff",
            str(original),
            str(compiled),
            "--report",
            str(report),
            "--format",
            "json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["events"][-1] == {
        "id": "fold_x",
        "status": "added",
        "pass": "failed_hypothesis_folding",
    }


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


def test_replay_missing_api_key_env_exits_2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    trace = tmp_path / "trace.jsonl"
    original = tmp_path / "original.jsonl"
    with TraceWriter(trace) as writer:
        writer.write(TraceEvent(id="a", seq=0, type=EventType.USER_GOAL))
    with TraceWriter(original) as writer:
        writer.write(TraceEvent(id="a", seq=0, type=EventType.USER_GOAL))

    result = runner.invoke(
        app,
        ["replay", str(trace), "--compare-with", str(original), "--model", "gpt-test"],
    )

    assert result.exit_code == 2


def test_fork_invalid_context_policy_exits_2(tmp_path: Path) -> None:
    trace = tmp_path / "trace.jsonl"
    with TraceWriter(trace) as writer:
        writer.write(TraceEvent(id="a", seq=0, type=EventType.USER_GOAL))

    result = runner.invoke(
        app,
        [
            "fork",
            str(trace),
            "--at",
            "a",
            "--model",
            "gpt-test",
            "--context-policy",
            "last-n",
        ],
    )

    assert result.exit_code == 2


def test_fork_missing_at_event_exits_2(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    trace = tmp_path / "trace.jsonl"
    with TraceWriter(trace) as writer:
        writer.write(TraceEvent(id="a", seq=0, type=EventType.USER_GOAL))

    result = runner.invoke(app, ["fork", str(trace), "--at", "missing", "--model", "gpt-test"])

    assert result.exit_code == 2
    assert "no event 'missing'" in result.output
