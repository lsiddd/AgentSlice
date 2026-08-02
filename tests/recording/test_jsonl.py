from pathlib import Path

import pytest

from agentslice.errors import TraceFormatError, TraceValidationError
from agentslice.ir.events import EventType, TraceEvent
from agentslice.recording.jsonl import TraceReader, TraceWriter


def make_event(id: str, seq: int) -> TraceEvent:
    return TraceEvent(id=id, seq=seq, type=EventType.STATE_UPDATE)


def test_roundtrip_write_then_read(tmp_path: Path) -> None:
    path = tmp_path / "trace.jsonl"
    events = [make_event("a", 0), make_event("b", 1)]
    with TraceWriter(path) as writer:
        for event in events:
            writer.write(event)
    assert TraceReader(path).read_all() == events


def test_empty_file_yields_no_events(tmp_path: Path) -> None:
    path = tmp_path / "empty.jsonl"
    path.write_text("")
    assert TraceReader(path).read_all() == []


def test_corrupted_line_raises_trace_format_error_with_line_number(tmp_path: Path) -> None:
    path = tmp_path / "trace.jsonl"
    good = make_event("a", 0).model_dump_json()
    path.write_text(f"{good}\nnot json\n")
    with pytest.raises(TraceFormatError, match=r":2: invalid JSON"):
        TraceReader(path).read_all()


def test_missing_required_field_raises_trace_validation_error(tmp_path: Path) -> None:
    path = tmp_path / "trace.jsonl"
    path.write_text('{"id": "a"}\n')
    with pytest.raises(TraceValidationError, match=r":1: invalid trace event"):
        TraceReader(path).read_all()


def test_writer_is_a_context_manager_and_closes_the_file(tmp_path: Path) -> None:
    path = tmp_path / "trace.jsonl"
    with TraceWriter(path) as writer:
        writer.write(make_event("a", 0))
    assert len(TraceReader(path).read_all()) == 1


def test_append_mode_preserves_existing_lines(tmp_path: Path) -> None:
    path = tmp_path / "trace.jsonl"
    with TraceWriter(path) as writer:
        writer.write(make_event("a", 0))
    with TraceWriter(path, append=True) as writer:
        writer.write(make_event("b", 1))
    assert [e.id for e in TraceReader(path).read_all()] == ["a", "b"]
