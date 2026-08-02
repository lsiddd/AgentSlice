from agentslice.compiler.base import CompilationReport, CompileContext, CompiledContext, ToolSchema
from agentslice.ir.events import EventType, TraceEvent


def test_compile_context_defaults() -> None:
    ctx = CompileContext()
    assert ctx.budget_tokens is None
    assert ctx.tool_catalog == {}
    assert ctx.strict is False
    assert ctx.strict_schema is False
    assert ctx.anchor_event_id is None


def test_tool_schema_defaults() -> None:
    schema = ToolSchema(name="get_weather")
    assert schema.description == ""
    assert schema.parameters == {}


def test_compilation_report_list_fields_default_empty() -> None:
    report = CompilationReport(
        pass_name="noop", events_before=1, events_after=1, tokens_before=1, tokens_after=1
    )
    assert report.removed_event_ids == []
    assert report.modified_event_ids == []
    assert report.pinned_event_ids == []
    assert report.notes == []


def test_compiled_context_round_trip() -> None:
    events = [TraceEvent(id="a", seq=0, type=EventType.STATE_UPDATE)]
    compiled = CompiledContext(events=events, tokens_before=10, tokens_after=5)
    assert compiled.events == events
    assert compiled.budget_satisfied is None
