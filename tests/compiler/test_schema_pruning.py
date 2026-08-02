import pytest

from agentslice.compiler.base import CompileContext, ToolSchema
from agentslice.compiler.schema_pruning import SchemaPruningPass
from agentslice.errors import UnknownToolError
from agentslice.ir.events import EventType, TraceEvent
from agentslice.ir.graph import build_causal_graph


def test_catalog_larger_than_used_tools_is_pruned() -> None:
    call = TraceEvent(id="c1", seq=0, type=EventType.TOOL_CALL, tool_name="get_weather")
    graph = build_causal_graph([call])
    ctx = CompileContext(
        tool_catalog={
            "get_weather": ToolSchema(name="get_weather"),
            "delete_file": ToolSchema(name="delete_file"),
        }
    )
    outcome = SchemaPruningPass().apply(graph, ctx)
    assert set(outcome.ctx.tool_catalog) == {"get_weather"}


def test_tool_used_without_catalog_entry_is_noted_by_default() -> None:
    call = TraceEvent(id="c1", seq=0, type=EventType.TOOL_CALL, tool_name="mystery_tool")
    graph = build_causal_graph([call])
    ctx = CompileContext(tool_catalog={"get_weather": ToolSchema(name="get_weather")})
    outcome = SchemaPruningPass().apply(graph, ctx)
    assert any("mystery_tool" in note for note in outcome.report.notes)


def test_tool_used_without_catalog_entry_raises_in_strict_schema_mode() -> None:
    call = TraceEvent(id="c1", seq=0, type=EventType.TOOL_CALL, tool_name="mystery_tool")
    graph = build_causal_graph([call])
    ctx = CompileContext(
        tool_catalog={"get_weather": ToolSchema(name="get_weather")}, strict_schema=True
    )
    with pytest.raises(UnknownToolError):
        SchemaPruningPass().apply(graph, ctx)


def test_empty_catalog_is_a_no_op() -> None:
    call = TraceEvent(id="c1", seq=0, type=EventType.TOOL_CALL, tool_name="get_weather")
    graph = build_causal_graph([call])
    outcome = SchemaPruningPass().apply(graph, CompileContext())
    assert outcome.ctx.tool_catalog == {}
    assert outcome.graph is graph
