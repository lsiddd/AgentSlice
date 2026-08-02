"""Pass: narrow the tool catalog down to tools actually used in the graph."""

from __future__ import annotations

from dataclasses import replace

from agentslice.compiler.base import CompilationReport, CompileContext, Pass, PassOutcome
from agentslice.compiler.tokens import estimate_graph_tokens
from agentslice.errors import UnknownToolError
from agentslice.ir.graph import CausalGraph

_NOOP_CATALOG_NOTE = "no tool catalog supplied, nothing to prune"


class SchemaPruningPass(Pass):
    """Drops tool schemas that no surviving event actually invokes.

    A tool that was used but has no entry in the catalog is noted in the
    report by default; with ``ctx.strict_schema=True`` it raises
    :class:`~agentslice.errors.UnknownToolError` instead — including when
    the catalog is empty and something was used, since an empty catalog is
    just the limit case of "no entry for the tools in use", not a reason to
    exempt it. Without ``strict_schema``, an empty catalog is a no-op: with
    no catalog there is nothing to prune, and the graph is returned
    unchanged. This pass never touches events or their token cost, only
    ``ctx.tool_catalog``; schema token accounting is out of scope for v0.1.
    """

    name = "schema_pruning"

    def apply(self, graph: CausalGraph, ctx: CompileContext) -> PassOutcome:
        events = graph.events_in_order()
        tokens = estimate_graph_tokens(events)
        used_tool_names = {event.tool_name for event in events if event.tool_name}

        if not ctx.tool_catalog:
            if ctx.strict_schema and used_tool_names:
                raise UnknownToolError(
                    f"tools {sorted(used_tool_names)} were used but the tool catalog is empty"
                )
            report = CompilationReport(
                pass_name=self.name,
                events_before=len(events),
                events_after=len(events),
                tokens_before=tokens,
                tokens_after=tokens,
                notes=[_NOOP_CATALOG_NOTE],
            )
            return PassOutcome(graph=graph, ctx=ctx, report=report)

        notes: list[str] = []
        for tool_name in sorted(used_tool_names):
            if tool_name not in ctx.tool_catalog:
                if ctx.strict_schema:
                    raise UnknownToolError(
                        f"tool {tool_name!r} was used but has no entry in the tool catalog"
                    )
                notes.append(f"tool {tool_name!r} used but missing from catalog")

        pruned_catalog = {
            name: schema for name, schema in ctx.tool_catalog.items() if name in used_tool_names
        }
        pruned_names = sorted(set(ctx.tool_catalog) - set(pruned_catalog))
        if pruned_names:
            notes.append(f"pruned unused tools: {pruned_names}")

        new_ctx = replace(ctx, tool_catalog=pruned_catalog)
        report = CompilationReport(
            pass_name=self.name,
            events_before=len(events),
            events_after=len(events),
            tokens_before=tokens,
            tokens_after=tokens,
            notes=notes,
        )
        return PassOutcome(graph=graph, ctx=new_ctx, report=report)
