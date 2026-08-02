"""Contracts shared by every compiler pass: context, reports, and results."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from agentslice.compiler.fold_plans import FoldPlan
from agentslice.ir.events import TraceEvent
from agentslice.ir.graph import CausalGraph


class ToolEffect(StrEnum):
    """Whether invoking a tool is known to mutate state outside the trace."""

    PURE = "pure"
    EFFECTFUL = "effectful"
    UNKNOWN = "unknown"


class ToolSchema(BaseModel):
    """A tool's name, parameter schema, and explicit effect classification.

    Mirrors the shape of an OpenAI-style ``function`` tool definition
    closely enough to build a catalog directly from one. ``effects``
    deliberately defaults to ``unknown`` rather than assuming that a tool
    is pure because nobody marked it as side-effecting.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str = ""
    parameters: dict[str, Any] = Field(default_factory=dict)
    effects: ToolEffect = ToolEffect.UNKNOWN


@dataclass(frozen=True)
class CompileContext:
    """Parameters that shape how a graph is compiled.

    Immutable: a pass that needs to change something here (schema pruning
    narrowing ``tool_catalog``, for instance) returns a new ``CompileContext``
    via :func:`dataclasses.replace` rather than mutating this one.
    """

    budget_tokens: int | None = None
    tool_catalog: dict[str, ToolSchema] = field(default_factory=dict)
    strict: bool = False
    strict_schema: bool = False
    anchor_event_id: str | None = None
    accepted_fold_annotators: frozenset[str] = field(
        default_factory=lambda: frozenset({"runtime", "human"})
    )
    fold_plans: tuple[FoldPlan, ...] = ()


class CompilationReport(BaseModel):
    """What a single pass changed, for use by ``agentslice diff`` and debugging."""

    model_config = ConfigDict(extra="forbid")

    pass_name: str
    events_before: int
    events_after: int
    tokens_before: int
    tokens_after: int
    removed_event_ids: list[str] = Field(default_factory=list)
    modified_event_ids: list[str] = Field(default_factory=list)
    added_event_ids: list[str] = Field(default_factory=list)
    pinned_event_ids: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class CompiledContext(BaseModel):
    """The result of running a :class:`~agentslice.compiler.pipeline.Pipeline`."""

    model_config = ConfigDict(extra="forbid")

    events: list[TraceEvent]
    tool_catalog: dict[str, ToolSchema] = Field(default_factory=dict)
    reports: list[CompilationReport] = Field(default_factory=list)
    budget_tokens: int | None = None
    tokens_before: int
    tokens_after: int
    budget_satisfied: bool | None = None


@dataclass(frozen=True)
class PassOutcome:
    """What a pass produced: the (possibly rebuilt) graph, context, and its report.

    ``ctx`` is threaded through even though most passes leave it unchanged,
    because schema pruning needs to narrow ``tool_catalog`` and there is no
    other channel for a pass to communicate a context change downstream.
    """

    graph: CausalGraph
    ctx: CompileContext
    report: CompilationReport


class Pass(ABC):
    """A single compiler transformation over a causal graph."""

    name: str

    @abstractmethod
    def apply(self, graph: CausalGraph, ctx: CompileContext) -> PassOutcome:
        """Apply this pass, returning the resulting graph, context, and report."""
