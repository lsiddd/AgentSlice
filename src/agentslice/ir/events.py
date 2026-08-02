"""The event types that make up a trace."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class EventType(StrEnum):
    """The kind of thing a :class:`TraceEvent` represents."""

    USER_GOAL = "user_goal"
    MODEL_MESSAGE = "model_message"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    STATE_UPDATE = "state_update"
    CONSTRAINT = "constraint"
    ERROR = "error"
    FINAL_OUTPUT = "final_output"


class TraceEvent(BaseModel):
    """A single step in an agent's execution history.

    ``seq`` is the canonical ordering of events within a trace and takes
    priority over ``timestamp``, which is informational only (clocks can be
    skewed, missing, or absent entirely for hand-built traces).

    ``reads`` and ``writes`` name the fact keys this event depends on and
    produces, respectively. They are what :func:`agentslice.ir.graph.build_causal_graph`
    uses to connect events into a causal graph.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    seq: int
    type: EventType
    timestamp: datetime | None = None
    tool_name: str | None = None
    inputs: dict[str, Any] | None = None
    outputs: dict[str, Any] | None = None
    reads: frozenset[str] = Field(default_factory=frozenset)
    writes: frozenset[str] = Field(default_factory=frozenset)
    side_effects: bool = False
    pinned: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)
