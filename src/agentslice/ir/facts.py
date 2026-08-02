"""A single versioned piece of state derived from a trace."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


class Fact(BaseModel):
    """A value written by some event, and what it replaced, if anything.

    ``value`` is whatever the writing event's ``outputs`` (or, if that is
    ``None``, its ``inputs``) held at write time. Facts do not drill into
    individual fields of a compound result: a tool result that writes three
    keys produces three ``Fact`` objects that all point at the same whole
    ``outputs`` dict. Per-field precision is left to callers that inspect
    the originating :class:`~agentslice.ir.events.TraceEvent` directly.
    """

    model_config = ConfigDict(extra="forbid")

    key: str
    value: Any
    origin_event_id: str
    supersedes: str | None = None
