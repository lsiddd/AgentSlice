"""Typed payloads for synthetic epistemic state updates."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class EpistemicEvidence(BaseModel):
    """One machine-verifiable observation retained by a hypothesis fold."""

    model_config = ConfigDict(extra="forbid", strict=True)

    event_id: str = Field(min_length=1)
    json_pointer: str
    operator: Literal["=="]
    value: Any


class RuledOutHypothesis(BaseModel):
    """One explicitly rejected hypothesis and the evidence that rejected it."""

    model_config = ConfigDict(extra="forbid", strict=True)

    fold_id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    hypothesis: str = Field(min_length=1)
    evidence: list[EpistemicEvidence] = Field(min_length=1)


class EpistemicStateV1(BaseModel):
    """The only synthetic ``STATE_UPDATE`` subtype currently replayable."""

    model_config = ConfigDict(extra="forbid", strict=True)

    kind: Literal["epistemic_state"]
    schema_version: Literal[1]
    ruled_out: list[RuledOutHypothesis] = Field(min_length=1)
