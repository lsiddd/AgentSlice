"""The internal task schema loaded from the vendored BFCL fixtures."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


class BFCLTask(BaseModel):
    """One multi-turn BFCL task: a scripted conversation plus its ground truth.

    ``turns`` and ``ground_truth`` are parallel: ``ground_truth[i]`` is the
    list of function-call strings (e.g. ``"cd(folder='x')"``) a correct
    agent issues in response to ``turns[i]``, the user message(s) for that
    turn. ``initial_config`` seeds the simulated environment before turn 0;
    its top-level keys name the environment classes involved, matching
    ``involved_classes``.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    turns: list[list[dict[str, Any]]]
    initial_config: dict[str, Any]
    involved_classes: list[str]
    ground_truth: list[list[str]]
