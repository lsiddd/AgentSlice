"""Deterministic replay and fork: resending a compiled or forked context to a real model."""

from __future__ import annotations

from agentslice.replay.comparator import extract_next_recorded_action, next_action_equivalence
from agentslice.replay.runtime import ReplaySession, replay_compiled_context
from agentslice.replay.tool_stubs import fill_pending_tool_results

__all__ = [
    "ReplaySession",
    "extract_next_recorded_action",
    "fill_pending_tool_results",
    "next_action_equivalence",
    "replay_compiled_context",
]
