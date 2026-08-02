"""A dependency-free, approximate token count used for budget accounting.

This is a heuristic (``len(text) // 4``), not a real tokenizer. It is
deliberately isolated in this module so a provider-accurate backend (e.g.
one built on ``tiktoken``) can be swapped in later without touching any
compiler pass: they only ever call :func:`estimate_event_tokens` or
:func:`estimate_graph_tokens`.
"""

from __future__ import annotations

import json
from collections.abc import Iterable

from agentslice.ir.events import TraceEvent


def estimate_tokens(text: str) -> int:
    """Approximate the token count of ``text``."""
    if not text:
        return 0
    return max(1, len(text) // 4)


def estimate_event_tokens(event: TraceEvent) -> int:
    """Approximate the token cost of the parts of ``event`` a model would see.

    Only ``type``, ``tool_name``, ``inputs``, and ``outputs`` count: the
    rest of the fields (``id``, ``seq``, ``reads``, ``writes``, ``pinned``,
    ``metadata``, ``timestamp``) are bookkeeping for the compiler itself,
    not content that would be serialized into a model's context.
    """
    payload = {
        "type": event.type.value,
        "tool_name": event.tool_name,
        "inputs": event.inputs,
        "outputs": event.outputs,
    }
    return estimate_tokens(json.dumps(payload, default=str))


def estimate_graph_tokens(events: Iterable[TraceEvent]) -> int:
    """Approximate the total token cost of a sequence of events."""
    return sum(estimate_event_tokens(event) for event in events)
