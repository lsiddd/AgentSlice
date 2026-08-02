"""The context-construction policies compared by the benchmark runner.

Each policy takes the live event history recorded so far (rebuilt fresh
every turn via ``from_openai_messages`` — same "simple and correct beats
incremental patching" choice the compiler itself makes, see
``agentslice.compiler``) and decides what a model actually gets to see for
its next move. ``full_trace`` is the ceiling (nothing dropped, nothing
projected); the other four are what get compared against it.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from agentslice.compiler.base import Pass, ToolSchema
from agentslice.compiler.pipeline import DEFAULT_PASSES, compile_graph
from agentslice.compiler.tokens import estimate_graph_tokens, estimate_tokens
from agentslice.ir.events import EventType, TraceEvent
from agentslice.ir.graph import build_causal_graph
from agentslice.recording.openai_adapter import to_openai_messages


@dataclass(frozen=True)
class ContextRequest:
    """What a policy decided to send: messages, the tools to offer, and its token cost."""

    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]] | None
    tokens: int


class Summarizer(Protocol):
    """Compresses earlier OpenAI-shaped messages into a short text blurb.

    A real implementation calls out to a (typically cheaper) model; tests
    inject a deterministic stub instead, the same dependency-injection
    pattern ``ReplaySession`` uses for its HTTP transport.
    """

    def __call__(self, messages: list[dict[str, Any]]) -> str: ...


class ContextPolicy(Protocol):
    """Decides what subset/transformation of the trace so far a model sees next."""

    @property
    def name(self) -> str: ...

    def build_request(
        self, events: list[TraceEvent], tool_catalog: dict[str, ToolSchema]
    ) -> ContextRequest: ...


def _tools_payload(tool_catalog: dict[str, ToolSchema]) -> list[dict[str, Any]] | None:
    if not tool_catalog:
        return None
    return [
        {
            "type": "function",
            "function": {
                "name": schema.name,
                "description": schema.description,
                "parameters": schema.parameters,
            },
        }
        for schema in tool_catalog.values()
    ]


def _ordered(events: list[TraceEvent]) -> list[TraceEvent]:
    return sorted(events, key=lambda event: event.seq)


def _split_into_turns(events: list[TraceEvent]) -> list[list[TraceEvent]]:
    """Group non-constraint events by user turn, each starting at a `USER_GOAL`."""
    groups: list[list[TraceEvent]] = []
    for event in _ordered(events):
        if event.type is EventType.CONSTRAINT:
            continue
        if event.type is EventType.USER_GOAL or not groups:
            groups.append([])
        groups[-1].append(event)
    return groups


@dataclass(frozen=True)
class FullTracePolicy:
    """No reduction: every event, verbatim. The baseline the others are measured against."""

    name: str = "full_trace"

    def build_request(
        self, events: list[TraceEvent], tool_catalog: dict[str, ToolSchema]
    ) -> ContextRequest:
        ordered = _ordered(events)
        return ContextRequest(
            messages=to_openai_messages(ordered),
            tools=_tools_payload(tool_catalog),
            tokens=estimate_graph_tokens(ordered),
        )


@dataclass(frozen=True)
class LastNTurnsPolicy:
    """Keeps only the last `n_turns` user turns (plus any constraint/system message).

    Truncates by whole turn, not by message count, so a `tool_call` and its
    `tool_result` are never split across the cut — either both survive or
    neither does.
    """

    n_turns: int = 2
    name: str = "last_n_turns"

    def build_request(
        self, events: list[TraceEvent], tool_catalog: dict[str, ToolSchema]
    ) -> ContextRequest:
        constraints = [event for event in events if event.type is EventType.CONSTRAINT]
        turns = _split_into_turns(events)
        kept_turns = turns[-self.n_turns :] if self.n_turns > 0 else []
        kept = _ordered(constraints + [event for turn in kept_turns for event in turn])
        return ContextRequest(
            messages=to_openai_messages(kept),
            tools=_tools_payload(tool_catalog),
            tokens=estimate_graph_tokens(kept),
        )


_STRUCTURAL_FACT_PREFIXES = ("tool_result:",)


@dataclass(frozen=True)
class RollingStatePolicy:
    """Replaces every turn but the latest with a flat JSON snapshot of known facts.

    Unlike `causal_compile`, this never reasons about what a survivor still
    reads — it always collapses to the same flat shape regardless of causal
    relevance, which is exactly the naive "structured state" baseline the
    causal compiler is meant to improve on.
    """

    name: str = "rolling_state"

    def build_request(
        self, events: list[TraceEvent], tool_catalog: dict[str, ToolSchema]
    ) -> ContextRequest:
        constraints = _ordered([event for event in events if event.type is EventType.CONSTRAINT])
        turns = _split_into_turns(events)
        latest_turn = _ordered(turns[-1]) if turns else []
        earlier_turns = [event for turn in turns[:-1] for event in turn]

        messages = to_openai_messages(constraints)
        if earlier_turns:
            graph = build_causal_graph(events)
            state = {
                key: versions[-1].value
                for key, versions in graph.facts.items()
                if key.startswith(_STRUCTURAL_FACT_PREFIXES)
            }
            messages.append(
                {
                    "role": "assistant",
                    "content": "Known state so far (JSON): "
                    + json.dumps(state, default=str, sort_keys=True),
                }
            )
        messages.extend(to_openai_messages(latest_turn))
        return ContextRequest(
            messages=messages,
            tools=_tools_payload(tool_catalog),
            tokens=estimate_tokens(json.dumps(messages, default=str)),
        )


@dataclass(frozen=True)
class LLMSummaryPolicy:
    """Replaces every turn but the latest with one LLM-generated summary sentence.

    Costs one extra model call per turn on top of the main completion —
    real runs should budget for that, and a cheap/small summarizer model is
    strongly recommended over reusing the model under test.
    """

    summarizer: Summarizer
    name: str = "llm_summary"

    def build_request(
        self, events: list[TraceEvent], tool_catalog: dict[str, ToolSchema]
    ) -> ContextRequest:
        constraints = _ordered([event for event in events if event.type is EventType.CONSTRAINT])
        turns = _split_into_turns(events)
        latest_turn = _ordered(turns[-1]) if turns else []
        earlier_turns = [event for turn in turns[:-1] for event in turn]

        messages = to_openai_messages(constraints)
        if earlier_turns:
            summary = self.summarizer(to_openai_messages(_ordered(earlier_turns)))
            messages.append(
                {"role": "assistant", "content": f"Summary of earlier turns: {summary}"}
            )
        messages.extend(to_openai_messages(latest_turn))
        return ContextRequest(
            messages=messages,
            tools=_tools_payload(tool_catalog),
            tokens=estimate_tokens(json.dumps(messages, default=str)),
        )


@dataclass(frozen=True)
class CausalCompilePolicy:
    """AgentSlice's own default pipeline: causal compilation up to the current point.

    Offers the model the full, unnarrowed `tool_catalog`, not
    `compiled.tool_catalog`. `schema_pruning` narrows the catalog to tools
    *already used* in the survived graph, which is the right question to
    ask about a completed trace but the wrong one here: at the start of a
    turn nothing has been called yet, so the narrowed catalog is routinely
    empty and a model offered zero tools can never call one. Message
    compaction (`compiled.events`/`tokens_after`) is unaffected — only
    which tools the model is told it can call next.

    `passes` defaults to `DEFAULT_PASSES`; overriding it lets the benchmark
    compare experimental pipelines (e.g. a candidate pass not yet promoted
    to the library default) against the shipped one without duplicating
    this class.
    """

    budget_tokens: int | None = None
    passes: Sequence[Pass] | None = None
    name: str = "causal_compile"

    def build_request(
        self, events: list[TraceEvent], tool_catalog: dict[str, ToolSchema]
    ) -> ContextRequest:
        ordered = _ordered(events)
        graph = build_causal_graph(ordered)
        anchor_event_id = ordered[-1].id if ordered else None
        compiled = compile_graph(
            graph,
            budget_tokens=self.budget_tokens,
            tool_catalog=tool_catalog,
            anchor_event_id=anchor_event_id,
            passes=self.passes or DEFAULT_PASSES,
        )
        return ContextRequest(
            messages=to_openai_messages(compiled.events),
            tools=_tools_payload(tool_catalog),
            tokens=compiled.tokens_after,
        )
