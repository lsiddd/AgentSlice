"""Construction of the causal graph that connects a trace's events."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from agentslice.errors import TraceValidationError
from agentslice.ir.events import TraceEvent
from agentslice.ir.facts import Fact


@dataclass(frozen=True)
class CausalEdge:
    """A dependency: ``to_event_id`` reads a fact last written by ``from_event_id``."""

    from_event_id: str
    to_event_id: str
    fact_key: str


@dataclass
class CausalGraph:
    """The causal graph of a trace: events, the edges between them, and fact history.

    Built by :func:`build_causal_graph`. Compiler passes consume and
    transform this structure; it is never persisted directly (only the
    underlying :class:`TraceEvent` list is).
    """

    events: dict[str, TraceEvent]
    order: list[str]
    edges: list[CausalEdge] = field(default_factory=list)
    facts: dict[str, list[Fact]] = field(default_factory=dict)
    unresolved_reads: set[tuple[str, str]] = field(default_factory=set)

    def events_in_order(self) -> list[TraceEvent]:
        """Return events in canonical (``seq``) order."""
        return [self.events[event_id] for event_id in self.order]

    def latest_fact(self, key: str) -> Fact | None:
        """Return the most recent :class:`Fact` written for ``key``, if any."""
        versions = self.facts.get(key)
        if not versions:
            return None
        return versions[-1]

    def ancestors(self, event_id: str) -> set[str]:
        """Return the ids of every event that ``event_id`` causally depends on.

        Walks ``edges`` backward from ``event_id``. Safe against cycles even
        though :func:`build_causal_graph` never produces one: a graph
        assembled by hand (e.g. in a test) could contain one, and this must
        still terminate rather than loop forever.
        """
        incoming: dict[str, list[str]] = {}
        for edge in self.edges:
            incoming.setdefault(edge.to_event_id, []).append(edge.from_event_id)

        visited: set[str] = set()
        stack = list(incoming.get(event_id, []))
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            stack.extend(incoming.get(current, []))
        return visited


def _fact_value(event: TraceEvent) -> object:
    return event.outputs if event.outputs is not None else event.inputs


def build_causal_graph(events: Sequence[TraceEvent]) -> CausalGraph:
    """Assemble a :class:`CausalGraph` from a sequence of events.

    Events are reordered by ``seq`` regardless of input order. Raises
    :class:`~agentslice.errors.TraceValidationError` if two events share a
    ``seq`` value, since that leaves the canonical order ambiguous.
    """
    ordered = sorted(events, key=lambda e: e.seq)

    seen_seq: set[int] = set()
    for event in ordered:
        if event.seq in seen_seq:
            raise TraceValidationError(f"duplicate seq {event.seq} on event {event.id!r}")
        seen_seq.add(event.seq)

    events_by_id = {event.id: event for event in ordered}
    order = [event.id for event in ordered]

    facts: dict[str, list[Fact]] = {}
    edges: list[CausalEdge] = []
    unresolved_reads: set[tuple[str, str]] = set()

    for event in ordered:
        for key in sorted(event.reads):
            versions = facts.get(key)
            if not versions:
                unresolved_reads.add((event.id, key))
                continue
            writer = versions[-1]
            edges.append(
                CausalEdge(from_event_id=writer.origin_event_id, to_event_id=event.id, fact_key=key)
            )

        for key in sorted(event.writes):
            previous = facts.get(key)
            supersedes = previous[-1].origin_event_id if previous else None
            new_fact = Fact(
                key=key,
                value=_fact_value(event),
                origin_event_id=event.id,
                supersedes=supersedes,
            )
            facts.setdefault(key, []).append(new_fact)

    return CausalGraph(
        events=events_by_id,
        order=order,
        edges=edges,
        facts=facts,
        unresolved_reads=unresolved_reads,
    )
