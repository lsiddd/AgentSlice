# AgentSlice

A causal context compiler for tool-using agents.

> `audit history ≠ executable state ≠ model context`

The execution history of a tool-using agent is not a conversation, it is an
execution trace: goals, tool calls, tool results, constraints, and state
changes, each depending on some of what came before and irrelevant to the
rest. AgentSlice treats that trace as a causal graph and compiles it down to
the smallest subset of events that still preserves the state an agent needs
to keep going correctly.

## What this is not

- **Not a summarizer.** Nothing is rewritten or paraphrased. Events are kept
  verbatim, projected down to the fields something still reads, or dropped
  because nothing causally depends on them.
- **Not RAG.** There is no embedding index or similarity search. Relevance is
  derived from explicit `reads`/`writes` dependencies between events, not from
  vector distance.
- **Not a workflow graph.** AgentSlice doesn't orchestrate an agent or decide
  what it does next; it only compiles the history a *next step* would be
  given.

## Core concepts

- **`TraceEvent`** — one step in a trace: a user goal, a model message, a
  tool call, a tool result, a constraint, a state update, an error, or a
  final output. Carries `reads`/`writes` (the fact keys it depends on or
  produces), `side_effects`, and `pinned`.
- **`Fact`** — a versioned value written by some event, with a pointer to
  whatever version it superseded.
- **`CausalGraph`** — a trace's events connected by fact dependencies, built
  by `build_causal_graph()`.
- **Compiler passes** — pure functions from one `CausalGraph` to a smaller
  one, run in sequence by a `Pipeline`.

### The default pipeline

```
constraint_pinning → dead_events → superseded_state → tool_result_projection → schema_pruning
```

1. **`constraint_pinning`** marks every `constraint` event as pinned so later
   passes can't drop it for lacking causal edges.
2. **`dead_events`** removes events with no causal path to the current state
   (the "anchor" event) and that aren't pinned.
3. **`superseded_state`** collapses events whose written facts have all been
   overwritten since: dropped outright if they had no side effects, kept but
   redacted if they did.
4. **`tool_result_projection`** shrinks a tool result down to only the
   fields something still reads.
5. **`schema_pruning`** narrows a supplied tool catalog down to the tools
   actually used.

## Installation

```bash
uv add agentslice
```

Or, for local development:

```bash
git clone https://github.com/lsiddd/AgentSlice.git
cd AgentSlice
uv sync
```

## CLI quickstart

Given a JSON file with an array of OpenAI-compatible chat messages:

```bash
agentslice record --input messages.json --output trace.jsonl
agentslice compile trace.jsonl --budget 2048 -o compiled.jsonl
agentslice diff trace.jsonl compiled.jsonl --report compiled.jsonl.report.json
```

`record` converts a message log into a trace. `compile` runs the default
pipeline against it, optionally enforcing a token budget (`--strict`) or a
tool catalog (`--tools catalog.json --strict-schema`). `diff` shows what
survived, what was dropped, and by which pass.

Run `agentslice --help`, or `agentslice <command> --help`, for the full set
of options.

## Library usage

```python
from agentslice.ir import build_causal_graph
from agentslice.recording import from_openai_messages
from agentslice.compiler import compile_graph

events = from_openai_messages(messages)
graph = build_causal_graph(events)
compiled = compile_graph(graph, budget_tokens=2048)

print(compiled.tokens_before, "->", compiled.tokens_after)
for event in compiled.events:
    ...
```

## v0.1.0 scope

This release is the foundational core: the intermediate representation, a
JSONL trace format, an OpenAI-compatible message adapter (which also covers
OpenRouter, since it mirrors the same API shape), a live HTTP recorder for
one completion turn at a time, the five compiler passes above, and the
`record` / `compile` / `diff` CLI commands.

### Roadmap (not in v0.1.0)

- Deterministic replay and fork of a recorded execution.
- A next-action comparator for measuring behavioral equivalence after
  compilation.
- A benchmark harness against a subset of the Berkeley Function Calling
  Leaderboard (BFCL).
- Export of compiled/uncompiled pairs as a DPO preference dataset.
- Adapters for MCP, LangGraph, and the OpenAI Agents SDK.
- Two additional compiler passes: failed-hypothesis folding and
  duplicate-result elimination.

## Development

```bash
uv sync
uv run ruff check .
uv run ruff format --check .
uv run mypy src
uv run pytest --cov=agentslice
```

## Contributing

Commit messages follow [Conventional Commits](https://www.conventionalcommits.org/).
Lint, type checking, and tests must pass before a change is merged.

## License

[MIT](LICENSE)
