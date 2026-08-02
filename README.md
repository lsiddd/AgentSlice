# AgentSlice

[![CI](https://github.com/lsiddd/AgentSlice/actions/workflows/ci.yml/badge.svg)](https://github.com/lsiddd/AgentSlice/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](pyproject.toml)

A causal context compiler for tool-using agents.

> `audit history ≠ executable state ≠ model context`

The execution history of a tool-using agent is not a conversation, it is an
execution trace: goals, tool calls, tool results, constraints, and state
changes, each depending on some of what came before and irrelevant to the
rest. AgentSlice treats that trace as a causal graph and compiles it down to
the smallest subset of events that still preserves the state an agent needs
to keep going correctly — then lets you replay that compiled context against
a real model to check whether it actually still behaves the same way.

## What this is not

- **Not a summarizer.** Nothing is rewritten or paraphrased. Events are kept
  verbatim, projected down to the fields something still reads, or dropped
  because nothing causally depends on them.
- **Not RAG.** There is no embedding index or similarity search. Relevance is
  derived from explicit `reads`/`writes` dependencies between events, not from
  vector distance.
- **Not a workflow graph.** AgentSlice doesn't orchestrate an agent or decide
  what it does next; it only compiles the history a *next step* would be
  given, and can check that compilation against reality.

## How it works

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

```
constraint_pinning → dead_events → superseded_state → duplicate_result_elimination → tool_result_projection → schema_pruning
```

1. **`constraint_pinning`** marks every `constraint` event as pinned so later
   passes can't drop it for lacking causal edges.
2. **`dead_events`** removes events with no causal path to the current state
   (the "anchor" event) and that aren't pinned at or before it.
3. **`superseded_state`** collapses events whose written facts have all been
   overwritten since and that no surviving event still reads: dropped
   outright if they had no side effects, kept but redacted if they did.
4. **`duplicate_result_elimination`** collapses a later `tool_call`/
   `tool_result` pair that exactly repeats an earlier one, same
   remove-vs-redact split as `superseded_state`.
5. **`tool_result_projection`** shrinks a tool result down to only the
   fields something still reads.
6. **`schema_pruning`** narrows a supplied tool catalog down to the tools
   actually used.

## Replay and fork

Compiling a smaller context is only useful if the model still does the right
thing with it. `agentslice replay` and `agentslice fork` close that loop by
sending a compiled context to a real, OpenAI-compatible model and comparing
its next action — tool call or final message — against what the agent
actually did next in the original trace:

```bash
agentslice fork session.jsonl --at call_17 --model gpt-4o-mini \
  --tools catalog.json --api-key-env OPENAI_API_KEY
```

`fork` compiles the causal context up to event `call_17` from scratch and
sends it straight to the model. `replay` does the same against an
already-compiled trace, so you can inspect or version the context first:

```bash
agentslice compile session.jsonl --budget 4096 -o compiled.jsonl
agentslice replay compiled.jsonl --compare-with session.jsonl --model gpt-4o-mini
```

Either way, the report is `EQUIVALENT` or `DIFFERENT` — exact tool-name and
argument comparison for now, not semantic similarity (see
[CHANGELOG](CHANGELOG.md) for the precise rules). Any `tool_call` left
without a matching `tool_result` in the compiled context is answered from
the original trace rather than executed for real, so replay never has side
effects. Both commands default to OpenRouter's endpoint but work against any
OpenAI-compatible API.

## Installation

Not published to PyPI yet — install straight from the repository:

```bash
git clone https://github.com/lsiddd/AgentSlice.git
cd AgentSlice
uv sync
uv run agentslice --help
```

Or as a tool, if you just want the CLI on your `PATH`:

```bash
uv tool install git+https://github.com/lsiddd/AgentSlice.git
```

## CLI quickstart

Given a JSON file with an array of OpenAI-compatible chat messages:

```bash
agentslice record --input messages.json --output trace.jsonl
agentslice compile trace.jsonl --budget 2048 -o compiled.jsonl
agentslice diff trace.jsonl compiled.jsonl --report compiled.jsonl.report.json
```

- **`record`** converts a message log into a trace. `--format openai`
  (default) expects a JSON array; `--format claude-code` and `--format
  codex` read a Claude Code session transcript or a Codex CLI rollout
  directly as JSON Lines — no manual reformatting needed.
- **`compile`** runs the default pipeline against a trace, optionally
  enforcing a token budget (`--budget`, `--strict` to fail instead of
  under-delivering) or a tool catalog (`--tools catalog.json
  --strict-schema` to fail on an undeclared tool).
- **`diff`** shows what survived, what was dropped, and by which pass
  (`--format table` or `json`).
- **`replay`** / **`fork`** — see [Replay and fork](#replay-and-fork) above.

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

Replay a compiled context and compare it to what actually happened next:

```python
from agentslice.replay import (
    ReplaySession,
    replay_compiled_context,
    next_action_equivalence,
    extract_next_recorded_action,
)

with ReplaySession(base_url, api_key, model="gpt-4o-mini") as session:
    replayed = replay_compiled_context(compiled.events, original_events=events, session=session)

original = extract_next_recorded_action(events, anchor_id=compiled.events[-1].id)
print(next_action_equivalence(original, replayed) if original else "nothing to compare")
```

## Status

Core IR, the five-pass compiler, and deterministic replay/fork are
implemented and tested against real recorded sessions (Claude Code, Codex
CLI, and raw OpenAI/OpenRouter chat completions), not just synthetic
fixtures. See [CHANGELOG.md](CHANGELOG.md) for the full history, including a
recent audit that closed several causal-completeness gaps (superseded state
with live readers, future pinned events leaking into a fork, multi-turn
conversational context) uncovered by real-world usage.

### Roadmap

- One more compiler pass: failed-hypothesis folding.
- A benchmark harness against a subset of the Berkeley Function Calling
  Leaderboard (BFCL).
- Export of compiled/uncompiled pairs as a DPO preference dataset.
- Adapters for MCP and LangGraph.

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
