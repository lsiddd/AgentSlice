# AgentSlice benchmark harness

Compares context-construction policies — including AgentSlice's own causal
compiler — on their ability to preserve an agent's next correct action
while shrinking what a model actually sees. This is Marco 10 from the
project roadmap. It is deliberately **not** part of the installable
`agentslice` package: it depends on a vendored dataset subset and, when
actually run, on a paid model API.

## What it measures

A live, turn-by-turn run of a multi-turn task from the [Berkeley Function
Calling Leaderboard](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard)
(BFCL) against a real model, under one of five context policies:

- `full_trace` — no reduction, the ceiling every other policy is measured against.
- `last_n_turns` — keeps only the most recent user turn(s).
- `rolling_state` — collapses every earlier turn into a flat JSON snapshot of known facts.
- `llm_summary` — collapses every earlier turn into one LLM-generated summary sentence.
- `causal_compile` — AgentSlice's own default compiler pipeline.

Each tool call the model makes is executed against a simulated backend
(see "Scope" below), and the final state is compared against the state a
correct sequence of calls (the dataset's ground truth) would have
produced. See `benchmarks/metrics.py` for the exact definition of every
reported metric — a few (constraint retention, argument equivalence)
needed a precise definition beyond what `NEXT_STEPS.md` originally stated.

## Scope

Only the `GorillaFileSystem` environment (a simulated Unix-like file
system) and the `multi_turn_base` category are covered, as the 13 tasks in
that category whose `involved_classes` is exactly `["GorillaFileSystem"]`
— see `bfcl/fixtures/NOTICE.md` for exactly where that data comes from and
why the scope stops there. BFCL's other ten environment classes
(`TwitterAPI`, `TicketAPI`, `TradingBot`, etc.) and its other multi-turn
categories (`multi_turn_long_context`, `multi_turn_miss_func`,
`multi_turn_miss_param`) are not implemented. Expanding coverage means
porting another environment class the same way `environments/gorilla_file_system.py`
was: faithfully, against the reference implementation, since the ground
truth was generated against that exact behavior.

## Running the tests

Fully offline, no API key or network access needed — the dataset subset
and tool catalog are bundled, and every model call in the test suite goes
through `httpx.MockTransport`:

```bash
uv run pytest tests/benchmarks
```

## Running the benchmark for real

This spends real API budget. Nothing runs automatically; every request is
cached under `--cache-dir` by exact payload hash, so re-running the same
(policy, model, task) triple after an interruption costs nothing more.

```bash
uv run python -m benchmarks.cli list-tasks
uv run python -m benchmarks.cli run \
  --model openrouter/some-model \
  --policy full_trace --policy causal_compile \
  --api-key-env OPENROUTER_API_KEY \
  --limit 3 \
  --output results.json
```

There is no built-in spending cap beyond `--limit` (there are 13 bundled
tasks total) and the number of `--policy` values requested — choose both
deliberately. `--policy llm_summary` issues one extra summarization call
per turn on top of the main completion; that extra cost isn't reflected in
`--prompt-price-per-1k`/`--completion-price-per-1k`, which only estimate
the cost of the main model calls the runner itself tracks. `--check-determinism`
re-issues each task's final turn twice more, uncached, purely to measure
agreement — also not free.

Costs reported anywhere in this harness (`estimated_cost_usd`) are computed
from AgentSlice's own token *estimate* (`agentslice.compiler.tokens`,
`len(text) // 4`), the same heuristic the compiler itself uses for budget
accounting — good for comparing policies against each other, not for
reconciling against an actual invoice.
