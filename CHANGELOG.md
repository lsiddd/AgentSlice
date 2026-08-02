# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `compiler.duplicate_result_elimination.DuplicateResultEliminationPass`,
  the sixth compiler pass: collapses a later `tool_call`/`tool_result`
  pair that exactly repeats an earlier one (same `tool_name` and `inputs`,
  identical `outputs`). Dropped outright when `side_effects=False`, kept
  but redacted when `True`, mirroring `superseded_state`'s remove-vs-redact
  split. Runs in `DEFAULT_PASSES` right after `superseded_state`. A
  candidate is left untouched if the anchor or a pinned event is either
  half of the pair, or if some other surviving event still has a causal
  edge reading from the call or the result *and* doesn't also depend on
  the retained baseline pair — a reader covered redundantly by both
  (routine, since value-matching links a later call to every prior fact
  sharing that value, not just the newest) doesn't block elimination.

## [0.2.0] - 2026-08-02

### Added

- `to_openai_messages`, the inverse of `from_openai_messages`: converts a
  `TraceEvent` sequence back into OpenAI-compatible chat messages.
- A `recording.openai_adapter` fix closing a causal gap: `user_goal` now
  writes a versioned `user_goal:current` fact read by every `tool_call`
  and `model_message`, and every `tool_result` reads the `tool_call:{id}`
  fact its own `tool_call` wrote. Previously neither the user's goal nor
  the arguments of a tool call had any causal edge pointing to them, so
  `dead_events` routinely dropped both from a compiled context.
- `agentslice.replay`: deterministic replay and fork of a recorded
  execution. `ReplaySession`/`replay_compiled_context` resend a compiled
  or forked context to a real model and capture its next action;
  `fill_pending_tool_results` answers any pending `tool_call` from the
  original trace rather than executing a tool for real;
  `next_action_equivalence`/`extract_next_recorded_action` compare that
  action, exactly (tool name + JSON-normalized arguments, or presence of
  a final text answer), against what the original trace actually did
  next.
- CLI: `agentslice replay` and `agentslice fork` commands.
- `recording.claude_code_adapter.from_claude_code_transcript` and
  `recording.codex_adapter.from_codex_rollout`: adapters for Claude Code
  session transcripts and Codex CLI rollouts, both reshaping into
  OpenAI-compatible messages and delegating to `from_openai_messages`
  rather than re-implementing causal inference. `agentslice record --format
  claude-code|codex` reads the corresponding JSON Lines log directly.
- `ReplaySession`/`LiveSession` now accept a `timeout` (default 120s,
  `httpx`'s own default of 5s was too short for a reasoning model).

### Fixed

- `ir.graph.build_causal_graph` now rejects a duplicate event `id` (only
  duplicate `seq` was checked before), which previously let one event
  silently shadow another in the graph without raising. The OpenAI
  adapter also detects the specific case of a `tool_call`'s externally
  supplied id colliding with an internally generated one.
- `compiler.superseded_state`: a superseded event is no longer
  removed/redacted if a surviving event still has a causal edge reading
  its historical value, which previously left that reader's `reads`
  unresolved.
- `compiler.dead_events`: a pinned event is now only kept when its `seq`
  is at or before the anchor's — a pinned event recorded after the
  anchor (e.g. a later constraint) no longer leaks into a forked context.
- `compiler.schema_pruning`: `strict_schema=True` now raises
  `UnknownToolError` when the tool catalog is empty but tools were used,
  instead of silently no-opping.
- `compiler.tool_result_projection`: the per-field write key mapping is
  matched by exact suffix instead of splitting on the last dot, so a
  field name containing a dot is no longer truncated, and an opaque
  result whose call id happens to contain a dot is no longer mistaken
  for a field-projectable one. Write keys unrelated to a specific output
  field now survive projection instead of being dropped.
- `recording.openai_adapter`: every event now also writes a versioned
  `conversation:current` fact, read by each `user` turn after the first,
  so forking mid-conversation no longer strands a follow-up turn with no
  causal path back to earlier context. A nested `tool_result` now indexes
  its individual leaf values so a later `tool_call` referencing one is
  still linked by value match. A `tool_result` whose content wasn't a
  JSON object preserves its original text verbatim across a
  `to_openai_messages` round trip instead of re-wrapping it as JSON.
- `replay.comparator.next_action_equivalence` now compares tool calls by
  multiset instead of set, so a repeated identical call is no longer
  conflated with a single call.

## [0.1.0] - 2026-08-02

### Added

- Typed intermediate representation: `TraceEvent`, `EventType`, `Fact`, and
  `CausalGraph`/`build_causal_graph`.
- JSON Lines trace persistence via `TraceWriter`/`TraceReader`.
- An adapter converting OpenAI-compatible chat messages (also covering
  OpenRouter) into `TraceEvent` sequences, inferring `reads`/`writes` by
  value matching against known facts.
- `LiveSession`, a synchronous, non-streaming recorder for one HTTP chat
  completion turn at a time.
- A compiler with five passes — constraint pinning, dead event elimination,
  superseded state collapsing, tool result projection, and schema pruning —
  orchestrated by a configurable `Pipeline`.
- A dependency-free token estimation heuristic used for budget accounting.
- The `agentslice` CLI: `record`, `compile`, and `diff` commands.

### Roadmap (not in this release)

- Deterministic replay and fork of a recorded execution.
- A next-action comparator for measuring behavioral equivalence.
- A benchmark harness against a subset of the Berkeley Function Calling
  Leaderboard (BFCL).
- Export of compiled/uncompiled pairs as a DPO preference dataset.
- Adapters for MCP, LangGraph, and the OpenAI Agents SDK.
- Failed-hypothesis folding and duplicate-result elimination passes.
