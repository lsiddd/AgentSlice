# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
