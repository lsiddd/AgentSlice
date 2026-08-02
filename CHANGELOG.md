# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
