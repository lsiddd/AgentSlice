"""Benchmark harness comparing context policies against a subset of the BFCL.

Deliberately outside ``src/agentslice``: it depends on vendored dataset
fixtures and, when actually run, on a paid model API, neither of which
belongs in the installable library (see ``NEXT_STEPS.md``, Marco 10).
"""

from __future__ import annotations
