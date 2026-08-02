"""Exception hierarchy for the benchmark harness.

Kept separate from :mod:`agentslice.errors`: this package lives outside the
installable library (see ``NEXT_STEPS.md``), so its failure modes — a
malformed fixture, an unknown simulated tool — aren't part of AgentSlice's
own public contract.
"""

from __future__ import annotations


class BenchmarkError(Exception):
    """Base class for all errors raised deliberately by the benchmark harness."""


class UnknownFunctionError(BenchmarkError):
    """A model called a function name the target environment doesn't implement."""


class InvalidCallError(BenchmarkError):
    """A model's call to a known function had missing or malformed arguments."""


class MalformedTaskError(BenchmarkError):
    """A BFCL task fixture is missing a field the loader requires."""
