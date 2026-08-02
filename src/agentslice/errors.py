"""Exception hierarchy raised deliberately by agentslice.

Every exception the library raises on purpose inherits from
:class:`AgentSliceError`. Callers that want to catch "anything agentslice
might throw" can catch that single class; callers that want finer control
can catch the specific subclass instead.
"""

from __future__ import annotations


class AgentSliceError(Exception):
    """Base class for all errors raised deliberately by agentslice."""


class TraceError(AgentSliceError):
    """Base class for errors related to reading or validating a trace."""


class TraceValidationError(TraceError):
    """A trace was well-formed JSON but violated a structural invariant.

    Examples: a duplicate ``seq`` across two events, a ``tool_result``
    with no matching ``tool_call``, or an event missing a required field.
    """


class TraceFormatError(TraceError):
    """A line in a trace file was not valid JSON."""


class AdapterError(AgentSliceError):
    """Base class for errors raised while converting external data to the IR."""


class UnsupportedMessageFormatError(AdapterError):
    """An input message did not match the format an adapter expects."""


class CompilerError(AgentSliceError):
    """Base class for errors raised while compiling a causal graph."""


class UnknownToolError(CompilerError):
    """A tool was invoked that has no entry in the supplied tool catalog.

    Only raised when the pipeline runs with ``strict_schema=True``.
    """


class BudgetNotSatisfiableError(CompilerError):
    """The token budget was not met after running every configured pass.

    Only raised when the pipeline runs with ``strict=True``.
    """


class CLIUsageError(AgentSliceError):
    """The CLI was invoked in a way that isn't a well-formed usage error.

    Reserved for usage problems that Typer's own argument parsing does not
    already cover (e.g. a value that parses but is semantically invalid).
    """


class ReplayError(AgentSliceError):
    """Base class for errors raised while replaying or forking a trace."""


class MissingToolResultError(ReplayError):
    """A ``tool_call`` has no recorded ``tool_result`` to substitute during replay.

    Replay never executes a tool for real, so if neither the context being
    replayed nor the original trace it was derived from has an answer for
    some call, there is nothing else it can do.
    """


class UnknownAnchorError(ReplayError):
    """A fork's ``--at`` event id has no matching event in the given trace."""
