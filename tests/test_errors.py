import pytest

from agentslice.errors import (
    AdapterError,
    AgentSliceError,
    BudgetNotSatisfiableError,
    CLIUsageError,
    CompilerError,
    TraceError,
    TraceFormatError,
    TraceValidationError,
    UnknownToolError,
    UnsupportedMessageFormatError,
)


@pytest.mark.parametrize(
    ("exc_type", "expected_bases"),
    [
        (TraceError, (AgentSliceError,)),
        (TraceValidationError, (TraceError, AgentSliceError)),
        (TraceFormatError, (TraceError, AgentSliceError)),
        (AdapterError, (AgentSliceError,)),
        (UnsupportedMessageFormatError, (AdapterError, AgentSliceError)),
        (CompilerError, (AgentSliceError,)),
        (UnknownToolError, (CompilerError, AgentSliceError)),
        (BudgetNotSatisfiableError, (CompilerError, AgentSliceError)),
        (CLIUsageError, (AgentSliceError,)),
    ],
)
def test_hierarchy(exc_type: type[Exception], expected_bases: tuple[type[Exception], ...]) -> None:
    for base in expected_bases:
        assert issubclass(exc_type, base)


def test_all_deliberate_errors_are_catchable_as_agentslice_error() -> None:
    with pytest.raises(AgentSliceError):
        raise TraceValidationError("duplicate seq: 3")


def test_siblings_are_not_related() -> None:
    assert not issubclass(TraceError, CompilerError)
    assert not issubclass(CompilerError, TraceError)
    assert not issubclass(AdapterError, TraceError)


def test_messages_are_preserved() -> None:
    err = TraceFormatError("trace.jsonl:12: invalid JSON: Expecting value")
    assert str(err) == "trace.jsonl:12: invalid JSON: Expecting value"
