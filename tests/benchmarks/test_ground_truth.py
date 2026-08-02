import pytest

from benchmarks.errors import MalformedTaskError
from benchmarks.ground_truth import (
    ParsedCall,
    param_names_from_tool_catalog,
    parse_call,
    parse_turn,
)

_PARAM_NAMES = {"cd": ["folder"], "mv": ["source", "destination"], "sort": ["file_name"]}


def test_parse_call_with_keyword_arguments() -> None:
    assert parse_call("cd(folder='document')", _PARAM_NAMES) == ParsedCall(
        name="cd", kwargs={"folder": "document"}
    )


def test_parse_call_with_positional_argument() -> None:
    assert parse_call("sort('final_report.pdf')", _PARAM_NAMES) == ParsedCall(
        name="sort", kwargs={"file_name": "final_report.pdf"}
    )


def test_parse_call_with_multiple_keyword_arguments() -> None:
    parsed = parse_call("mv(source='a.txt', destination='temp')", _PARAM_NAMES)
    assert parsed.kwargs == {"source": "a.txt", "destination": "temp"}


def test_parse_call_rejects_non_call_expression() -> None:
    with pytest.raises(MalformedTaskError):
        parse_call("1 + 1", _PARAM_NAMES)


def test_parse_call_rejects_double_star_kwargs() -> None:
    with pytest.raises(MalformedTaskError):
        parse_call("cd(**{'folder': 'x'})", _PARAM_NAMES)


def test_parse_call_rejects_unknown_positional_arity() -> None:
    with pytest.raises(MalformedTaskError):
        parse_call("touch('a.txt')", _PARAM_NAMES)


def test_parse_call_rejects_syntax_error() -> None:
    with pytest.raises(MalformedTaskError):
        parse_call("cd(folder=", _PARAM_NAMES)


def test_parse_turn_parses_every_call_in_order() -> None:
    calls = parse_turn(["cd(folder='a')", "cd(folder='b')"], _PARAM_NAMES)
    assert [c.kwargs["folder"] for c in calls] == ["a", "b"]


def test_param_names_from_tool_catalog() -> None:
    tools = [
        {
            "type": "function",
            "function": {
                "name": "cd",
                "parameters": {"type": "object", "properties": {"folder": {"type": "string"}}},
            },
        }
    ]
    assert param_names_from_tool_catalog(tools) == {"cd": ["folder"]}
