"""Parsing of BFCL's Python-call-syntax ground truth strings.

Each ground-truth entry is a literal call expression, e.g.
``"mv(source='final_report.pdf', destination='temp')"`` or the positional
``"sort('final_report.pdf')"``. Parsed via `ast`, never `eval`: only
`ast.literal_eval`-safe argument values are accepted, matching the shape
BFCL itself generates (strings, numbers, booleans, lists, dicts — never an
arbitrary expression).
"""

from __future__ import annotations

import ast
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from benchmarks.errors import MalformedTaskError


@dataclass(frozen=True)
class ParsedCall:
    name: str
    kwargs: dict[str, Any]


def param_names_from_tool_catalog(tools: list[dict[str, Any]]) -> dict[str, list[str]]:
    """Map each tool's name to its parameter names, in declaration order.

    Needed to resolve a ground-truth call's positional arguments (BFCL
    mixes positional and keyword syntax) to the keyword form a model's
    structured `tool_calls` always uses.
    """
    names: dict[str, list[str]] = {}
    for tool in tools:
        function = tool["function"]
        properties = function.get("parameters", {}).get("properties", {})
        names[function["name"]] = list(properties.keys())
    return names


def parse_call(call_str: str, param_names: Mapping[str, list[str]]) -> ParsedCall:
    """Parse one ground-truth call string into a function name and keyword arguments.

    Raises:
        MalformedTaskError: `call_str` isn't a simple `name(...)` call, uses
            `**kwargs`, or has more positional arguments than `param_names`
            knows the name of.
    """
    try:
        tree = ast.parse(call_str.strip(), mode="eval")
    except SyntaxError as exc:
        raise MalformedTaskError(f"not a valid call expression: {call_str!r}") from exc

    call = tree.body
    if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name):
        raise MalformedTaskError(f"not a simple function call: {call_str!r}")

    name = call.func.id
    kwargs: dict[str, Any] = {}

    if call.args:
        positions = param_names.get(name)
        if positions is None or len(positions) < len(call.args):
            raise MalformedTaskError(
                f"positional argument(s) in {call_str!r} but no known parameter order for {name!r}"
            )
        for index, arg in enumerate(call.args):
            kwargs[positions[index]] = ast.literal_eval(arg)

    for keyword in call.keywords:
        if keyword.arg is None:
            raise MalformedTaskError(f"**kwargs is not supported: {call_str!r}")
        kwargs[keyword.arg] = ast.literal_eval(keyword.value)

    return ParsedCall(name=name, kwargs=kwargs)


def parse_turn(calls: list[str], param_names: Mapping[str, list[str]]) -> list[ParsedCall]:
    return [parse_call(call, param_names) for call in calls]
