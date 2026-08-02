import json
from pathlib import Path
from typing import Any

import httpx

from agentslice.compiler.base import ToolEffect, ToolSchema
from agentslice.replay.runtime import ReplaySession
from benchmarks.bfcl.schema import BFCLTask
from benchmarks.cache import ResponseCache
from benchmarks.environments.gorilla_file_system import create as create_gorilla_file_system
from benchmarks.policies import FullTracePolicy
from benchmarks.runner import BenchmarkRunner, RunnerConfig

_TOOLS_PAYLOAD = [
    {
        "type": "function",
        "function": {
            "name": "mkdir",
            "description": "make a directory",
            "parameters": {
                "type": "object",
                "properties": {"dir_name": {"type": "string"}},
                "required": ["dir_name"],
            },
        },
    }
]
_TOOL_CATALOG = {"mkdir": ToolSchema(name="mkdir", description="make a directory")}

_TASK = BFCLTask(
    id="t1",
    turns=[[{"role": "user", "content": "make a directory called notes"}]],
    initial_config={"GorillaFileSystem": {"root": {"home": {"type": "directory", "contents": {}}}}},
    involved_classes=["GorillaFileSystem"],
    ground_truth=[["mkdir(dir_name='notes')"]],
)


def _mkdir_tool_call(dir_name: str = "notes", call_id: str | None = "call_0") -> dict[str, Any]:
    tool_call: dict[str, Any] = {
        "type": "function",
        "function": {"name": "mkdir", "arguments": json.dumps({"dir_name": dir_name})},
    }
    if call_id is not None:
        tool_call["id"] = call_id
    return {"role": "assistant", "content": None, "tool_calls": [tool_call]}


def _final(content: str = "Done.") -> dict[str, Any]:
    return {"role": "assistant", "content": content}


class _ScriptedHandler:
    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self._responses = responses
        self.call_count = 0

    def __call__(self, request: httpx.Request) -> httpx.Response:
        index = min(self.call_count, len(self._responses) - 1)
        self.call_count += 1
        return httpx.Response(200, json={"choices": [{"message": self._responses[index]}]})


def _runner(
    handler: _ScriptedHandler,
    *,
    cache: ResponseCache | None = None,
    config: RunnerConfig | None = None,
) -> tuple[BenchmarkRunner, ReplaySession]:
    session = ReplaySession(
        "https://api.example.com/v1", "sk-test", model="m", transport=httpx.MockTransport(handler)
    )
    runner = BenchmarkRunner(
        session,
        "m",
        FullTracePolicy(),
        _TOOL_CATALOG,
        _TOOLS_PAYLOAD,
        create_gorilla_file_system,
        cache=cache,
        config=config,
    )
    return runner, session


def test_run_task_matches_ground_truth_and_succeeds() -> None:
    handler = _ScriptedHandler([_mkdir_tool_call(), _final()])
    runner, session = _runner(handler)
    outcome = runner.run_task(_TASK)
    session.close()

    assert outcome.end_to_end_success is True
    assert outcome.turns[0].next_action_equivalent is True
    assert outcome.turns[0].invalid_call_count == 0


def test_run_task_reports_failure_when_model_does_something_else() -> None:
    handler = _ScriptedHandler([_mkdir_tool_call(dir_name="wrong_name"), _final()])
    runner, session = _runner(handler)
    outcome = runner.run_task(_TASK)
    session.close()

    assert outcome.end_to_end_success is False
    assert outcome.turns[0].next_action_equivalent is False


def test_run_task_counts_invalid_calls_without_crashing() -> None:
    bogus_call = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_0",
                "type": "function",
                "function": {"name": "delete_all", "arguments": "{}"},
            }
        ],
    }
    handler = _ScriptedHandler([bogus_call, _final()])
    runner, session = _runner(handler)
    outcome = runner.run_task(_TASK)
    session.close()

    assert outcome.turns[0].invalid_call_count == 1
    assert outcome.end_to_end_success is False


def test_missing_tool_call_id_gets_a_synthetic_one_instead_of_crashing() -> None:
    handler = _ScriptedHandler([_mkdir_tool_call(call_id=None), _final()])
    runner, session = _runner(handler)
    outcome = runner.run_task(_TASK)
    session.close()

    assert outcome.end_to_end_success is True


def test_cache_avoids_a_second_network_round_trip(tmp_path: Path) -> None:
    handler = _ScriptedHandler([_mkdir_tool_call(), _final()])
    cache = ResponseCache(tmp_path / "cache")
    runner, session = _runner(handler, cache=cache)

    runner.run_task(_TASK)
    calls_after_first_run = handler.call_count
    runner.run_task(_TASK)
    session.close()

    assert handler.call_count == calls_after_first_run


def test_check_determinism_reports_full_agreement_when_response_is_stable() -> None:
    handler = _ScriptedHandler([_mkdir_tool_call(), _final()])
    runner, session = _runner(handler, config=RunnerConfig(check_determinism=True))
    outcome = runner.run_task(_TASK)
    session.close()

    assert outcome.determinism_rate == 1.0
    assert handler.call_count == 4


def test_check_determinism_reports_partial_agreement_when_response_flips() -> None:
    handler = _ScriptedHandler([_mkdir_tool_call(), _final(), _final(), _mkdir_tool_call()])
    runner, session = _runner(handler, config=RunnerConfig(check_determinism=True))
    outcome = runner.run_task(_TASK)
    session.close()

    assert outcome.determinism_rate == 0.5


def test_determinism_is_none_when_not_requested() -> None:
    handler = _ScriptedHandler([_mkdir_tool_call(), _final()])
    runner, session = _runner(handler)
    outcome = runner.run_task(_TASK)
    session.close()

    assert outcome.determinism_rate is None


def test_positional_ground_truth_is_resolved_via_tools_payload() -> None:
    task = BFCLTask(
        id="t2",
        turns=_TASK.turns,
        initial_config=_TASK.initial_config,
        involved_classes=_TASK.involved_classes,
        ground_truth=[["mkdir('notes')"]],
    )
    handler = _ScriptedHandler([_mkdir_tool_call(), _final()])
    runner, session = _runner(handler)
    outcome = runner.run_task(task)
    session.close()

    assert outcome.end_to_end_success is True
    assert outcome.turns[0].ground_truth_calls[0].kwargs == {"dir_name": "notes"}


def test_side_effect_tools_are_derived_from_the_catalogs_tool_effect() -> None:
    """`BenchmarkRunner` must classify events by the real per-tool effect, not a blanket default.

    `from_openai_messages` only marks a `tool_result` as `side_effects=True`
    for tools named in `side_effect_tools`; without this derivation every
    GFS mutation (`mkdir`, `rm`, `echo`, ...) was silently treated as
    side-effect-free. `superseded_state`/`duplicate_result_elimination`
    already have their own dedicated coverage for what they do with that
    flag (see `tests/recording/test_openai_adapter.py::
    test_side_effects_flag_only_set_for_listed_tools` for the adapter side)
    - this test only checks the wiring that was actually missing: deriving
    the set from the catalog at all.
    """
    catalog = {
        "mkdir": ToolSchema(name="mkdir", effects=ToolEffect.EFFECTFUL),
        "ls": ToolSchema(name="ls", effects=ToolEffect.PURE),
        "unclassified": ToolSchema(name="unclassified"),
    }
    session = ReplaySession(
        "https://api.example.com/v1",
        "sk-test",
        model="m",
        transport=httpx.MockTransport(_ScriptedHandler([_final()])),
    )
    runner = BenchmarkRunner(
        session, "m", FullTracePolicy(), catalog, _TOOLS_PAYLOAD, create_gorilla_file_system
    )
    session.close()

    assert runner._side_effect_tools == {"mkdir"}
