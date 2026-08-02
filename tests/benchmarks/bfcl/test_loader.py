from pathlib import Path

import pytest

from benchmarks.bfcl.loader import load_gorilla_file_system_tool_catalog, load_tasks
from benchmarks.errors import MalformedTaskError


def test_load_tasks_returns_only_gorilla_file_system_tasks() -> None:
    tasks = load_tasks()
    assert len(tasks) == 13
    assert all(task.involved_classes == ["GorillaFileSystem"] for task in tasks)
    assert all(len(task.turns) == len(task.ground_truth) for task in tasks)
    assert len({task.id for task in tasks}) == len(tasks)


def test_load_tasks_rejects_invalid_json_line(tmp_path: Path) -> None:
    path = tmp_path / "broken.jsonl"
    path.write_text("{not json}\n")
    with pytest.raises(MalformedTaskError):
        load_tasks(path)


def test_load_tasks_rejects_a_line_missing_required_fields(tmp_path: Path) -> None:
    path = tmp_path / "broken.jsonl"
    path.write_text('{"id": "x"}\n')
    with pytest.raises(MalformedTaskError):
        load_tasks(path)


def test_load_tasks_skips_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "tasks.jsonl"
    path.write_text(
        '{"id": "t1", "turns": [], "initial_config": {}, '
        '"involved_classes": [], "ground_truth": []}\n\n\n'
    )
    assert len(load_tasks(path)) == 1


def test_load_gorilla_file_system_tool_catalog_has_18_functions_shaped_for_openai() -> None:
    tools = load_gorilla_file_system_tool_catalog()
    assert len(tools) == 18
    names = {tool["function"]["name"] for tool in tools}
    assert {"cd", "mkdir", "mv", "grep", "sort", "diff", "ls", "cat"} <= names
    for tool in tools:
        assert tool["type"] == "function"
        assert tool["function"]["parameters"]["type"] == "object"
        assert tool["effects"] in {"pure", "effectful", "unknown"}


def test_load_gorilla_file_system_tool_catalog_rejects_non_array(tmp_path: Path) -> None:
    path = tmp_path / "tools.json"
    path.write_text('{"not": "a list"}')
    with pytest.raises(MalformedTaskError):
        load_gorilla_file_system_tool_catalog(path)
