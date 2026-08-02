import pytest

from benchmarks.bfcl.loader import load_tasks
from benchmarks.environments.base import safe_call
from benchmarks.environments.gorilla_file_system import GorillaFileSystemEnvironment, create
from benchmarks.errors import InvalidCallError, UnknownFunctionError
from benchmarks.ground_truth import param_names_from_tool_catalog, parse_turn

_INITIAL_CONFIG = {
    "GorillaFileSystem": {
        "root": {
            "alex": {
                "type": "directory",
                "contents": {
                    "workspace": {
                        "type": "directory",
                        "contents": {
                            "log.txt": {"type": "file", "content": "line one\nline two"},
                            "archive": {"type": "directory", "contents": {}},
                        },
                    }
                },
            }
        }
    }
}


def test_create_loads_scenario_and_snapshot_matches() -> None:
    env = create(_INITIAL_CONFIG)
    snapshot = env.snapshot()
    assert snapshot["type"] == "directory"
    assert "workspace" in snapshot["contents"]


def test_create_with_empty_config_is_an_empty_root() -> None:
    env = create({})
    assert env.snapshot() == {"type": "directory", "contents": {}}


def test_cd_and_pwd() -> None:
    env = create(_INITIAL_CONFIG)
    assert env.call("cd", {"folder": "workspace"}) == {"current_working_directory": "workspace"}
    assert env.call("pwd", {}) == {"current_working_directory": "/workspace"}


def test_cd_into_missing_directory_returns_error_not_raise() -> None:
    env = create(_INITIAL_CONFIG)
    result = env.call("cd", {"folder": "nope"})
    assert "error" in result


def test_cd_dotdot_at_root_returns_error() -> None:
    env = create(_INITIAL_CONFIG)
    result = env.call("cd", {"folder": ".."})
    assert "error" in result


def test_mkdir_touch_echo_cat_round_trip() -> None:
    env = create(_INITIAL_CONFIG)
    env.call("cd", {"folder": "workspace"})
    assert env.call("mkdir", {"dir_name": "notes"}) == {}
    assert env.call("touch", {"file_name": "notes/todo.txt"}).get("error") is not None
    assert env.call("touch", {"file_name": "todo.txt"}) == {}
    env.call("echo", {"content": "buy milk", "file_name": "todo.txt"})
    assert env.call("cat", {"file_name": "todo.txt"}) == {"file_content": "buy milk"}


def test_mkdir_existing_name_is_an_error() -> None:
    env = create(_INITIAL_CONFIG)
    env.call("cd", {"folder": "workspace"})
    result = env.call("mkdir", {"dir_name": "archive"})
    assert "error" in result


def test_mv_file_into_directory_and_grep_sort_diff() -> None:
    env = create(_INITIAL_CONFIG)
    env.call("cd", {"folder": "workspace"})
    assert env.call("mv", {"source": "log.txt", "destination": "archive"}) == {
        "result": "'log.txt' moved to 'archive/log.txt'"
    }
    env.call("cd", {"folder": "archive"})
    assert env.call("grep", {"file_name": "log.txt", "pattern": "two"}) == {
        "matching_lines": ["line two"]
    }
    assert env.call("sort", {"file_name": "log.txt"}) == {"sorted_content": "line one\nline two"}
    env.call("echo", {"content": "line one\nline two", "file_name": "copy.txt"})
    env.call("touch", {"file_name": "copy.txt"}).get("error")
    assert env.call("diff", {"file_name1": "log.txt", "file_name2": "log.txt"}) == {
        "diff_lines": ""
    }


def test_mv_rejects_path_in_destination() -> None:
    env = create(_INITIAL_CONFIG)
    env.call("cd", {"folder": "workspace"})
    result = env.call("mv", {"source": "log.txt", "destination": "a/b"})
    assert "error" in result


def test_cp_copies_without_removing_source() -> None:
    env = create(_INITIAL_CONFIG)
    env.call("cd", {"folder": "workspace"})
    env.call("cp", {"source": "log.txt", "destination": "archive"})
    assert env.call("ls", {}) == {"current_directory_content": ["log.txt", "archive"]}
    env.call("cd", {"folder": "archive"})
    assert env.call("cat", {"file_name": "log.txt"}) == {"file_content": "line one\nline two"}


def test_rm_and_rmdir() -> None:
    env = create(_INITIAL_CONFIG)
    env.call("cd", {"folder": "workspace"})
    assert env.call("rmdir", {"dir_name": "archive"}) == {"result": "'archive' removed"}
    assert "error" in env.call("rmdir", {"dir_name": "does_not_exist"})
    assert env.call("rm", {"file_name": "log.txt"}) == {"result": "'log.txt' removed"}


def test_rmdir_non_empty_is_an_error() -> None:
    env = create(_INITIAL_CONFIG)
    env.call("cd", {"folder": "workspace"})
    env.call("touch", {"file_name": "keep.txt"})
    env.call("mkdir", {"dir_name": "full"})
    env.call("cd", {"folder": "full"})
    env.call("touch", {"file_name": "x.txt"})
    env.call("cd", {"folder": ".."})
    assert "error" in env.call("rmdir", {"dir_name": "full"})


def test_find_and_wc_and_du_and_tail() -> None:
    env = create(_INITIAL_CONFIG)
    assert env.call("find", {"name": "log.txt"}) == {"matches": ["./workspace/log.txt"]}
    env.call("cd", {"folder": "workspace"})
    assert env.call("wc", {"file_name": "log.txt", "mode": "l"}) == {"count": 2, "type": "lines"}
    assert env.call("du", {}) == {"disk_usage": "17 bytes"}
    assert env.call("tail", {"file_name": "log.txt", "lines": 1}) == {"last_lines": "line two"}


def test_unknown_function_raises() -> None:
    env = GorillaFileSystemEnvironment()
    with pytest.raises(UnknownFunctionError):
        env.call("delete_everything", {})


def test_private_method_is_not_callable_through_call() -> None:
    env = GorillaFileSystemEnvironment()
    with pytest.raises(UnknownFunctionError):
        env.call("_navigate", {"path": "/"})


def test_bad_kwargs_raise_invalid_call_error() -> None:
    env = GorillaFileSystemEnvironment()
    with pytest.raises(InvalidCallError):
        env.call("cd", {"nonexistent_kwarg": "x"})


def test_safe_call_turns_errors_into_a_result_dict() -> None:
    env = GorillaFileSystemEnvironment()
    result, is_valid = safe_call(env, "delete_everything", {})
    assert is_valid is False
    assert "error" in result


def test_snapshot_equality_ignores_current_directory() -> None:
    left = create(_INITIAL_CONFIG)
    right = create(_INITIAL_CONFIG)
    left.call("cd", {"folder": "workspace"})
    assert left.snapshot() == right.snapshot()


def test_bundled_ground_truth_executes_cleanly_against_the_reference_port() -> None:
    from benchmarks.bfcl.loader import load_gorilla_file_system_tool_catalog

    param_names = param_names_from_tool_catalog(load_gorilla_file_system_tool_catalog())
    for task in load_tasks():
        env = create(task.initial_config)
        for turn_calls in task.ground_truth:
            for call in parse_turn(turn_calls, param_names):
                result = env.call(call.name, call.kwargs)
                assert "error" not in result, (task.id, call, result)
