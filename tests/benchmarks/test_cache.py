from pathlib import Path

from benchmarks.cache import ResponseCache


def test_get_returns_none_for_a_missing_key(tmp_path: Path) -> None:
    cache = ResponseCache(tmp_path / "cache")
    assert cache.get("missing") is None


def test_set_then_get_round_trips(tmp_path: Path) -> None:
    cache = ResponseCache(tmp_path / "cache")
    cache.set("key1", {"role": "assistant", "content": "hi"})
    assert cache.get("key1") == {"role": "assistant", "content": "hi"}


def test_key_for_is_deterministic_for_the_same_payload() -> None:
    messages = [{"role": "user", "content": "hi"}]
    assert ResponseCache.key_for("m", messages, None) == ResponseCache.key_for("m", messages, None)


def test_key_for_differs_by_model() -> None:
    messages = [{"role": "user", "content": "hi"}]
    assert ResponseCache.key_for("m1", messages, None) != ResponseCache.key_for(
        "m2", messages, None
    )


def test_key_for_differs_by_tools() -> None:
    messages = [{"role": "user", "content": "hi"}]
    tools = [{"type": "function", "function": {"name": "cd"}}]
    assert ResponseCache.key_for("m", messages, None) != ResponseCache.key_for("m", messages, tools)


def test_directory_is_created_on_construction(tmp_path: Path) -> None:
    directory = tmp_path / "nested" / "cache"
    ResponseCache(directory)
    assert directory.is_dir()
