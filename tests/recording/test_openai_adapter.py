import pytest

from agentslice.errors import UnsupportedMessageFormatError
from agentslice.ir.events import EventType
from agentslice.recording.openai_adapter import from_openai_messages


def tool_call(call_id: str, name: str, arguments: str) -> dict[str, object]:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def test_empty_history_produces_no_events() -> None:
    assert from_openai_messages([]) == []


def test_message_without_role_raises() -> None:
    with pytest.raises(UnsupportedMessageFormatError):
        from_openai_messages([{"content": "hi"}])


def test_first_system_message_becomes_a_pinned_constraint() -> None:
    events = from_openai_messages([{"role": "system", "content": "never delete prod"}])
    assert len(events) == 1
    assert events[0].type is EventType.CONSTRAINT
    assert events[0].pinned is True
    assert events[0].outputs == {"content": "never delete prod"}


def test_simple_assistant_message_becomes_model_message() -> None:
    events = from_openai_messages(
        [
            {"role": "user", "content": "what's the weather?"},
            {"role": "assistant", "content": "it's sunny"},
        ]
    )
    assert [e.type for e in events] == [EventType.USER_GOAL, EventType.MODEL_MESSAGE]
    assert events[1].outputs == {"content": "it's sunny"}


def test_multiple_tool_calls_in_one_message_each_become_an_event() -> None:
    messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                tool_call("call_1", "get_weather", '{"city": "nyc"}'),
                tool_call("call_2", "get_time", '{"tz": "utc"}'),
            ],
        },
    ]
    events = from_openai_messages(messages)
    assert [e.type for e in events] == [EventType.TOOL_CALL, EventType.TOOL_CALL]
    assert [e.id for e in events] == ["call_1", "call_2"]
    assert events[0].tool_name == "get_weather"
    assert events[1].tool_name == "get_time"


def test_orphan_tool_call_id_raises() -> None:
    messages = [{"role": "tool", "tool_call_id": "call_missing", "content": "{}"}]
    with pytest.raises(UnsupportedMessageFormatError):
        from_openai_messages(messages)


def test_duplicate_tool_call_id_raises() -> None:
    messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                tool_call("call_1", "get_weather", "{}"),
                tool_call("call_1", "get_weather", "{}"),
            ],
        }
    ]
    with pytest.raises(UnsupportedMessageFormatError):
        from_openai_messages(messages)


def test_tool_result_with_shallow_dict_writes_one_key_per_field() -> None:
    messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [tool_call("call_1", "get_weather", '{"city": "nyc"}')],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": '{"temp": 70, "condition": "sunny"}'},
    ]
    events = from_openai_messages(messages)
    result_event = events[-1]
    assert result_event.type is EventType.TOOL_RESULT
    assert result_event.writes == frozenset(
        {"tool_result:call_1.temp", "tool_result:call_1.condition"}
    )


def test_tool_result_with_nested_data_writes_one_opaque_key() -> None:
    messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [tool_call("call_1", "get_issue", "{}")],
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": '{"title": "bug", "comments": [{"body": "x"}]}',
        },
    ]
    events = from_openai_messages(messages)
    result_event = events[-1]
    assert result_event.writes == frozenset({"tool_result:call_1"})


def test_side_effects_flag_only_set_for_listed_tools() -> None:
    messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                tool_call("call_1", "delete_file", '{"path": "a.txt"}'),
                tool_call("call_2", "read_file", '{"path": "b.txt"}'),
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "{}"},
        {"role": "tool", "tool_call_id": "call_2", "content": "{}"},
    ]
    events = from_openai_messages(messages, side_effect_tools={"delete_file"})
    results = {e.tool_name: e for e in events if e.type is EventType.TOOL_RESULT}
    assert results["delete_file"].side_effects is True
    assert results["read_file"].side_effects is False


def test_tool_call_reads_inferred_by_value_match() -> None:
    messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [tool_call("call_1", "search_issue", '{"query": "race condition"}')],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": '{"issue_id": 184}'},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [tool_call("call_2", "get_issue", '{"id": 184}')],
        },
    ]
    events = from_openai_messages(messages)
    second_call = next(e for e in events if e.id == "call_2")
    assert second_call.reads == frozenset({"tool_result:call_1.issue_id"})


def test_empty_assistant_message_with_no_content_and_no_tool_calls_is_skipped() -> None:
    events = from_openai_messages(
        [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": None},
        ]
    )
    assert [e.type for e in events] == [EventType.USER_GOAL]
