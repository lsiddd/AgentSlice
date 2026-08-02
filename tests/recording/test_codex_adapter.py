from agentslice.ir.events import EventType
from agentslice.recording.codex_adapter import from_codex_rollout


def _item(payload: dict[str, object]) -> dict[str, object]:
    return {"type": "response_item", "payload": payload}


def _message(role: str, text: str) -> dict[str, object]:
    return _item(
        {"type": "message", "role": role, "content": [{"type": "input_text", "text": text}]}
    )


def test_developer_role_maps_to_system() -> None:
    records = [_message("developer", "be careful"), _message("user", "hi")]
    events = from_codex_rollout(records)
    assert events[0].type is EventType.CONSTRAINT
    assert events[0].outputs == {"content": "be careful"}


def test_groups_consecutive_function_calls_into_one_assistant_turn() -> None:
    records = [
        _message("user", "check the weather and the time"),
        _item(
            {
                "type": "function_call",
                "name": "get_weather",
                "arguments": '{"city": "nyc"}',
                "call_id": "call_1",
            }
        ),
        _item(
            {
                "type": "function_call",
                "name": "get_time",
                "arguments": '{"tz": "utc"}',
                "call_id": "call_2",
            }
        ),
        _item({"type": "function_call_output", "call_id": "call_1", "output": "72F"}),
        _item({"type": "function_call_output", "call_id": "call_2", "output": "12:00"}),
    ]
    events = from_codex_rollout(records)
    assert [e.type for e in events] == [
        EventType.USER_GOAL,
        EventType.TOOL_CALL,
        EventType.TOOL_CALL,
        EventType.TOOL_RESULT,
        EventType.TOOL_RESULT,
    ]
    assert [e.tool_name for e in events if e.type is EventType.TOOL_CALL] == [
        "get_weather",
        "get_time",
    ]


def test_function_call_output_dict_shape_extracts_output_field() -> None:
    records = [
        _item(
            {
                "type": "function_call",
                "name": "exec_command",
                "arguments": "{}",
                "call_id": "call_1",
            }
        ),
        _item(
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": {"output": "hello from shell", "metadata": {"exit_code": 0}},
            }
        ),
    ]
    events = from_codex_rollout(records)
    result = next(e for e in events if e.type is EventType.TOOL_RESULT)
    assert result.outputs == {"result": "hello from shell"}


def test_reasoning_items_are_skipped() -> None:
    records = [
        _item({"type": "reasoning", "summary": [{"type": "summary_text", "text": "thinking..."}]}),
        _message("user", "hi"),
    ]
    events = from_codex_rollout(records)
    assert [e.type for e in events] == [EventType.USER_GOAL]


def test_message_with_an_unrecognized_role_is_skipped() -> None:
    records = [_message("tool", "orphan"), _message("user", "hi")]
    events = from_codex_rollout(records)
    assert [e.type for e in events] == [EventType.USER_GOAL]


def test_non_response_item_records_are_skipped() -> None:
    records = [
        {"type": "session_meta", "payload": {"id": "abc"}},
        {"type": "turn_context", "payload": {}},
        _message("user", "hi"),
    ]
    events = from_codex_rollout(records)
    assert [e.type for e in events] == [EventType.USER_GOAL]
