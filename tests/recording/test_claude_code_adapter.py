from agentslice.ir.events import EventType
from agentslice.recording.claude_code_adapter import from_claude_code_transcript


def _user(content: object, is_sidechain: bool = False) -> dict[str, object]:
    return {
        "type": "user",
        "isSidechain": is_sidechain,
        "message": {"role": "user", "content": content},
    }


def _assistant(
    message_id: str, content: list[dict[str, object]], is_sidechain: bool = False
) -> dict[str, object]:
    return {
        "type": "assistant",
        "isSidechain": is_sidechain,
        "message": {"id": message_id, "content": content},
    }


def test_groups_split_assistant_blocks_and_drops_thinking() -> None:
    records = [
        _user("what's the weather in nyc?"),
        _assistant("msg_1", [{"type": "thinking", "thinking": ""}]),
        _assistant("msg_1", [{"type": "text", "text": "let me check"}]),
        _assistant(
            "msg_1",
            [
                {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "get_weather",
                    "input": {"city": "nyc"},
                }
            ],
        ),
        _user([{"type": "tool_result", "tool_use_id": "toolu_1", "content": "72F"}]),
        _assistant("msg_2", [{"type": "text", "text": "it's 72F in nyc"}]),
    ]

    events = from_claude_code_transcript(records)

    assert [e.type for e in events] == [
        EventType.USER_GOAL,
        EventType.TOOL_CALL,
        EventType.TOOL_RESULT,
        EventType.MODEL_MESSAGE,
    ]
    tool_call = events[1]
    assert tool_call.tool_name == "get_weather"
    assert tool_call.inputs == {"city": "nyc"}
    assert events[3].outputs == {"content": "it's 72F in nyc"}


def test_skips_bookkeeping_record_types() -> None:
    records = [
        {"type": "mode", "mode": "default"},
        {"type": "system", "subtype": "turn_duration", "durationMs": 100},
        {"type": "attachment", "attachment": {"type": "deferred_tools_delta"}},
        _user("hi"),
    ]
    events = from_claude_code_transcript(records)
    assert [e.type for e in events] == [EventType.USER_GOAL]


def test_drops_sidechain_records() -> None:
    records = [
        _user("hi"),
        _assistant("msg_side", [{"type": "text", "text": "sub-agent chatter"}], is_sidechain=True),
        _assistant("msg_1", [{"type": "text", "text": "real reply"}]),
    ]
    events = from_claude_code_transcript(records)
    model_messages = [e for e in events if e.type is EventType.MODEL_MESSAGE]
    assert len(model_messages) == 1
    assert model_messages[0].outputs == {"content": "real reply"}


def test_tool_result_content_as_a_list_of_blocks_is_flattened_to_text() -> None:
    records = [
        _assistant(
            "msg_1", [{"type": "tool_use", "id": "toolu_1", "name": "read_file", "input": {}}]
        ),
        _user(
            [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_1",
                    "content": [
                        {"type": "text", "text": "line one"},
                        {"type": "image", "source": {}},
                        {"type": "text", "text": "line two"},
                    ],
                }
            ]
        ),
    ]
    events = from_claude_code_transcript(records)
    result = next(e for e in events if e.type is EventType.TOOL_RESULT)
    assert result.outputs == {"result": "line one\nline two"}


def test_user_text_block_without_tool_result_is_kept() -> None:
    records = [_user([{"type": "text", "text": "hi with an attachment"}])]
    events = from_claude_code_transcript(records)
    assert [e.type for e in events] == [EventType.USER_GOAL]
    assert events[0].outputs == {"content": "hi with an attachment"}


def test_trailing_unanswered_tool_call_is_kept_as_is() -> None:
    records = [
        _user("hi"),
        _assistant(
            "msg_1", [{"type": "tool_use", "id": "toolu_1", "name": "read_file", "input": {}}]
        ),
    ]
    events = from_claude_code_transcript(records)
    assert [e.type for e in events] == [EventType.USER_GOAL, EventType.TOOL_CALL]


def test_tool_result_content_of_an_unexpected_shape_flattens_to_empty_string() -> None:
    records = [
        _assistant(
            "msg_1", [{"type": "tool_use", "id": "toolu_1", "name": "read_file", "input": {}}]
        ),
        _user([{"type": "tool_result", "tool_use_id": "toolu_1", "content": None}]),
    ]
    events = from_claude_code_transcript(records)
    result = next(e for e in events if e.type is EventType.TOOL_RESULT)
    assert result.outputs == {"result": ""}


def test_no_system_message_is_ever_produced() -> None:
    records = [_user("hi"), _assistant("msg_1", [{"type": "text", "text": "hello"}])]
    events = from_claude_code_transcript(records)
    assert all(e.type is not EventType.CONSTRAINT for e in events)
