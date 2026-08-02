import pytest

from agentslice.errors import UnknownAnchorError
from agentslice.ir.events import EventType, TraceEvent
from agentslice.recording.openai_adapter import from_openai_messages
from agentslice.replay.comparator import extract_next_recorded_action, next_action_equivalence


def _tool_call_message(call_id: str, name: str, arguments: str) -> dict[str, object]:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}
        ],
    }


def test_extract_next_recorded_action_for_a_tool_call_group() -> None:
    messages = [
        {"role": "user", "content": "what's the weather in nyc?"},
        _tool_call_message("call_1", "get_weather", '{"city": "nyc"}'),
        {"role": "tool", "tool_call_id": "call_1", "content": '{"temp": 72}'},
        _tool_call_message("call_2", "get_time", '{"tz": "utc"}'),
    ]
    events = from_openai_messages(messages)
    tool_result = next(e for e in events if e.type is EventType.TOOL_RESULT)

    action = extract_next_recorded_action(events, tool_result.id)

    assert action is not None
    assert action["tool_calls"][0]["function"]["name"] == "get_time"


def test_extract_next_recorded_action_for_a_model_message() -> None:
    messages = [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": '{"temp": 72}'},
        {"role": "assistant", "content": "it's 72F"},
    ]
    events = from_openai_messages(messages)
    tool_result = next(e for e in events if e.type is EventType.TOOL_RESULT)

    action = extract_next_recorded_action(events, tool_result.id)

    assert action == {"role": "assistant", "content": "it's 72F"}


def test_extract_next_recorded_action_stops_grouping_at_the_next_non_call_event() -> None:
    messages = [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": "{}"},
                },
                {
                    "id": "call_2",
                    "type": "function",
                    "function": {"name": "get_time", "arguments": "{}"},
                },
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "{}"},
        {"role": "tool", "tool_call_id": "call_2", "content": "{}"},
    ]
    events = from_openai_messages(messages)
    user_goal = next(e for e in events if e.type is EventType.USER_GOAL)

    action = extract_next_recorded_action(events, user_goal.id)

    assert action is not None
    assert [c["function"]["name"] for c in action["tool_calls"]] == ["get_weather", "get_time"]


def test_extract_next_recorded_action_returns_none_when_anchor_is_last_event() -> None:
    events = from_openai_messages([{"role": "user", "content": "hi"}])
    assert extract_next_recorded_action(events, events[-1].id) is None


def test_extract_next_recorded_action_returns_none_when_next_is_a_new_user_goal() -> None:
    events = from_openai_messages(
        [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "user", "content": "and now?"},
        ]
    )
    model_message = next(e for e in events if e.type is EventType.MODEL_MESSAGE)
    assert extract_next_recorded_action(events, model_message.id) is None


def test_extract_next_recorded_action_raises_for_unknown_anchor() -> None:
    events = [TraceEvent(id="a", seq=0, type=EventType.USER_GOAL)]
    with pytest.raises(UnknownAnchorError):
        extract_next_recorded_action(events, "missing")


def test_equivalence_true_for_matching_tool_calls_regardless_of_order() -> None:
    original = {
        "tool_calls": [
            {"function": {"name": "a", "arguments": '{"x": 1}'}},
            {"function": {"name": "b", "arguments": '{"y": 2}'}},
        ]
    }
    replayed = {
        "tool_calls": [
            {"function": {"name": "b", "arguments": '{"y":2}'}},
            {"function": {"name": "a", "arguments": '{"x":1}'}},
        ]
    }
    assert next_action_equivalence(original, replayed) is True


def test_equivalence_false_for_different_arguments() -> None:
    original = {"tool_calls": [{"function": {"name": "a", "arguments": '{"x": 1}'}}]}
    replayed = {"tool_calls": [{"function": {"name": "a", "arguments": '{"x": 2}'}}]}
    assert next_action_equivalence(original, replayed) is False


def test_equivalence_true_for_two_final_text_answers() -> None:
    original = {"content": "it's sunny"}
    replayed = {"content": "sunny today"}
    assert next_action_equivalence(original, replayed) is True


def test_equivalence_false_when_only_one_side_calls_a_tool() -> None:
    original = {"content": "it's sunny"}
    replayed = {"tool_calls": [{"function": {"name": "get_weather", "arguments": "{}"}}]}
    assert next_action_equivalence(original, replayed) is False


def test_equivalence_false_when_one_side_repeats_an_identical_call() -> None:
    original = {
        "tool_calls": [
            {"function": {"name": "send_email", "arguments": '{"to": "a@example.com"}'}},
            {"function": {"name": "send_email", "arguments": '{"to": "a@example.com"}'}},
        ]
    }
    replayed = {
        "tool_calls": [
            {"function": {"name": "send_email", "arguments": '{"to": "a@example.com"}'}},
        ]
    }
    assert next_action_equivalence(original, replayed) is False


def test_equivalence_falls_back_to_raw_string_for_non_json_arguments() -> None:
    original = {"tool_calls": [{"function": {"name": "a", "arguments": "not json"}}]}
    replayed = {"tool_calls": [{"function": {"name": "a", "arguments": "not json"}}]}
    assert next_action_equivalence(original, replayed) is True
