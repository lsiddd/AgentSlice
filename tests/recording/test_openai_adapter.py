import pytest

from agentslice.errors import UnsupportedMessageFormatError
from agentslice.ir.events import EventType, TraceEvent
from agentslice.ir.graph import build_causal_graph
from agentslice.recording.openai_adapter import (
    from_openai_messages,
    to_openai_messages,
    tool_call_id_of,
)


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
        {"tool_result:call_1.temp", "tool_result:call_1.condition", "conversation:current"}
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
    assert "tool_result:call_1" in result_event.writes
    assert "conversation:current" in result_event.writes
    # every leaf value nested inside the result also gets its own key, so a
    # later tool_call referencing e.g. just the id can still be linked back.
    assert any(key.startswith("tool_result:call_1#") for key in result_event.writes)


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
    assert second_call.reads == frozenset({"tool_result:call_1.issue_id", "user_goal:current"})


def test_empty_assistant_message_with_no_content_and_no_tool_calls_is_skipped() -> None:
    events = from_openai_messages(
        [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": None},
        ]
    )
    assert [e.type for e in events] == [EventType.USER_GOAL]


def test_user_goal_writes_a_versioned_fact() -> None:
    events = from_openai_messages([{"role": "user", "content": "hi"}])
    assert events[0].writes == frozenset({"user_goal:current", "conversation:current"})


def test_model_message_reads_user_goal() -> None:
    events = from_openai_messages(
        [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
    )
    model_message = next(e for e in events if e.type is EventType.MODEL_MESSAGE)
    assert "user_goal:current" in model_message.reads


def test_tool_result_reads_its_own_tool_call() -> None:
    messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [tool_call("call_1", "get_weather", '{"city": "nyc"}')],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "{}"},
    ]
    events = from_openai_messages(messages)
    result_event = next(e for e in events if e.type is EventType.TOOL_RESULT)
    assert result_event.reads == frozenset({"tool_call:call_1"})
    assert tool_call_id_of(result_event) == "call_1"


def test_tool_call_id_of_raises_when_no_link_present() -> None:
    event = TraceEvent(id="a", seq=0, type=EventType.TOOL_RESULT, outputs={"x": 1})
    with pytest.raises(UnsupportedMessageFormatError):
        tool_call_id_of(event)


def test_to_openai_messages_round_trips_a_full_turn() -> None:
    messages = [
        {"role": "system", "content": "be helpful"},
        {"role": "user", "content": "what's the weather in nyc?"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [tool_call("call_1", "get_weather", '{"city": "nyc"}')],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": '{"temp": 72}'},
        {"role": "assistant", "content": "it's 72F in nyc"},
    ]
    events = from_openai_messages(messages)
    rebuilt = to_openai_messages(events)
    assert [m["role"] for m in rebuilt] == ["system", "user", "assistant", "tool", "assistant"]
    assert rebuilt[2]["tool_calls"][0]["function"]["name"] == "get_weather"
    assert rebuilt[3]["tool_call_id"] == "call_1"
    assert rebuilt[4]["content"] == "it's 72F in nyc"


def test_to_openai_messages_groups_consecutive_tool_calls() -> None:
    messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                tool_call("call_1", "get_weather", '{"city": "nyc"}'),
                tool_call("call_2", "get_time", '{"tz": "utc"}'),
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "{}"},
        {"role": "tool", "tool_call_id": "call_2", "content": "{}"},
    ]
    events = from_openai_messages(messages)
    rebuilt = to_openai_messages(events)
    assert [m["role"] for m in rebuilt] == ["assistant", "tool", "tool"]
    assert len(rebuilt[0]["tool_calls"]) == 2


def test_to_openai_messages_raises_on_unsupported_event_type() -> None:
    events = [TraceEvent(id="a", seq=0, type=EventType.STATE_UPDATE)]
    with pytest.raises(UnsupportedMessageFormatError):
        to_openai_messages(events)


def test_to_openai_messages_lowers_epistemic_state_v1_to_canonical_assistant_json() -> None:
    events = [
        TraceEvent(
            id="fold_fh_token_expired",
            seq=0,
            type=EventType.STATE_UPDATE,
            outputs={
                "kind": "epistemic_state",
                "schema_version": 1,
                "ruled_out": [
                    {
                        "fold_id": "fh_token_expired",
                        "hypothesis": "O token expirou",
                        "evidence": [
                            {
                                "event_id": "r1",
                                "json_pointer": "/valid",
                                "operator": "==",
                                "value": True,
                            }
                        ],
                    }
                ],
            },
        )
    ]
    assert to_openai_messages(events) == [
        {
            "role": "assistant",
            "content": (
                '{"_agentslice":{"kind":"epistemic_state","version":1},'
                '"ruled_out":[{"evidence":[{"event_id":"r1","json_pointer":"/valid",'
                '"operator":"==","value":true}],"fold_id":"fh_token_expired",'
                '"hypothesis":"O token expirou"}]}'
            ),
        }
    ]


def test_to_openai_messages_rejects_unknown_epistemic_state_version() -> None:
    events = [
        TraceEvent(
            id="future",
            seq=0,
            type=EventType.STATE_UPDATE,
            outputs={"kind": "epistemic_state", "schema_version": 2, "ruled_out": []},
        )
    ]
    with pytest.raises(UnsupportedMessageFormatError, match="unsupported state_update subtype"):
        to_openai_messages(events)


def test_to_openai_messages_rejects_malformed_epistemic_state_payload() -> None:
    events = [
        TraceEvent(
            id="malformed",
            seq=0,
            type=EventType.STATE_UPDATE,
            outputs={
                "kind": "epistemic_state",
                "schema_version": 1,
                "ruled_out": [{"hypothesis": "missing fold id and evidence"}],
            },
        )
    ]
    with pytest.raises(UnsupportedMessageFormatError, match="unsupported state_update subtype"):
        to_openai_messages(events)


def test_tool_call_id_colliding_with_a_generated_message_id_raises() -> None:
    messages = [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [tool_call("msg_0", "get_weather", "{}")],
        },
    ]
    with pytest.raises(UnsupportedMessageFormatError):
        from_openai_messages(messages)


def test_generated_message_id_colliding_with_an_earlier_tool_call_id_raises() -> None:
    messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [tool_call("msg_1", "get_weather", "{}")],
        },
        {"role": "user", "content": "hi"},
    ]
    with pytest.raises(UnsupportedMessageFormatError):
        from_openai_messages(messages)


def test_second_user_turn_reads_the_previous_turn_via_conversation_key() -> None:
    events = from_openai_messages(
        [
            {"role": "user", "content": "my name is Alice"},
            {"role": "assistant", "content": "nice to meet you, Alice"},
            {"role": "user", "content": "what's my name?"},
        ]
    )
    second_goal = events[-1]
    assert second_goal.type is EventType.USER_GOAL
    assert "conversation:current" in second_goal.reads


def test_forking_at_a_later_turn_keeps_the_conversational_chain_alive() -> None:
    events = from_openai_messages(
        [
            {"role": "user", "content": "my name is Alice"},
            {"role": "assistant", "content": "nice to meet you, Alice"},
            {"role": "user", "content": "what's my name?"},
        ]
    )
    graph = build_causal_graph(events)
    anchor = events[-1]
    assert graph.ancestors(anchor.id) == {events[0].id, events[1].id}


def test_tool_call_reads_a_leaf_value_nested_inside_a_prior_tool_result() -> None:
    messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [tool_call("call_1", "search_users", "{}")],
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": '{"users": [{"id": 123, "name": "Bob"}]}',
        },
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [tool_call("call_2", "get_user", '{"id": 123}')],
        },
    ]
    events = from_openai_messages(messages)
    second_call = next(e for e in events if e.id == "call_2")
    assert any(key.startswith("tool_result:call_1#") for key in second_call.reads)


def test_user_message_reads_a_fact_it_names_as_a_free_text_substring() -> None:
    messages = [
        {"role": "user", "content": "rename report.txt to ArchivedFinalReport2024.txt"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [tool_call("call_mv", "mv", "{}")],
        },
        {
            "role": "tool",
            "tool_call_id": "call_mv",
            "content": '{"result": "report.txt moved to ArchivedFinalReport2024.txt"}',
        },
        {"role": "user", "content": "now sort ArchivedFinalReport2024.txt alphabetically"},
    ]
    events = from_openai_messages(messages)
    second_user_goal = next(e for e in events if e.type is EventType.USER_GOAL and e.seq == 3)
    assert any(key.startswith("tool_result:call_mv") for key in second_user_goal.reads)


def test_user_message_ignores_short_substrings_below_the_match_floor() -> None:
    messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [tool_call("call_1", "get_status", "{}")],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": '{"code": "OK"}'},
        {"role": "user", "content": "is everything OK?"},
    ]
    events = from_openai_messages(messages)
    second_user_goal = next(e for e in events if e.type is EventType.USER_GOAL)
    assert not any(key.startswith("tool_result:call_1") for key in second_user_goal.reads)


def test_tool_result_with_plain_text_content_round_trips_verbatim() -> None:
    messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [tool_call("call_1", "run_command", "{}")],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "Operation completed successfully"},
    ]
    events = from_openai_messages(messages)
    rebuilt = to_openai_messages(events)
    tool_message = next(m for m in rebuilt if m["role"] == "tool")
    assert tool_message["content"] == "Operation completed successfully"


def test_tool_result_with_a_json_object_still_round_trips_as_json() -> None:
    messages = [
        {"role": "assistant", "content": None, "tool_calls": [tool_call("call_1", "run", "{}")]},
        {"role": "tool", "tool_call_id": "call_1", "content": '{"temp": 72}'},
    ]
    events = from_openai_messages(messages)
    rebuilt = to_openai_messages(events)
    tool_message = next(m for m in rebuilt if m["role"] == "tool")
    assert tool_message["content"] == '{"temp": 72}'
