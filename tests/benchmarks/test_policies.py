import json
from typing import Any

from agentslice.compiler.base import ToolSchema
from agentslice.compiler.pipeline import compile_graph
from agentslice.ir.events import TraceEvent
from agentslice.ir.graph import build_causal_graph
from agentslice.recording.openai_adapter import from_openai_messages, to_openai_messages
from benchmarks.policies import (
    CausalCompilePolicy,
    FullTracePolicy,
    LastNTurnsPolicy,
    LLMSummaryPolicy,
    RollingStatePolicy,
)

_TOOL_CATALOG = {"add": ToolSchema(name="add", description="Add two numbers.")}

_MESSAGES = [
    {"role": "user", "content": "what's 2+2?"},
    {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "add", "arguments": json.dumps({"a": 2, "b": 2})},
            }
        ],
    },
    {"role": "tool", "tool_call_id": "call_1", "content": json.dumps({"sum": 4})},
    {"role": "assistant", "content": "It's 4."},
    {"role": "user", "content": "and 3 more?"},
    {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_2",
                "type": "function",
                "function": {"name": "add", "arguments": json.dumps({"a": 4, "b": 3})},
            }
        ],
    },
    {"role": "tool", "tool_call_id": "call_2", "content": json.dumps({"sum": 7})},
    {"role": "assistant", "content": "7."},
]


def _events() -> list[TraceEvent]:
    return from_openai_messages(_MESSAGES)


def test_full_trace_policy_keeps_everything() -> None:
    events = _events()
    request = FullTracePolicy().build_request(events, _TOOL_CATALOG)
    assert len(request.messages) == len(_MESSAGES)
    assert request.tools is not None
    assert request.tools[0]["function"]["name"] == "add"


def test_full_trace_policy_with_no_tools_has_no_tools_field() -> None:
    request = FullTracePolicy().build_request(_events(), {})
    assert request.tools is None


def test_last_n_turns_keeps_only_the_last_turn() -> None:
    request = LastNTurnsPolicy(n_turns=1).build_request(_events(), _TOOL_CATALOG)
    assert request.messages[0] == {"role": "user", "content": "and 3 more?"}
    assert request.tokens < FullTracePolicy().build_request(_events(), _TOOL_CATALOG).tokens


def test_last_n_turns_keeps_tool_call_and_result_paired() -> None:
    request = LastNTurnsPolicy(n_turns=1).build_request(_events(), _TOOL_CATALOG)
    roles = [message["role"] for message in request.messages]
    assert "tool" not in roles or "assistant" in roles


def test_last_n_turns_zero_keeps_only_constraints() -> None:
    request = LastNTurnsPolicy(n_turns=0).build_request(_events(), _TOOL_CATALOG)
    assert request.messages == []


def test_rolling_state_injects_a_json_snapshot_of_earlier_facts() -> None:
    request = RollingStatePolicy().build_request(_events(), _TOOL_CATALOG)
    state_messages = [
        m for m in request.messages if "Known state so far" in (m.get("content") or "")
    ]
    assert len(state_messages) == 1
    assert "4" in state_messages[0]["content"]
    assert request.messages[-1] == {"role": "assistant", "content": "7."}


def test_rolling_state_on_first_turn_has_no_state_message_yet() -> None:
    first_turn_events = from_openai_messages(_MESSAGES[:1])
    request = RollingStatePolicy().build_request(first_turn_events, _TOOL_CATALOG)
    assert not any("Known state so far" in (m.get("content") or "") for m in request.messages)


def test_llm_summary_uses_the_injected_summarizer_for_earlier_turns() -> None:
    seen_messages: list[dict[str, Any]] = []

    def stub_summarizer(messages: list[dict[str, Any]]) -> str:
        seen_messages.extend(messages)
        return "STUB_SUMMARY"

    request = LLMSummaryPolicy(summarizer=stub_summarizer).build_request(_events(), _TOOL_CATALOG)
    summary_messages = [m for m in request.messages if "STUB_SUMMARY" in (m.get("content") or "")]
    assert len(summary_messages) == 1
    assert any(m["content"] == "what's 2+2?" for m in seen_messages)
    assert request.messages[-1] == {"role": "assistant", "content": "7."}


def test_llm_summary_does_not_call_summarizer_on_first_turn() -> None:
    calls: list[Any] = []
    first_turn_events = from_openai_messages(_MESSAGES[:1])
    LLMSummaryPolicy(summarizer=lambda m: calls.append(m) or "x").build_request(
        first_turn_events, _TOOL_CATALOG
    )
    assert calls == []


def test_causal_compile_policy_matches_compile_graph_directly() -> None:
    events = _events()
    request = CausalCompilePolicy().build_request(events, _TOOL_CATALOG)
    graph = build_causal_graph(events)
    compiled = compile_graph(graph, tool_catalog=_TOOL_CATALOG, anchor_event_id=events[-1].id)
    assert request.tokens == compiled.tokens_after
    assert len(request.messages) <= len(_MESSAGES)


def test_causal_compile_policy_respects_a_token_budget() -> None:
    events = _events()
    tight = CausalCompilePolicy(budget_tokens=1).build_request(events, _TOOL_CATALOG)
    loose = CausalCompilePolicy(budget_tokens=None).build_request(events, _TOOL_CATALOG)
    assert tight.tokens <= loose.tokens


def test_causal_compile_policy_accepts_a_custom_pass_sequence() -> None:
    events = _events()
    request = CausalCompilePolicy(passes=()).build_request(events, _TOOL_CATALOG)
    assert request.messages == to_openai_messages(events)


def test_causal_compile_policy_still_offers_every_tool_before_any_is_used() -> None:
    first_turn_events = from_openai_messages(_MESSAGES[:1])
    request = CausalCompilePolicy().build_request(first_turn_events, _TOOL_CATALOG)
    assert request.tools is not None
    assert request.tools[0]["function"]["name"] == "add"
