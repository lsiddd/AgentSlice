import json

import httpx
import pytest

from agentslice.compiler.base import ToolSchema
from agentslice.errors import AdapterError
from agentslice.ir.events import EventType, TraceEvent
from agentslice.replay.runtime import ReplaySession, replay_compiled_context


def _next_action_response(request: httpx.Request) -> httpx.Response:
    return httpx.Response(
        200,
        json={"choices": [{"message": {"role": "assistant", "content": "it's 72F in nyc"}}]},
    )


def test_next_action_returns_the_assistant_message() -> None:
    session = ReplaySession(
        "https://api.example.com/v1",
        "sk-test",
        model="gpt-test",
        transport=httpx.MockTransport(_next_action_response),
    )
    message = session.next_action([{"role": "user", "content": "weather?"}])
    session.close()
    assert message == {"role": "assistant", "content": "it's 72F in nyc"}


def test_session_usable_as_a_context_manager() -> None:
    with ReplaySession(
        "https://api.example.com/v1",
        "sk-test",
        model="gpt-test",
        transport=httpx.MockTransport(_next_action_response),
    ) as session:
        message = session.next_action([{"role": "user", "content": "weather?"}])
    assert message["content"] == "it's 72F in nyc"


def test_next_action_raises_adapter_error_on_http_500() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="internal error")

    session = ReplaySession(
        "https://api.example.com/v1",
        "sk-test",
        model="gpt-test",
        transport=httpx.MockTransport(handler),
    )
    with pytest.raises(AdapterError):
        session.next_action([{"role": "user", "content": "hi"}])
    session.close()


def test_next_action_raises_adapter_error_on_invalid_json_body() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="not json")

    session = ReplaySession(
        "https://api.example.com/v1",
        "sk-test",
        model="gpt-test",
        transport=httpx.MockTransport(handler),
    )
    with pytest.raises(AdapterError):
        session.next_action([{"role": "user", "content": "hi"}])
    session.close()


def test_next_action_raises_adapter_error_on_malformed_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"unexpected": "shape"})

    session = ReplaySession(
        "https://api.example.com/v1",
        "sk-test",
        model="gpt-test",
        transport=httpx.MockTransport(handler),
    )
    with pytest.raises(AdapterError):
        session.next_action([{"role": "user", "content": "hi"}])
    session.close()


def test_replay_compiled_context_fills_pending_call_and_sends_tools() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["payload"] = json.loads(request.content)
        return httpx.Response(
            200, json={"choices": [{"message": {"role": "assistant", "content": "done"}}]}
        )

    call = TraceEvent(
        id="call_1",
        seq=0,
        type=EventType.TOOL_CALL,
        tool_name="get_weather",
        inputs={"city": "nyc"},
        writes=frozenset({"tool_call:call_1"}),
    )
    result = TraceEvent(
        id="result_1",
        seq=1,
        type=EventType.TOOL_RESULT,
        outputs={"temp": 72},
        reads=frozenset({"tool_call:call_1"}),
        writes=frozenset({"tool_result:call_1.temp"}),
    )

    session = ReplaySession(
        "https://api.example.com/v1",
        "sk-test",
        model="gpt-test",
        transport=httpx.MockTransport(handler),
    )
    reply = replay_compiled_context(
        [call],
        original_events=[call, result],
        session=session,
        tool_catalog={"get_weather": ToolSchema(name="get_weather", description="", parameters={})},
    )
    session.close()

    assert reply == {"role": "assistant", "content": "done"}
    payload = captured["payload"]
    assert isinstance(payload, dict)
    tool_message = next(m for m in payload["messages"] if m["role"] == "tool")
    assert json.loads(tool_message["content"]) == {"temp": 72}
    assert payload["tools"][0]["function"]["name"] == "get_weather"


def test_replay_compiled_context_lowers_epistemic_state_as_assistant_context() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["payload"] = json.loads(request.content)
        return httpx.Response(
            200, json={"choices": [{"message": {"role": "assistant", "content": "continue"}}]}
        )

    epistemic_state = TraceEvent(
        id="fold_fh_token",
        seq=0,
        type=EventType.STATE_UPDATE,
        outputs={
            "kind": "epistemic_state",
            "schema_version": 1,
            "ruled_out": [
                {
                    "fold_id": "fh_token",
                    "hypothesis": "The token expired",
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
    session = ReplaySession(
        "https://api.example.com/v1",
        "sk-test",
        model="gpt-test",
        transport=httpx.MockTransport(handler),
    )

    reply = replay_compiled_context(
        [epistemic_state],
        original_events=[epistemic_state],
        session=session,
    )
    session.close()

    assert reply["content"] == "continue"
    payload = captured["payload"]
    assert isinstance(payload, dict)
    message = payload["messages"][0]
    assert message["role"] == "assistant"
    assert json.loads(message["content"]) == {
        "_agentslice": {"kind": "epistemic_state", "version": 1},
        "ruled_out": epistemic_state.outputs["ruled_out"],
    }
