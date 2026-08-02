from pathlib import Path

import httpx
import pytest

from agentslice.errors import AdapterError
from agentslice.ir.events import EventType
from agentslice.recording.jsonl import TraceReader, TraceWriter
from agentslice.recording.live import LiveSession


def _tool_call_response(request: httpx.Request) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"city": "nyc"}',
                                },
                            }
                        ],
                    }
                }
            ]
        },
    )


def test_run_turn_writes_only_new_events(tmp_path: Path) -> None:
    transport = httpx.MockTransport(_tool_call_response)
    session = LiveSession(
        "https://api.example.com/v1", "sk-test", model="gpt-test", transport=transport
    )
    path = tmp_path / "trace.jsonl"
    messages = [{"role": "user", "content": "what's the weather in nyc?"}]

    with TraceWriter(path) as writer:
        updated = session.run_turn(messages, writer)
    session.close()

    assert updated[-1]["tool_calls"][0]["id"] == "call_1"
    events = TraceReader(path).read_all()
    assert [e.type for e in events] == [EventType.USER_GOAL, EventType.TOOL_CALL]


def test_run_turn_raises_adapter_error_on_http_500(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="internal error")

    session = LiveSession(
        "https://api.example.com/v1",
        "sk-test",
        model="gpt-test",
        transport=httpx.MockTransport(handler),
    )
    path = tmp_path / "trace.jsonl"

    with TraceWriter(path) as writer, pytest.raises(AdapterError):
        session.run_turn([{"role": "user", "content": "hi"}], writer)
    session.close()


def test_run_turn_raises_adapter_error_on_timeout(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("timed out", request=request)

    session = LiveSession(
        "https://api.example.com/v1",
        "sk-test",
        model="gpt-test",
        transport=httpx.MockTransport(handler),
    )
    path = tmp_path / "trace.jsonl"

    with TraceWriter(path) as writer, pytest.raises(AdapterError):
        session.run_turn([{"role": "user", "content": "hi"}], writer)
    session.close()


def test_run_turn_raises_adapter_error_on_malformed_response(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"unexpected": "shape"})

    session = LiveSession(
        "https://api.example.com/v1",
        "sk-test",
        model="gpt-test",
        transport=httpx.MockTransport(handler),
    )
    path = tmp_path / "trace.jsonl"

    with TraceWriter(path) as writer, pytest.raises(AdapterError):
        session.run_turn([{"role": "user", "content": "hi"}], writer)
    session.close()
