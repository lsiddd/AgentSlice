"""Conversion of OpenAI-compatible chat messages into a TraceEvent sequence.

OpenAI-compatible chat completions (the format OpenAI itself and OpenRouter
both use) carry no explicit ``reads``/``writes``/``side_effects`` metadata,
so this adapter has to infer it. The rules, deliberately conservative:

- A ``tool_call``'s ``reads`` are the keys of every known fact whose current
  value appears, verbatim, as a leaf value somewhere in the call's
  arguments. No match, no read: a call motivated by free-form reasoning
  rather than a traceable prior value gets an empty ``reads``.
- A ``tool_result``'s ``writes`` is one key per top-level field when its
  parsed output is a shallow dict (no nested dict/list values), or a single
  opaque key otherwise. ``side_effects`` is only ever ``True`` when the
  tool's name appears in the caller-supplied ``side_effect_tools`` set;
  there is no prefix-based guessing (``delete_``, ``write_`` etc.).
- A ``model_message`` (assistant content with no tool calls) conservatively
  reads every fact written since the previous ``model_message``. This
  over-approximates on purpose: it never omits something the model could
  plausibly have used, at the cost of limiting how much
  ``tool_result_projection`` can later trim from free-form text.

An assistant message that carries both ``tool_calls`` and ``content``
emits only the tool calls; the accompanying content is treated as
non-load-bearing reasoning and dropped, a deliberate v0.1 simplification.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from typing import Any

from agentslice.errors import UnsupportedMessageFormatError
from agentslice.ir.events import EventType, TraceEvent


def _iter_leaf_values(value: Any) -> Iterable[Any]:
    if isinstance(value, Mapping):
        for item in value.values():
            yield from _iter_leaf_values(item)
    elif isinstance(value, list):
        for item in value:
            yield from _iter_leaf_values(item)
    else:
        yield value


def _is_matchable(value: Any) -> bool:
    return value is not None and not isinstance(value, bool) and value != ""


def _is_shallow_dict(value: Any) -> bool:
    if not isinstance(value, dict) or not value:
        return False
    return all(not isinstance(v, dict | list) for v in value.values())


def from_openai_messages(
    messages: list[dict[str, Any]],
    *,
    side_effect_tools: set[str] | None = None,
) -> list[TraceEvent]:
    """Convert an OpenAI-compatible message history into ``TraceEvent`` objects.

    Args:
        messages: Chat messages in the OpenAI chat-completions shape, each
            with a ``role`` of ``system``, ``user``, ``assistant``, or
            ``tool``.
        side_effect_tools: Names of tools whose results should be marked
            ``side_effects=True``. Tools not listed here are assumed
            side-effect-free.

    Returns:
        Events in canonical order, with ``seq`` assigned sequentially. A
        single message can produce zero events (an empty assistant
        message), one event, or several (an assistant message with
        multiple tool calls).

    Raises:
        UnsupportedMessageFormatError: A message is missing ``role``, a
            ``tool`` message's ``tool_call_id`` has no matching prior tool
            call, a ``tool_call`` id is reused, or a tool call's arguments
            are not valid JSON.
    """
    side_effect_tools = side_effect_tools or set()

    events: list[TraceEvent] = []
    seq = 0
    known_values: dict[str, Any] = {}
    since_last_model_message: set[str] = set()
    call_id_to_tool_name: dict[str, str] = {}
    seen_tool_call_ids: set[str] = set()

    for message_index, message in enumerate(messages):
        role = message.get("role")
        if not role:
            raise UnsupportedMessageFormatError(f"message {message_index}: missing 'role'")

        if role == "system":
            events.append(
                TraceEvent(
                    id=f"msg_{message_index}",
                    seq=seq,
                    type=EventType.CONSTRAINT,
                    outputs={"content": message.get("content") or ""},
                    pinned=True,
                )
            )
            seq += 1

        elif role == "user":
            events.append(
                TraceEvent(
                    id=f"msg_{message_index}",
                    seq=seq,
                    type=EventType.USER_GOAL,
                    outputs={"content": message.get("content") or ""},
                )
            )
            seq += 1

        elif role == "assistant":
            tool_calls = message.get("tool_calls") or []
            if tool_calls:
                for call in tool_calls:
                    call_id = call["id"]
                    if call_id in seen_tool_call_ids:
                        raise UnsupportedMessageFormatError(
                            f"message {message_index}: duplicate tool_call id {call_id!r}"
                        )
                    seen_tool_call_ids.add(call_id)

                    function = call["function"]
                    tool_name = function["name"]
                    call_id_to_tool_name[call_id] = tool_name
                    try:
                        inputs = json.loads(function.get("arguments") or "{}")
                    except json.JSONDecodeError as exc:
                        raise UnsupportedMessageFormatError(
                            f"message {message_index}: tool_call {call_id!r} has invalid "
                            f"JSON arguments: {exc}"
                        ) from exc

                    reads = {
                        key
                        for key, value in known_values.items()
                        if _is_matchable(value)
                        and any(value == leaf for leaf in _iter_leaf_values(inputs))
                    }
                    events.append(
                        TraceEvent(
                            id=call_id,
                            seq=seq,
                            type=EventType.TOOL_CALL,
                            tool_name=tool_name,
                            inputs=inputs,
                            reads=frozenset(reads),
                        )
                    )
                    seq += 1
            else:
                content = message.get("content")
                if not content:
                    continue
                events.append(
                    TraceEvent(
                        id=f"msg_{message_index}",
                        seq=seq,
                        type=EventType.MODEL_MESSAGE,
                        outputs={"content": content},
                        reads=frozenset(since_last_model_message),
                    )
                )
                since_last_model_message = set()
                seq += 1

        elif role == "tool":
            call_id = message.get("tool_call_id")
            if call_id is None or call_id not in call_id_to_tool_name:
                raise UnsupportedMessageFormatError(
                    f"message {message_index}: tool result has no matching tool_call "
                    f"(tool_call_id={call_id!r})"
                )
            tool_name = call_id_to_tool_name[call_id]
            raw_content = message.get("content") or ""
            parsed: Any = None
            if isinstance(raw_content, str):
                try:
                    parsed = json.loads(raw_content)
                except json.JSONDecodeError:
                    parsed = None
            else:
                parsed = raw_content
            outputs: dict[str, Any] = (
                parsed if isinstance(parsed, dict) else {"result": raw_content}
            )

            writes: set[str] = set()
            if _is_shallow_dict(outputs):
                for field_name, value in outputs.items():
                    key = f"tool_result:{call_id}.{field_name}"
                    writes.add(key)
                    known_values[key] = value
            else:
                key = f"tool_result:{call_id}"
                writes.add(key)
                known_values[key] = outputs
            since_last_model_message |= writes

            events.append(
                TraceEvent(
                    id=call_id,
                    seq=seq,
                    type=EventType.TOOL_RESULT,
                    tool_name=tool_name,
                    outputs=outputs,
                    writes=frozenset(writes),
                    side_effects=tool_name in side_effect_tools,
                )
            )
            seq += 1

        else:
            raise UnsupportedMessageFormatError(
                f"message {message_index}: unsupported role {role!r}"
            )

    return events
