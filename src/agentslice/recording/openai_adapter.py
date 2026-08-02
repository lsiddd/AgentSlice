"""Conversion of OpenAI-compatible chat messages into a TraceEvent sequence.

OpenAI-compatible chat completions (the format OpenAI itself and OpenRouter
both use) carry no explicit ``reads``/``writes``/``side_effects`` metadata,
so this adapter has to infer it. The rules, deliberately conservative:

- A ``tool_call``'s ``reads`` are the keys of every known fact whose current
  value appears, verbatim, as a leaf value somewhere in the call's
  arguments, plus ``user_goal:current`` (see below). A call motivated by
  free-form reasoning rather than a traceable prior value still reads the
  active goal.
- A ``tool_result``'s ``writes`` is one key per top-level field when its
  parsed output is a shallow dict (no nested dict/list values), or a single
  opaque key otherwise. Its ``reads`` always includes ``tool_call:{id}``,
  the fact its own ``tool_call`` wrote: a result cannot exist without the
  call that produced it, so this keeps the two alive or dead together
  under ``dead_events`` rather than letting a result survive as an orphan.
  ``side_effects`` is only ever ``True`` when the tool's name appears in
  the caller-supplied ``side_effect_tools`` set; there is no prefix-based
  guessing (``delete_``, ``write_`` etc.).
- Every ``user`` message writes ``user_goal:current``, a fact re-versioned
  (not accumulated) by each new user turn. Every ``tool_call`` and
  ``model_message`` reads it, so whichever goal was active when they
  happened stays a causal ancestor of anything that followed from it —
  without this, a user's goal writes no fact at all and is invisible to
  ``dead_events``, which only tracks fact dependencies, not message order.
- A ``model_message`` (assistant content with no tool calls) conservatively
  reads every fact written since the previous ``model_message``, plus
  ``user_goal:current``. This over-approximates on purpose: it never omits
  something the model could plausibly have used, at the cost of limiting
  how much ``tool_result_projection`` can later trim from free-form text.
- Every event also writes a versioned ``conversation:current`` fact, and
  every ``user`` message (after the first) reads it. This chains each new
  user turn to whatever the immediately preceding event was — usually the
  previous turn's final ``model_message`` — so that forking mid-conversation
  doesn't strand a follow-up question ("what's my name?") with no causal
  path back to the turn that actually answers it. It deliberately does
  *not* chain every event to every other one: only ``user`` messages read
  it, so a tool call or result that isn't otherwise depended on can still
  be pruned normally.
- A ``tool_result`` whose parsed output is a *nested* structure (a dict
  containing another dict/list, or a list) still writes one opaque key for
  the whole value, but additionally indexes every leaf value it contains
  under its own key (``tool_result:{id}#{n}``), so a later ``tool_call``
  that references, say, a single id buried inside a nested response can
  still be linked back to it by value match — without this, a value only
  reachable through nesting was invisible to the read/write matching above
  and could never establish a causal edge.
- A ``tool_result`` whose content wasn't valid JSON (or wasn't a JSON
  object) keeps its verbatim original string in ``metadata["raw_content"]``,
  so :func:`to_openai_messages` can reproduce it exactly. Without this, a
  plain-text result like ``"done"`` would round-trip through
  ``{"result": "done"}`` and back out as a JSON-encoded string instead of
  the plain text the model actually saw.

An assistant message that carries both ``tool_calls`` and ``content``
emits only the tool calls; the accompanying content is treated as
non-load-bearing reasoning and dropped, a deliberate v0.1 simplification.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from agentslice.errors import UnsupportedMessageFormatError
from agentslice.ir.events import EventType, TraceEvent

_USER_GOAL_KEY = "user_goal:current"
_TOOL_CALL_KEY_PREFIX = "tool_call:"
_CONVERSATION_KEY = "conversation:current"


def tool_call_id_of(event: TraceEvent) -> str:
    """Recover the ``tool_call_id`` a ``tool_result`` event answers.

    Reads it off the ``tool_call:{id}`` key in ``event.reads`` that links
    every ``tool_result`` produced here back to its ``tool_call``.

    Raises:
        UnsupportedMessageFormatError: ``event`` carries no such key —
            either it isn't a ``tool_result``, or it predates this linkage
            (hand-built, or recorded before the fix that added it).
    """
    for key in event.reads:
        if key.startswith(_TOOL_CALL_KEY_PREFIX):
            return key[len(_TOOL_CALL_KEY_PREFIX) :]
    raise UnsupportedMessageFormatError(
        f"event {event.id!r} has no 'tool_call:<id>' read; cannot recover its tool_call_id"
    )


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
            call, a ``tool_call`` id is reused, an event id collides with
            one generated for a different message, or a tool call's
            arguments are not valid JSON.
    """
    side_effect_tools = side_effect_tools or set()

    events: list[TraceEvent] = []
    seq = 0
    known_values: dict[str, Any] = {}
    since_last_model_message: set[str] = set()
    call_id_to_tool_name: dict[str, str] = {}
    seen_event_ids: set[str] = set()

    def claim_event_id(new_id: str, message_index: int) -> None:
        if new_id in seen_event_ids:
            raise UnsupportedMessageFormatError(
                f"message {message_index}: duplicate event id {new_id!r}"
            )
        seen_event_ids.add(new_id)

    for message_index, message in enumerate(messages):
        role = message.get("role")
        if not role:
            raise UnsupportedMessageFormatError(f"message {message_index}: missing 'role'")

        if role == "system":
            event_id = f"msg_{message_index}"
            claim_event_id(event_id, message_index)
            events.append(
                TraceEvent(
                    id=event_id,
                    seq=seq,
                    type=EventType.CONSTRAINT,
                    outputs={"content": message.get("content") or ""},
                    writes=frozenset({_CONVERSATION_KEY}),
                    pinned=True,
                )
            )
            seq += 1

        elif role == "user":
            event_id = f"msg_{message_index}"
            claim_event_id(event_id, message_index)
            events.append(
                TraceEvent(
                    id=event_id,
                    seq=seq,
                    type=EventType.USER_GOAL,
                    outputs={"content": message.get("content") or ""},
                    reads=frozenset({_CONVERSATION_KEY}),
                    writes=frozenset({_USER_GOAL_KEY, _CONVERSATION_KEY}),
                )
            )
            seq += 1

        elif role == "assistant":
            tool_calls = message.get("tool_calls") or []
            if tool_calls:
                for call in tool_calls:
                    call_id = call["id"]
                    if call_id in seen_event_ids:
                        raise UnsupportedMessageFormatError(
                            f"message {message_index}: duplicate tool_call id {call_id!r}"
                        )
                    seen_event_ids.add(call_id)

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
                    reads.add(_USER_GOAL_KEY)
                    events.append(
                        TraceEvent(
                            id=call_id,
                            seq=seq,
                            type=EventType.TOOL_CALL,
                            tool_name=tool_name,
                            inputs=inputs,
                            reads=frozenset(reads),
                            writes=frozenset(
                                {f"{_TOOL_CALL_KEY_PREFIX}{call_id}", _CONVERSATION_KEY}
                            ),
                        )
                    )
                    seq += 1
            else:
                content = message.get("content")
                if not content:
                    continue
                event_id = f"msg_{message_index}"
                claim_event_id(event_id, message_index)
                events.append(
                    TraceEvent(
                        id=event_id,
                        seq=seq,
                        type=EventType.MODEL_MESSAGE,
                        outputs={"content": content},
                        reads=frozenset(since_last_model_message | {_USER_GOAL_KEY}),
                        writes=frozenset({_CONVERSATION_KEY}),
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
            event_id = f"msg_{message_index}"
            claim_event_id(event_id, message_index)
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
            is_object_result = isinstance(parsed, dict)
            outputs: dict[str, Any] = parsed if is_object_result else {"result": raw_content}
            metadata: dict[str, Any] = {} if is_object_result else {"raw_content": raw_content}

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
                for index, leaf_value in enumerate(_iter_leaf_values(outputs)):
                    if _is_matchable(leaf_value):
                        leaf_key = f"{key}#{index}"
                        writes.add(leaf_key)
                        known_values[leaf_key] = leaf_value
            since_last_model_message |= writes
            writes.add(_CONVERSATION_KEY)

            events.append(
                TraceEvent(
                    id=event_id,
                    seq=seq,
                    type=EventType.TOOL_RESULT,
                    tool_name=tool_name,
                    outputs=outputs,
                    reads=frozenset({f"{_TOOL_CALL_KEY_PREFIX}{call_id}"}),
                    writes=frozenset(writes),
                    side_effects=tool_name in side_effect_tools,
                    metadata=metadata,
                )
            )
            seq += 1

        else:
            raise UnsupportedMessageFormatError(
                f"message {message_index}: unsupported role {role!r}"
            )

    return events


def to_openai_messages(events: Sequence[TraceEvent]) -> list[dict[str, Any]]:
    """Convert a ``TraceEvent`` sequence back into OpenAI-compatible chat messages.

    The inverse of :func:`from_openai_messages`, for feeding a recorded,
    compiled, or forked trace back to a real model (see
    ``agentslice.replay``). ``events`` must already be in canonical order
    (e.g. :meth:`~agentslice.ir.graph.CausalGraph.events_in_order`, or
    :attr:`~agentslice.compiler.base.CompiledContext.events`).

    Each consecutive run of ``tool_call`` events becomes a single assistant
    message with one ``tool_calls`` entry per call — one call per message
    would also be valid, but this mirrors the parallel-call shape the
    original API response likely had, and survives a compiler pass
    dropping some (not all) calls from an original parallel group. A
    ``tool_result``'s ``tool_call_id`` is recovered via
    :func:`tool_call_id_of`.

    Raises:
        UnsupportedMessageFormatError: An event's type has no OpenAI
            message equivalent (``state_update``, ``error``,
            ``final_output``), or a ``tool_result`` predates the
            ``tool_call:{id}`` linkage :func:`tool_call_id_of` relies on.
    """
    messages: list[dict[str, Any]] = []
    pending_tool_calls: list[dict[str, Any]] = []

    def flush_tool_calls() -> None:
        if pending_tool_calls:
            messages.append(
                {"role": "assistant", "content": None, "tool_calls": list(pending_tool_calls)}
            )
            pending_tool_calls.clear()

    for event in events:
        if event.type is EventType.TOOL_CALL:
            pending_tool_calls.append(
                {
                    "id": event.id,
                    "type": "function",
                    "function": {
                        "name": event.tool_name or "",
                        "arguments": json.dumps(event.inputs or {}),
                    },
                }
            )
            continue

        flush_tool_calls()

        if event.type is EventType.CONSTRAINT:
            messages.append({"role": "system", "content": (event.outputs or {}).get("content", "")})
        elif event.type is EventType.USER_GOAL:
            messages.append({"role": "user", "content": (event.outputs or {}).get("content", "")})
        elif event.type is EventType.MODEL_MESSAGE:
            messages.append(
                {"role": "assistant", "content": (event.outputs or {}).get("content", "")}
            )
        elif event.type is EventType.TOOL_RESULT:
            raw_content = event.metadata.get("raw_content")
            content = raw_content if raw_content is not None else json.dumps(event.outputs or {})
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id_of(event),
                    "content": content,
                }
            )
        else:
            raise UnsupportedMessageFormatError(
                f"event {event.id!r}: {event.type} has no OpenAI message equivalent"
            )

    flush_tool_calls()
    return messages
