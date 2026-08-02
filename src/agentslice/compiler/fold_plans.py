"""Immutable plans produced before destructive compiler transformations."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal

_SYNTHETIC_ID_PREFIX = "fold_"


@dataclass(frozen=True)
class EvidenceRef:
    """A verified reference to one exact JSON value in a source event."""

    event_id: str
    json_pointer: str
    operator: Literal["=="]
    value_json: str

    def value(self) -> Any:
        return json.loads(self.value_json)


@dataclass(frozen=True)
class FoldPlan:
    """An immutable, fully validated failed-hypothesis transformation plan."""

    fold_id: str
    annotation_event_id: str
    hypothesis_text: str
    hypothesis_event_id: str
    evidence: tuple[EvidenceRef, ...]
    remove_event_ids: tuple[str, ...]
    conclusion_event_ids: tuple[str, ...]
    annotator_kind: str
    annotator_name: str
    annotator_version: str
    external_reads: frozenset[str]
    preserve_conversation_write: bool

    @property
    def synthetic_event_id(self) -> str:
        return f"{_SYNTHETIC_ID_PREFIX}{self.fold_id}"
