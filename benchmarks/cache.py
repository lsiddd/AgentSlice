"""A response cache for real model calls, keyed by the exact request payload.

BFCL multi-turn tasks are deterministic scripts: replaying the same
(policy, model, turn) combination twice sends byte-identical requests, so
caching by request hash makes an interrupted or re-run benchmark free the
second time — the explicit "aggressive caching" requirement from
``NEXT_STEPS.md``'s Marco 10, since this is the first part of the roadmap
that spends real money.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


class ResponseCache:
    """One JSON file per cached request, under `directory`."""

    def __init__(self, directory: Path) -> None:
        self._directory = directory
        self._directory.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def key_for(
        model: str, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None
    ) -> str:
        payload = json.dumps(
            {"model": model, "messages": messages, "tools": tools},
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def get(self, key: str) -> dict[str, Any] | None:
        path = self._directory / f"{key}.json"
        if not path.exists():
            return None
        result: dict[str, Any] = json.loads(path.read_text())
        return result

    def set(self, key: str, value: dict[str, Any]) -> None:
        path = self._directory / f"{key}.json"
        path.write_text(json.dumps(value, sort_keys=True))
