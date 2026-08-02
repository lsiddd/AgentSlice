"""Loading of the vendored, offline BFCL task subset.

Deliberately reads a bundled JSON Lines fixture rather than fetching from
the network: every other test and command in this project runs without
live network access (see ``httpx.MockTransport`` in ``tests/``), and the
benchmark loader should be no different. Expanding the fixture to cover
more BFCL categories or environment classes is a manual, one-time step
(see ``fixtures/NOTICE.md``), not something the loader does at run time.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from benchmarks.bfcl.schema import BFCLTask
from benchmarks.errors import MalformedTaskError

FIXTURES_DIR = Path(__file__).parent / "fixtures"
DEFAULT_TASKS_PATH = FIXTURES_DIR / "multi_turn_base_gfs.jsonl"
GORILLA_FILE_SYSTEM_TOOLS_PATH = FIXTURES_DIR / "gorilla_file_system_tools.json"


def load_tasks(path: Path = DEFAULT_TASKS_PATH) -> list[BFCLTask]:
    """Load every task from a JSON Lines fixture, in file order.

    Raises:
        MalformedTaskError: A line isn't valid JSON or doesn't match
            :class:`~benchmarks.bfcl.schema.BFCLTask`.
    """
    tasks: list[BFCLTask] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise MalformedTaskError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
        try:
            tasks.append(BFCLTask.model_validate(payload))
        except ValueError as exc:
            raise MalformedTaskError(f"{path}:{line_number}: {exc}") from exc
    return tasks


def load_gorilla_file_system_tool_catalog(
    path: Path = GORILLA_FILE_SYSTEM_TOOLS_PATH,
) -> list[dict[str, Any]]:
    """Load the OpenAI-shaped tool definitions for the GorillaFileSystem environment."""
    payload = json.loads(path.read_text())
    if not isinstance(payload, list):
        raise MalformedTaskError(f"{path}: expected a JSON array of tool definitions")
    return list(payload)
