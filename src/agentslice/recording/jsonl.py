"""Reading and writing traces as JSON Lines, one event per line."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from types import TracebackType

from pydantic import ValidationError

from agentslice.errors import TraceFormatError, TraceValidationError
from agentslice.ir.events import TraceEvent


class TraceWriter:
    """Appends :class:`TraceEvent` objects to a JSON Lines file.

    Used as a context manager so the underlying file handle is always closed::

        with TraceWriter(path) as writer:
            writer.write(event)
    """

    def __init__(self, path: str | Path, *, append: bool = False) -> None:
        self._path = Path(path)
        self._file = self._path.open("a" if append else "w", encoding="utf-8")

    def write(self, event: TraceEvent) -> None:
        self._file.write(event.model_dump_json())
        self._file.write("\n")

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> TraceWriter:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


class TraceReader:
    """Streams :class:`TraceEvent` objects from a JSON Lines file.

    Iterating raises :class:`~agentslice.errors.TraceFormatError` for a line
    that isn't valid JSON, and :class:`~agentslice.errors.TraceValidationError`
    for a line that is valid JSON but doesn't satisfy the ``TraceEvent``
    schema. Both errors include the file path and the 1-based line number.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)

    def __iter__(self) -> Iterator[TraceEvent]:
        with self._path.open(encoding="utf-8") as file:
            for lineno, raw_line in enumerate(file, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise TraceFormatError(f"{self._path}:{lineno}: invalid JSON: {exc}") from exc
                try:
                    yield TraceEvent.model_validate(data)
                except ValidationError as exc:
                    raise TraceValidationError(
                        f"{self._path}:{lineno}: invalid trace event: {exc}"
                    ) from exc

    def read_all(self) -> list[TraceEvent]:
        return list(self)
