"""The contract every simulated backend environment implements."""

from __future__ import annotations

from typing import Any, Protocol

from benchmarks.errors import InvalidCallError, UnknownFunctionError


class Environment(Protocol):
    """A stateful backend a model's function calls are executed against.

    ``call`` and ``snapshot`` are the only two operations the runner and
    metrics code need: everything else (what functions exist, what state
    looks like) is environment-specific.
    """

    def call(self, name: str, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Invoke a named operation and return a JSON-serializable result.

        Raises:
            UnknownFunctionError: ``name`` isn't an operation this
                environment implements.
            InvalidCallError: ``kwargs`` don't match the operation's
                signature.
        """
        ...

    def snapshot(self) -> Any:
        """Return a deep, ``==``-comparable representation of current state."""
        ...


def safe_call(env: Environment, name: str, kwargs: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    """Invoke ``env.call`` without letting a bad model call abort the run.

    Returns ``(result, is_valid)``: ``is_valid`` is ``False`` when the call
    named an unknown function or supplied arguments that don't match its
    signature, in which case ``result`` carries an ``error`` field
    explaining why rather than the operation's real output.
    """
    try:
        return env.call(name, kwargs), True
    except (UnknownFunctionError, InvalidCallError) as exc:
        return {"error": str(exc)}, False
    except TypeError as exc:
        return {"error": f"invalid arguments for {name!r}: {exc}"}, False
