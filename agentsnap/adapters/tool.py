from __future__ import annotations

from typing import Any, Callable

from agentsnap.core.recorder import TraceAccumulator


class ToolAdapter:
    """Wraps any callable to intercept and record tool calls."""

    def __init__(self, func: Callable, name: str | None = None) -> None:
        self._func = func
        self._name = name or getattr(func, "__name__", "unknown_tool")

    def __call__(self, **kwargs: Any) -> Any:
        acc = TraceAccumulator.current()
        if acc is None:
            return self._func(**kwargs)

        result = self._func(**kwargs)
        acc.push(
            {
                "type": "tool_call",
                "name": self._name,
                "args": kwargs,
                "result": str(result),
            }
        )
        return result

    @property
    def __name__(self) -> str:
        return self._name

    def __repr__(self) -> str:
        return f"ToolAdapter({self._name!r})"
