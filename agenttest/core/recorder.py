from __future__ import annotations

import threading
from contextvars import ContextVar
from typing import Any

from agenttest.core.snapshot import write_snapshot

DEFAULT_SNAPSHOT_DIR = "__agent_snapshots__"

_accumulator_var: ContextVar[TraceAccumulator | None] = ContextVar(
    "_accumulator_var", default=None
)


class TraceAccumulator:
    def __init__(self, model: str = "unknown") -> None:
        self.model = model
        self._trace: list[dict] = []
        self._step = 0
        self._lock = threading.Lock()

    def push(self, event: dict) -> None:
        with self._lock:
            recorded = dict(event)
            recorded["step"] = self._step
            self._step += 1
            self._trace.append(recorded)

    @property
    def trace(self) -> list[dict]:
        with self._lock:
            return list(self._trace)

    @staticmethod
    def current() -> TraceAccumulator | None:
        return _accumulator_var.get()


class AgentRecorder:
    """Context manager that wraps an agent run and writes a snapshot on exit."""

    def __init__(
        self,
        test_name: str,
        snapshot_dir: str = DEFAULT_SNAPSHOT_DIR,
        model: str = "unknown",
    ) -> None:
        self.test_name = test_name
        self.snapshot_dir = snapshot_dir
        self.model = model
        self.input_data: Any = None
        self.output: str = ""
        self._accumulator: TraceAccumulator | None = None
        self._token = None

    def __enter__(self) -> AgentRecorder:
        self._accumulator = TraceAccumulator(model=self.model)
        self._token = _accumulator_var.set(self._accumulator)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        _accumulator_var.reset(self._token)
        if exc_type is None:
            assert self._accumulator is not None
            write_snapshot(
                self.test_name,
                self.snapshot_dir,
                self._accumulator.model,
                self.input_data,
                self._accumulator.trace,
                self.output,
            )
        return False

    @property
    def accumulator(self) -> TraceAccumulator:
        assert self._accumulator is not None
        return self._accumulator
