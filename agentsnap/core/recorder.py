from __future__ import annotations

import threading
from contextvars import ContextVar
from typing import Any

from agentsnap.core.snapshot import input_sha8, write_snapshot

DEFAULT_SNAPSHOT_DIR = "__agent_snapshots__"

_accumulator_var: ContextVar["TraceAccumulator | None"] = ContextVar(
    "_accumulator_var", default=None
)


class TraceAccumulator:
    def __init__(self, model: str = "unknown", replay=None) -> None:
        self.model = model
        self.replay = replay  # ReplaySession | None — set for replay-mode asserts
        self._trace: list[dict] = []
        self._step = 0
        self._lock = threading.Lock()
        self._streams: list = []

    def push(self, event: dict) -> None:
        with self._lock:
            recorded = dict(event)
            recorded["step"] = self._step
            self._step += 1
            self._trace.append(recorded)

    def register_stream(self, stream) -> None:
        with self._lock:
            self._streams.append(stream)

    def finalize_streams(self) -> None:
        # close() also records (idempotent) and releases the underlying SDK
        # connection; isolate failures so one bad stream can't skip the rest.
        with self._lock:
            streams = list(self._streams)
        for stream in streams:
            try:
                stream.close()
            except Exception:
                try:
                    stream._record()
                except Exception:
                    pass

    @property
    def trace(self) -> list[dict]:
        with self._lock:
            return list(self._trace)

    @staticmethod
    def current() -> "TraceAccumulator | None":
        return _accumulator_var.get()


class AgentRecorder:
    """Context manager that records an agent run and writes a golden snapshot.

    Intercepts every LLM/tool call made through an active adapter or
    `PatchSet` while the context is open, then writes
    `{snapshot_dir}/{test_name}.json` on clean exit (nothing is written if
    the block raises). Supports both sync (`with`) and async (`async with`)
    usage.

    Usage::

        with PatchSet():
            with AgentRecorder("my_agent", model="claude-haiku-4-5") as rec:
                rec.input_data = {"query": "hello"}   # optional metadata
                result = my_agent("hello")
                rec.output = result
        # Writes __agent_snapshots__/my_agent.json
    """

    def __init__(
        self,
        test_name: str,
        snapshot_dir: str = DEFAULT_SNAPSHOT_DIR,
        model: str = "unknown",
        scenario: str | None = None,
        result_sink: list | None = None,
    ) -> None:
        self.test_name = test_name
        self.snapshot_dir = snapshot_dir
        self.model = model
        self.scenario = scenario
        self.result_sink = result_sink
        self.input_data: Any = None
        self.output: str = ""
        self._accumulator: TraceAccumulator | None = None
        self._token = None

    def _resolved_scenario(self) -> str | None:
        if self.scenario is not None:
            return self.scenario
        if self.input_data is not None:
            return input_sha8(self.input_data)
        return None

    def __enter__(self) -> "AgentRecorder":
        self._accumulator = TraceAccumulator(model=self.model)
        self._token = _accumulator_var.set(self._accumulator)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        _accumulator_var.reset(self._token)
        if self._accumulator is not None:
            self._accumulator.finalize_streams()
        if exc_type is None:
            assert self._accumulator is not None
            write_snapshot(
                self.test_name,
                self.snapshot_dir,
                self._accumulator.model,
                self.input_data,
                self._accumulator.trace,
                self.output,
                scenario=self._resolved_scenario(),
            )
            if self.result_sink is not None:
                self.result_sink.append({
                    "test_name": self.test_name,
                    "mode": "record",
                    "passed": None,
                    "summary": "recorded golden run",
                })
        return False

    async def __aenter__(self) -> "AgentRecorder":
        return self.__enter__()

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> bool:
        return self.__exit__(exc_type, exc_val, exc_tb)

    @property
    def accumulator(self) -> TraceAccumulator:
        assert self._accumulator is not None
        return self._accumulator
