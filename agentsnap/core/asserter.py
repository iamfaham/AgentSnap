from __future__ import annotations

from typing import Any, Callable

from agentsnap.core.diff import LLMJudge, compute_diff
from agentsnap.core.recorder import DEFAULT_SNAPSHOT_DIR, TraceAccumulator, _accumulator_var
from agentsnap.core.snapshot import read_snapshot, write_last_run
from agentsnap.exceptions import AgentRegressionError


class AgentAsserter:
    """Context manager that replays an agent run and compares against the snapshot."""

    def __init__(
        self,
        test_name: str,
        snapshot_dir: str = DEFAULT_SNAPSHOT_DIR,
        semantic_threshold: float = 0.92,
        llm_threshold: float = 0.75,
        ignored_fields: list[str] | None = None,
        embed_fn: Callable[[list[str]], list[Any]] | None = None,
        judge: LLMJudge | None = None,
    ) -> None:
        self.test_name = test_name
        self.snapshot_dir = snapshot_dir
        self.semantic_threshold = semantic_threshold
        self.llm_threshold = llm_threshold
        self.ignored_fields = ignored_fields or []
        self.embed_fn = embed_fn
        self.judge = judge
        self.output: str = ""
        self._accumulator: TraceAccumulator | None = None
        self._snapshot: dict = {}
        self._token = None

    def __enter__(self) -> AgentAsserter:
        self._snapshot = read_snapshot(self.test_name, self.snapshot_dir)
        self._accumulator = TraceAccumulator(
            model=self._snapshot.get("model", "unknown")
        )
        self._token = _accumulator_var.set(self._accumulator)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        _accumulator_var.reset(self._token)
        if exc_type is not None:
            return False

        assert self._accumulator is not None
        new_trace = self._accumulator.trace

        write_last_run(
            self.test_name,
            self.snapshot_dir,
            self._accumulator.model,
            self._snapshot.get("input"),
            new_trace,
            self.output,
        )

        report = compute_diff(
            self._snapshot,
            new_trace,
            self.output,
            semantic_threshold=self.semantic_threshold,
            llm_threshold=self.llm_threshold,
            ignored_fields=self.ignored_fields,
            embed_fn=self.embed_fn,
            judge=self.judge,
        )
        if not report.passed:
            raise AgentRegressionError(
                f"Agent regression detected in '{self.test_name}'",
                report,
            )
        return False
