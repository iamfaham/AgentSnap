from __future__ import annotations

from typing import Any, Callable

from agentsnap.core.diff import DiffConfig, LLMJudge, compute_diff
from agentsnap.core.recorder import DEFAULT_SNAPSHOT_DIR, TraceAccumulator, _accumulator_var
from agentsnap.core.snapshot import read_snapshot, write_last_run, write_snapshot
from agentsnap.exceptions import AgentRegressionError, SnapshotNotFoundError


class AgentAsserter:
    """Context manager that replays an agent run and compares against the snapshot.

    On first use (no snapshot file), automatically records the run as the golden
    instead of raising SnapshotNotFoundError. Subsequent runs assert against it.
    """

    def __init__(
        self,
        test_name: str,
        snapshot_dir: str = DEFAULT_SNAPSHOT_DIR,
        semantic_threshold: float = 0.92,
        llm_threshold: float | None = None,
        ignored_fields: list[str] | None = None,
        embed_fn: Callable[[list[str]], list[Any]] | None = None,
        judge: LLMJudge | None = None,
    ) -> None:
        self.test_name = test_name
        self.snapshot_dir = snapshot_dir
        self.semantic_threshold = semantic_threshold
        self.llm_threshold = llm_threshold  # None is fine; DiffConfig resolves at call time
        self.ignored_fields = ignored_fields or []
        self.embed_fn = embed_fn
        self.judge = judge
        self.output: str = ""
        self._accumulator: TraceAccumulator | None = None
        self._snapshot: dict = {}
        self._token = None
        self._record_mode: bool = False

    def __enter__(self) -> AgentAsserter:
        try:
            self._snapshot = read_snapshot(self.test_name, self.snapshot_dir)
            self._record_mode = False
        except SnapshotNotFoundError:
            self._snapshot = {}
            self._record_mode = True
            print(f"\n  [agentsnap] no snapshot for '{self.test_name}' - recording golden run")
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

        if self._record_mode:
            write_snapshot(
                self.test_name,
                self.snapshot_dir,
                self._accumulator.model,
                None,
                new_trace,
                self.output,
            )
            return False

        write_last_run(
            self.test_name,
            self.snapshot_dir,
            self._accumulator.model,
            self._snapshot.get("input"),
            new_trace,
            self.output,
        )

        config = DiffConfig(
            semantic_threshold=self.semantic_threshold,
            llm_threshold=self.llm_threshold,
            ignored_fields=self.ignored_fields,
            judge=self.judge,
        )
        report = compute_diff(
            self._snapshot,
            new_trace,
            self.output,
            config=config,
            embed_fn=self.embed_fn,
        )
        if not report.passed:
            raise AgentRegressionError(
                self.test_name,
                report,
                self._snapshot,
                new_trace,
                self.output,
            )

        scores = report.semantic_scores or {}
        parts = ["structural: ok"] if not report.structural_diff else [f"structural: mismatch"]
        for step, score in scores.items():
            parts.append(f"{step}: {int(score * 100)}%")
        print(f"  [agentsnap] '{self.test_name}' PASSED | {' | '.join(parts)}")
        return False

    async def __aenter__(self) -> "AgentAsserter":
        return self.__enter__()

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> bool:
        return self.__exit__(exc_type, exc_val, exc_tb)
