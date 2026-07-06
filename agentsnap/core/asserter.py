from __future__ import annotations

from typing import Any, Callable

from agentsnap.core.diff import DiffConfig, LLMJudge, compute_diff
from agentsnap.core.recorder import DEFAULT_SNAPSHOT_DIR, TraceAccumulator, _accumulator_var
from agentsnap.core.snapshot import input_sha8, read_snapshot, write_last_run, write_snapshot
from agentsnap.exceptions import AgentRegressionError, SnapshotNotFoundError


class AgentAsserter:
    """Context manager that replays an agent run and compares against the snapshot.

    On first use (no snapshot file), automatically records the run as the golden
    instead of raising SnapshotNotFoundError. Subsequent runs assert against it.

    Snapshot read is deferred to __exit__ so that self.input (set inside the
    with block) can drive auto-hash scenario resolution before the file is looked up.
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
        scenario: str | None = None,
        structural_tolerance: int = 0,
    ) -> None:
        self.test_name = test_name
        self.snapshot_dir = snapshot_dir
        self.semantic_threshold = semantic_threshold
        self.llm_threshold = llm_threshold
        self.ignored_fields = ignored_fields or []
        self.embed_fn = embed_fn
        self.judge = judge
        self.scenario = scenario
        self.structural_tolerance = structural_tolerance
        self.output: str = ""
        self.input: Any = None
        self._accumulator: TraceAccumulator | None = None
        self._token = None

    def _resolved_scenario(self) -> str | None:
        if self.scenario is not None:
            return self.scenario
        if self.input is not None:
            return input_sha8(self.input)
        return None

    def __enter__(self) -> "AgentAsserter":
        self._accumulator = TraceAccumulator(model="unknown")
        self._token = _accumulator_var.set(self._accumulator)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        _accumulator_var.reset(self._token)
        if exc_type is not None:
            return False

        assert self._accumulator is not None
        new_trace = self._accumulator.trace
        scenario = self._resolved_scenario()

        try:
            snapshot = read_snapshot(self.test_name, self.snapshot_dir, scenario=scenario)
            record_mode = False
        except SnapshotNotFoundError:
            snapshot = {}
            record_mode = True

        if record_mode:
            print(f"\n  [agentsnap] no snapshot for '{self.test_name}' - recording golden run")
            write_snapshot(
                self.test_name,
                self.snapshot_dir,
                "unknown",
                None,
                new_trace,
                self.output,
                scenario=scenario,
            )
            return False

        # Input binding warning
        recorded_input = snapshot.get("input")
        if recorded_input is not None and self.input is not None:
            recorded_sha8 = input_sha8(recorded_input)
            current_sha8 = input_sha8(self.input)
            if recorded_sha8 != current_sha8:
                print(f"\n  [agentsnap] WARNING: input changed since snapshot was recorded for '{self.test_name}'")
                print(f"    recorded: sha8={recorded_sha8}")
                print(f"    current:  sha8={current_sha8}")
                print("    Comparison may be against the wrong baseline. Delete the snapshot file and re-record.")

        write_last_run(
            self.test_name,
            self.snapshot_dir,
            self._accumulator.model,
            snapshot.get("input"),
            new_trace,
            self.output,
            scenario=scenario,
        )

        config = DiffConfig(
            semantic_threshold=self.semantic_threshold,
            llm_threshold=self.llm_threshold,
            ignored_fields=self.ignored_fields,
            judge=self.judge,
            structural_tolerance=self.structural_tolerance,
        )
        report = compute_diff(
            snapshot,
            new_trace,
            self.output,
            config=config,
            embed_fn=self.embed_fn,
        )
        if not report.passed:
            raise AgentRegressionError(
                self.test_name,
                report,
                snapshot,
                new_trace,
                self.output,
            )

        scores = report.semantic_scores or {}
        parts = ["structural: ok"] if not report.structural_diff else ["structural: mismatch"]
        for step, score in scores.items():
            parts.append(f"{step}: {int(score * 100)}%")
        print(f"  [agentsnap] '{self.test_name}' PASSED | {' | '.join(parts)}")
        return False

    async def __aenter__(self) -> "AgentAsserter":
        return self.__enter__()

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> bool:
        return self.__exit__(exc_type, exc_val, exc_tb)
