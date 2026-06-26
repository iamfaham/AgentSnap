from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import pytest

from agentsnap.core.asserter import AgentAsserter
from agentsnap.core.diff import LLMJudge
from agentsnap.core.recorder import AgentRecorder
from agentsnap.core.snapshot import snapshot_path
from agentsnap.exceptions import SnapshotNotFoundError


# -- pytest ini options -------------------------------------------------------

def pytest_addoption(parser: pytest.Parser) -> None:
    """Register [tool.agentsnap] keys as pytest ini options.

    Set in pyproject.toml under [tool.pytest.ini_options]:

        [tool.pytest.ini_options]
        agentsnap_judge_model        = "openai/gpt-4o-mini"
        agentsnap_judge_base_url     = "https://openrouter.ai/api/v1"
        agentsnap_semantic_threshold = "0.92"
        agentsnap_llm_threshold      = "0.75"

    The API key is NEVER stored in a file — set AGENTSNAP_JUDGE_API_KEY
    (or your provider key, e.g. OPENROUTER_API_KEY) as an env var instead.
    """
    parser.addini("agentsnap_judge_model",        default=None,   help="LLM model slug for judge")
    parser.addini("agentsnap_judge_base_url",     default=None,   help="Base URL for judge LLM API")
    parser.addini("agentsnap_semantic_threshold", default="0.92", help="Threshold for final output similarity")
    parser.addini("agentsnap_llm_threshold",      default="0.75", help="Threshold for intermediate LLM response similarity")
    parser.addoption(
        "--agentsnap-record",
        action="store_true",
        default=False,
        help="Force re-record all agent snapshots, overwriting existing goldens.",
    )


def _ini(request: pytest.FixtureRequest, key: str, fallback: Any) -> Any:
    try:
        val = request.config.getini(key)
        return val if val not in (None, "") else fallback
    except ValueError:
        return fallback


# -- Snapshot directory discovery ---------------------------------------------

def _find_snapshot_dir(request: pytest.FixtureRequest) -> str:
    start = Path(request.fspath).parent
    for candidate in [start, *start.parents]:
        if (candidate / "conftest.py").exists():
            return str(candidate / "__agent_snapshots__")
    return "__agent_snapshots__"


# -- Auto context manager: record if no snapshot, assert if snapshot exists --

class _AutoContext:
    """Returned by snapshot.run(). Records on first call, asserts on subsequent calls."""

    def __init__(self, test_name: str, recorder: AgentRecorder, asserter: AgentAsserter, is_record: bool) -> None:
        self._test_name = test_name
        self._recorder = recorder
        self._asserter = asserter
        self._is_record = is_record
        self._ctx = None

    def __enter__(self) -> _AutoContext:
        if self._is_record:
            self._ctx = self._recorder.__enter__()
            print(f"\n  [agentsnap] recording '{self._test_name}'")
        else:
            self._ctx = self._asserter.__enter__()
            if not self._asserter._record_mode:
                print(f"\n  [agentsnap] asserting '{self._test_name}'")
            # if _record_mode, AgentAsserter.__enter__ already printed the recording message
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return (self._recorder if self._is_record else self._asserter).__exit__(exc_type, exc_val, exc_tb)

    @property
    def output(self) -> str:
        return self._recorder.output if self._is_record else self._asserter.output

    @output.setter
    def output(self, value: str) -> None:
        if self._is_record:
            self._recorder.output = value
        else:
            self._asserter.output = value

    @property
    def input_data(self):
        return self._recorder.input_data if self._is_record else None

    @input_data.setter
    def input_data(self, value) -> None:
        if self._is_record:
            self._recorder.input_data = value


# -- Fixture ------------------------------------------------------------------

class SnapshotFixture:
    def __init__(
        self,
        snapshot_dir: str,
        semantic_threshold: float,
        llm_threshold: float,
        judge: LLMJudge | None,
        force_record: bool = False,
    ) -> None:
        self.snapshot_dir = snapshot_dir
        self.semantic_threshold = semantic_threshold
        self.llm_threshold = llm_threshold
        self.judge = judge
        self.force_record = force_record

    def run(
        self,
        test_name: str,
        model: str = "unknown",
        semantic_threshold: float | None = None,
        llm_threshold: float | None = None,
        ignored_fields: list[str] | None = None,
        judge: LLMJudge | None = None,
    ) -> _AutoContext:
        """Auto context manager: records if no snapshot exists, asserts if it does.

        This is the simplest way to use agentsnap — no need to think about
        record vs assert mode:

            with snapshot.run("my_agent") as s:
                s.output = my_agent(client, tool, "test input")
        """
        snap_exists = snapshot_path(test_name, self.snapshot_dir).exists()
        is_record = not snap_exists or self.force_record
        recorder = AgentRecorder(test_name, snapshot_dir=self.snapshot_dir, model=model)
        asserter = self._make_asserter(test_name, semantic_threshold, llm_threshold, ignored_fields, judge)
        return _AutoContext(test_name, recorder, asserter, is_record=is_record)

    def record_agent(self, test_name: str, model: str = "unknown") -> AgentRecorder:
        """Explicit record mode."""
        return AgentRecorder(test_name, snapshot_dir=self.snapshot_dir, model=model)

    def assert_agent(
        self,
        test_name: str,
        semantic_threshold: float | None = None,
        llm_threshold: float | None = None,
        ignored_fields: list[str] | None = None,
        embed_fn: Callable[[list[str]], list[Any]] | None = None,
        judge: LLMJudge | None = None,
    ) -> AgentAsserter:
        """Explicit assert mode. Pass judge=False to force embeddings."""
        return self._make_asserter(test_name, semantic_threshold, llm_threshold, ignored_fields, judge, embed_fn)

    def _make_asserter(
        self,
        test_name: str,
        semantic_threshold: float | None,
        llm_threshold: float | None,
        ignored_fields: list[str] | None,
        judge: LLMJudge | None,
        embed_fn: Callable | None = None,
    ) -> AgentAsserter:
        effective_judge = judge if judge is not None else self.judge
        if judge is False:
            effective_judge = None
        return AgentAsserter(
            test_name,
            snapshot_dir=self.snapshot_dir,
            semantic_threshold=semantic_threshold if semantic_threshold is not None else self.semantic_threshold,
            llm_threshold=llm_threshold if llm_threshold is not None else self.llm_threshold,
            ignored_fields=ignored_fields,
            embed_fn=embed_fn,
            judge=effective_judge,
        )


@pytest.fixture
def snapshot(request: pytest.FixtureRequest) -> SnapshotFixture:
    """Provides run(), record_agent(), and assert_agent() context managers.

    Configured automatically from env vars and [tool.agentsnap] in pyproject.toml.
    The LLM judge is enabled automatically when OPENROUTER_API_KEY (or any
    matching provider key) is found in the environment.
    """
    import os
    from agentsnap.config import load

    snapshot_dir = _find_snapshot_dir(request)
    cfg = load(Path(request.fspath).parent)

    semantic_threshold = float(_ini(request, "agentsnap_semantic_threshold", cfg["semantic_threshold"]))
    llm_threshold      = float(_ini(request, "agentsnap_llm_threshold",      cfg["llm_threshold"]))

    judge: LLMJudge | None = None
    api_key = cfg.get("judge_api_key")
    if api_key:
        judge_model    = _ini(request, "agentsnap_judge_model",    cfg["judge_model"])
        judge_base_url = _ini(request, "agentsnap_judge_base_url", cfg["judge_base_url"])
        judge = LLMJudge(api_key=api_key, model=judge_model, base_url=judge_base_url)

    force_record = request.config.getoption("--agentsnap-record", default=False)
    return SnapshotFixture(
        snapshot_dir=snapshot_dir,
        semantic_threshold=semantic_threshold,
        llm_threshold=llm_threshold,
        judge=judge,
        force_record=force_record,
    )
