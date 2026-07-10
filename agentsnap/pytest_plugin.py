from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import pytest

from agentsnap.core.asserter import AgentAsserter
from agentsnap.core.diff import LLMJudge
from agentsnap.core.recorder import AgentRecorder
from agentsnap.core.snapshot import snapshot_path

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
    parser.addini(
        "agentsnap_structural_tolerance",
        default=None,
        help="Max allowed Levenshtein edit distance in tool call sequence before structural check fails",
    )
    parser.addoption(
        "--agentsnap-record",
        action="store_true",
        default=False,
        help="Force re-record all agent snapshots, overwriting existing goldens.",
    )
    parser.addoption(
        "--agentsnap-instrument",
        action="store_true",
        default=False,
        help="Auto-patch all installed LLM SDKs for zero-instrumentation capture.",
    )
    parser.addini(
        "agentsnap_mode",
        default=None,
        help="Assert mode: 'live' (default, calls real APIs) or 'replay' (deterministic, replays recorded responses)",
    )
    parser.addoption(
        "--agentsnap-replay",
        action="store_true",
        default=False,
        help="Assert in replay mode: recorded LLM responses are replayed; no live API calls.",
    )
    parser.addoption(
        "--agentsnap-live",
        action="store_true",
        default=False,
        help="Force live mode, overriding config and --agentsnap-replay.",
    )


def pytest_configure(config: pytest.Config) -> None:
    config._agentsnap_results = []


def pytest_terminal_summary(terminalreporter, exitstatus, config: pytest.Config) -> None:
    results = getattr(config, "_agentsnap_results", None)
    if not results:
        return
    terminalreporter.section("agentsnap snapshots")
    for r in results:
        if r["passed"] is True:
            line = f"PASSED   {r['test_name']} ({r['mode']}) {r['summary']}"
        elif r["passed"] is False:
            line = f"FAILED   {r['test_name']} ({r['mode']}) {r['summary']}"
        else:
            line = f"RECORDED {r['test_name']} {r['summary']}"
        terminalreporter.line(line)


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

    def __init__(
        self,
        test_name: str,
        recorder: AgentRecorder,
        asserter: AgentAsserter,
        is_record: bool,
        result_sink: list | None = None,
    ) -> None:
        self._test_name = test_name
        self._recorder = recorder
        self._asserter = asserter
        self._is_record = is_record
        self._result_sink = result_sink
        self._ctx = None

    def __enter__(self) -> _AutoContext:
        if self._is_record:
            self._ctx = self._recorder.__enter__()
            print(f"\n  [agentsnap] recording '{self._test_name}'")
        else:
            self._ctx = self._asserter.__enter__()
            print(f"\n  [agentsnap] asserting '{self._test_name}'")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        result = (self._recorder if self._is_record else self._asserter).__exit__(exc_type, exc_val, exc_tb)
        if self._is_record and exc_type is None and self._result_sink is not None:
            self._result_sink.append({
                "test_name": self._test_name,
                "mode": "record",
                "passed": None,
                "summary": "recorded golden run",
            })
        return result

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

    @property
    def input(self):
        return self._asserter.input if not self._is_record else self._recorder.input_data

    @input.setter
    def input(self, value) -> None:
        if self._is_record:
            self._recorder.input_data = value
        else:
            self._asserter.input = value


# -- Fixture ------------------------------------------------------------------

class SnapshotFixture:
    def __init__(
        self,
        snapshot_dir: str,
        semantic_threshold: float,
        llm_threshold: float,
        judge: LLMJudge | None,
        force_record: bool = False,
        structural_tolerance: int = 0,
        mode: str = "live",
        result_sink: list | None = None,
    ) -> None:
        self.snapshot_dir = snapshot_dir
        self.semantic_threshold = semantic_threshold
        self.llm_threshold = llm_threshold
        self.judge = judge
        self.force_record = force_record
        self.structural_tolerance = structural_tolerance
        self.mode = mode
        self.result_sink = result_sink

    def run(
        self,
        test_name: str,
        model: str = "unknown",
        semantic_threshold: float | None = None,
        llm_threshold: float | None = None,
        ignored_fields: list[str] | None = None,
        judge: LLMJudge | None = None,
        scenario: str | None = None,
        structural_tolerance: int | None = None,
        mode: str | None = None,
        replay_tools: bool = False,
    ) -> _AutoContext:
        """Auto context manager: records if no snapshot exists, asserts if it does.

        This is the simplest way to use agentsnap — no need to think about
        record vs assert mode:

            with snapshot.run("my_agent") as s:
                s.output = my_agent(client, tool, "test input")
        """
        snap_exists = snapshot_path(test_name, self.snapshot_dir, scenario=scenario).exists()
        is_record = not snap_exists or self.force_record
        recorder = AgentRecorder(test_name, snapshot_dir=self.snapshot_dir, model=model, scenario=scenario)
        asserter = self._make_asserter(test_name, semantic_threshold, llm_threshold, ignored_fields,
                                       judge, scenario=scenario, structural_tolerance=structural_tolerance,
                                       mode=mode, replay_tools=replay_tools)
        return _AutoContext(test_name, recorder, asserter, is_record=is_record, result_sink=self.result_sink)

    def record_agent(self, test_name: str, model: str = "unknown", scenario: str | None = None) -> AgentRecorder:
        """Explicit record mode."""
        return AgentRecorder(test_name, snapshot_dir=self.snapshot_dir, model=model, scenario=scenario)

    def assert_agent(
        self,
        test_name: str,
        semantic_threshold: float | None = None,
        llm_threshold: float | None = None,
        ignored_fields: list[str] | None = None,
        embed_fn: Callable[[list[str]], list[Any]] | None = None,
        judge: LLMJudge | None = None,
        scenario: str | None = None,
        structural_tolerance: int | None = None,
        mode: str | None = None,
        replay_tools: bool = False,
    ) -> AgentAsserter:
        """Explicit assert mode. Pass judge=False to force embeddings."""
        return self._make_asserter(test_name, semantic_threshold, llm_threshold, ignored_fields,
                                   judge, embed_fn, scenario=scenario,
                                   structural_tolerance=structural_tolerance,
                                   mode=mode, replay_tools=replay_tools)

    def _make_asserter(
        self,
        test_name: str,
        semantic_threshold: float | None,
        llm_threshold: float | None,
        ignored_fields: list[str] | None,
        judge: LLMJudge | None,
        embed_fn: Callable | None = None,
        scenario: str | None = None,
        structural_tolerance: int | None = None,
        mode: str | None = None,
        replay_tools: bool = False,
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
            scenario=scenario,
            structural_tolerance=structural_tolerance if structural_tolerance is not None else self.structural_tolerance,
            mode=mode if mode is not None else self.mode,
            replay_tools=replay_tools,
            result_sink=self.result_sink,
        )


@pytest.fixture
def agentsnap_instrument():
    """Zero-instrumentation: patches all installed LLM SDK methods for this test.

    Allows raw SDK clients (anthropic.Anthropic(), openai.OpenAI(), etc.) to be
    captured by agentsnap without wrapping them in adapter classes.

    Usage in conftest.py for project-wide auto-instrumentation::

        @pytest.fixture(autouse=True)
        def _(agentsnap_instrument):
            pass

    Or per-test::

        def test_my_agent(snapshot, agentsnap_instrument):
            with snapshot.run("test") as s:
                s.output = my_agent(anthropic.Anthropic(), "query")
    """
    from agentsnap.patches import PatchSet
    with PatchSet():
        yield


@pytest.fixture
def snapshot(request: pytest.FixtureRequest):
    """Provides run(), record_agent(), and assert_agent() context managers.

    Configured automatically from env vars and [tool.agentsnap] in pyproject.toml.
    The LLM judge is enabled automatically when OPENROUTER_API_KEY (or any
    matching provider key) is found in the environment.

    Pass --agentsnap-instrument to also patch all installed LLM SDKs so raw
    clients are captured without adapter wrapping.
    """
    from agentsnap.config import load

    snapshot_dir = _find_snapshot_dir(request)
    cfg = load(Path(request.fspath).parent)

    semantic_threshold   = float(_ini(request, "agentsnap_semantic_threshold",   cfg["semantic_threshold"]))
    llm_threshold        = float(_ini(request, "agentsnap_llm_threshold",        cfg["llm_threshold"]))
    structural_tolerance = int(_ini(request, "agentsnap_structural_tolerance", cfg.get("structural_tolerance", 0)))

    judge: LLMJudge | None = None
    api_key = cfg.get("judge_api_key")
    if api_key:
        judge_model    = _ini(request, "agentsnap_judge_model",    cfg["judge_model"])
        judge_base_url = _ini(request, "agentsnap_judge_base_url", cfg["judge_base_url"])
        judge = LLMJudge(api_key=api_key, model=judge_model, base_url=judge_base_url)

    force_record = request.config.getoption("--agentsnap-record", default=False)
    instrument   = request.config.getoption("--agentsnap-instrument", default=False)

    mode = str(_ini(request, "agentsnap_mode", cfg.get("mode", "live")))
    if request.config.getoption("--agentsnap-replay", default=False):
        mode = "replay"
    if request.config.getoption("--agentsnap-live", default=False):
        mode = "live"

    fixture = SnapshotFixture(
        snapshot_dir=snapshot_dir,
        semantic_threshold=semantic_threshold,
        llm_threshold=llm_threshold,
        judge=judge,
        force_record=force_record,
        structural_tolerance=structural_tolerance,
        mode=mode,
        result_sink=getattr(request.config, "_agentsnap_results", None),
    )

    if instrument:
        from agentsnap.patches import PatchSet
        with PatchSet():
            yield fixture
    else:
        yield fixture  # yield (not return) so pytest runs fixture teardown correctly
