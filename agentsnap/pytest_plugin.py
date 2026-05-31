from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import pytest

from agentsnap.core.asserter import AgentAsserter
from agentsnap.core.diff import LLMJudge
from agentsnap.core.recorder import AgentRecorder


# -- pytest ini options -------------------------------------------------------

def pytest_addoption(parser: pytest.Parser) -> None:
    """Register [tool.agentsnap] keys as pytest ini options.

    These can be set in pyproject.toml under [tool.pytest.ini_options]:

        [tool.pytest.ini_options]
        agentsnap_judge_model      = "openai/gpt-4o-mini"
        agentsnap_judge_base_url   = "https://openrouter.ai/api/v1"
        agentsnap_semantic_threshold = "0.92"
        agentsnap_llm_threshold      = "0.75"

    The API key is NEVER stored in a file — set AGENTSNAP_JUDGE_API_KEY instead.
    """
    parser.addini("agentsnap_judge_model",        default=None,   help="LLM model slug for judge")
    parser.addini("agentsnap_judge_base_url",     default=None,   help="Base URL for judge LLM API")
    parser.addini("agentsnap_semantic_threshold", default="0.92", help="Cosine similarity threshold for final output")
    parser.addini("agentsnap_llm_threshold",      default="0.75", help="Cosine similarity threshold for intermediate LLM responses")


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


# -- Fixture ------------------------------------------------------------------

class SnapshotFixture:
    def __init__(
        self,
        snapshot_dir: str,
        semantic_threshold: float,
        llm_threshold: float,
        judge: LLMJudge | None,
    ) -> None:
        self.snapshot_dir = snapshot_dir
        self.semantic_threshold = semantic_threshold
        self.llm_threshold = llm_threshold
        self.judge = judge

    def record_agent(self, test_name: str, model: str = "unknown") -> AgentRecorder:
        """Context manager: record an agent run and write a snapshot."""
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
        """Context manager: replay an agent run and assert against the snapshot.

        Per-call values override the project-level defaults from pyproject.toml
        and env vars. Pass judge=False to explicitly disable the LLM judge even
        if AGENTSNAP_JUDGE_API_KEY is set.
        """
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
    """Provides record_agent() and assert_agent() context managers.

    Configured automatically from AGENTSNAP_JUDGE_API_KEY env var and
    [tool.pytest.ini_options] in pyproject.toml. No code changes needed
    to enable the LLM judge — just set the env var.
    """
    import os
    from agentsnap.config import load

    snapshot_dir = _find_snapshot_dir(request)

    # Load project config (pyproject.toml + env vars)
    cfg = load(Path(request.fspath).parent)

    # ini options override pyproject.toml for pytest-specific config
    semantic_threshold = float(_ini(request, "agentsnap_semantic_threshold", cfg["semantic_threshold"]))
    llm_threshold      = float(_ini(request, "agentsnap_llm_threshold",      cfg["llm_threshold"]))

    # Build judge from env if key is present
    judge: LLMJudge | None = None
    api_key = cfg.get("judge_api_key") or os.getenv("AGENTSNAP_JUDGE_API_KEY")
    if api_key:
        judge_model    = _ini(request, "agentsnap_judge_model",    cfg["judge_model"])
        judge_base_url = _ini(request, "agentsnap_judge_base_url", cfg["judge_base_url"])
        judge = LLMJudge(api_key=api_key, model=judge_model, base_url=judge_base_url)

    return SnapshotFixture(
        snapshot_dir=snapshot_dir,
        semantic_threshold=semantic_threshold,
        llm_threshold=llm_threshold,
        judge=judge,
    )
