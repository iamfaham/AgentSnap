from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import pytest

from agentsnap.core.asserter import AgentAsserter
from agentsnap.core.recorder import AgentRecorder


def _find_snapshot_dir(request) -> str:
    """Walk up from the test file to find the nearest conftest.py and use that dir."""
    start = Path(request.fspath).parent
    for candidate in [start, *start.parents]:
        if (candidate / "conftest.py").exists():
            return str(candidate / "__agent_snapshots__")
    return "__agent_snapshots__"


class SnapshotFixture:
    def __init__(self, snapshot_dir: str) -> None:
        self.snapshot_dir = snapshot_dir

    def record_agent(
        self,
        test_name: str,
        model: str = "unknown",
    ) -> AgentRecorder:
        """Context manager: record an agent run and write a snapshot."""
        return AgentRecorder(test_name, snapshot_dir=self.snapshot_dir, model=model)

    def assert_agent(
        self,
        test_name: str,
        semantic_threshold: float = 0.92,
        ignored_fields: list[str] | None = None,
        embed_fn: Callable[[list[str]], list[Any]] | None = None,
    ) -> AgentAsserter:
        """Context manager: replay an agent run and assert against the snapshot."""
        return AgentAsserter(
            test_name,
            snapshot_dir=self.snapshot_dir,
            semantic_threshold=semantic_threshold,
            ignored_fields=ignored_fields,
            embed_fn=embed_fn,
        )


@pytest.fixture
def snapshot(request) -> SnapshotFixture:
    """Provides record_agent() and assert_agent() context managers."""
    return SnapshotFixture(_find_snapshot_dir(request))
