from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentsnap.adapters.anthropic import AnthropicAdapter
from agentsnap.adapters.tool import ToolAdapter
from agentsnap.core.snapshot import snapshot_path
from tests.fixtures.mock_agents import MockAnthropicClient, MockAnthropicResponse, SimpleToolAgent

import numpy as np

_DIM = 8

def _identical_embed(texts):
    v = np.ones(_DIM, dtype=float)
    v /= np.linalg.norm(v)
    return [v.copy() for _ in texts]


def _make_client():
    return AnthropicAdapter(MockAnthropicClient([MockAnthropicResponse("I'll search for that.")]))


def _search(query: str) -> str:
    return f"result_for_{query}"


def test_snapshot_run_auto_records_on_first_use(tmp_path, snapshot):
    """snapshot.run() records when no file exists (default behavior)."""
    snapshot._snapshot_dir_override = str(tmp_path / "snaps")
    snap_dir = str(tmp_path / "snaps")

    # Reach into the fixture to point it at tmp_path
    from agentsnap.pytest_plugin import SnapshotFixture
    from agentsnap.core.diff import LLMJudge
    sf = SnapshotFixture(
        snapshot_dir=snap_dir,
        semantic_threshold=0.92,
        llm_threshold=0.75,
        judge=None,
        force_record=False,
    )

    with sf.run("first_use", model="test") as s:
        client = _make_client()
        tool = ToolAdapter(_search, name="search")
        s.output = SimpleToolAgent(client, tool, "hello")

    assert snapshot_path("first_use", snap_dir).exists()


def test_force_record_overwrites_existing_snapshot(tmp_path):
    """force_record=True re-records even when a golden already exists."""
    from agentsnap.pytest_plugin import SnapshotFixture

    snap_dir = str(tmp_path / "snaps")

    # First: record a golden with output "v1"
    sf = SnapshotFixture(snap_dir, 0.92, 0.75, None, force_record=False)
    with sf.run("overwrite_test", model="test") as s:
        client = _make_client()
        tool = ToolAdapter(_search, name="search")
        s.output = "v1"
        SimpleToolAgent(client, tool, "hello")

    golden = json.loads(snapshot_path("overwrite_test", snap_dir).read_text())
    assert golden["output"] == "v1"

    # Second: force_record=True — should overwrite with "v2"
    sf2 = SnapshotFixture(snap_dir, 0.92, 0.75, None, force_record=True)
    with sf2.run("overwrite_test", model="test") as s:
        client = _make_client()
        tool = ToolAdapter(_search, name="search")
        s.output = "v2"
        SimpleToolAgent(client, tool, "hello")

    golden2 = json.loads(snapshot_path("overwrite_test", snap_dir).read_text())
    assert golden2["output"] == "v2"


def test_force_record_false_asserts_when_snapshot_exists(tmp_path):
    """force_record=False uses assert mode when snapshot exists — identical run passes."""
    from agentsnap.pytest_plugin import SnapshotFixture

    snap_dir = str(tmp_path / "snaps")

    sf = SnapshotFixture(snap_dir, 0.92, 0.75, None, force_record=False)
    with sf.run("assert_test", model="test") as s:
        client = _make_client()
        tool = ToolAdapter(_search, name="search")
        s.output = SimpleToolAgent(client, tool, "hello")

    # Second run: same inputs → should assert and pass
    sf2 = SnapshotFixture(snap_dir, 0.0, 0.0, None, force_record=False)
    with sf2.run("assert_test", model="test") as s:
        client = _make_client()
        tool = ToolAdapter(_search, name="search")
        s.output = SimpleToolAgent(client, tool, "hello")
