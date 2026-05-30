from __future__ import annotations

import json

import pytest

from agenttest.core.snapshot import list_snapshots, read_snapshot, write_snapshot
from agenttest.exceptions import SnapshotNotFoundError

_TRACE = [
    {
        "step": 0,
        "type": "llm_call",
        "messages": [{"role": "user", "content": "hello"}],
        "response": "hi there",
        "tokens": 30,
    },
    {
        "step": 1,
        "type": "tool_call",
        "name": "search",
        "args": {"query": "foo"},
        "result": "bar",
    },
]


def test_round_trip(tmp_path):
    snapshot_dir = str(tmp_path / "snaps")
    write_snapshot("t1", snapshot_dir, "claude-3", {"input": "hello"}, _TRACE, "hi there")
    loaded = read_snapshot("t1", snapshot_dir)
    assert loaded["version"] == "1.0"
    assert loaded["model"] == "claude-3"
    assert loaded["output"] == "hi there"
    assert loaded["trace"] == _TRACE
    assert loaded["input"] == {"input": "hello"}


def test_not_found_raises(tmp_path):
    with pytest.raises(SnapshotNotFoundError):
        read_snapshot("nonexistent", str(tmp_path))


def test_json_keys_are_sorted(tmp_path):
    snapshot_dir = str(tmp_path / "snaps")
    write_snapshot("sorted", snapshot_dir, "m", {}, _TRACE, "out")
    raw = (tmp_path / "snaps" / "sorted.json").read_text(encoding="utf-8")
    data = json.loads(raw)
    top_keys = list(data.keys())
    assert top_keys == sorted(top_keys), "Top-level keys must be sorted"


def test_list_snapshots(tmp_path):
    snapshot_dir = str(tmp_path / "snaps")
    assert list_snapshots(snapshot_dir) == []
    write_snapshot("a", snapshot_dir, "m", {}, [], "")
    write_snapshot("b", snapshot_dir, "m", {}, [], "")
    names = [p.stem for p in list_snapshots(snapshot_dir)]
    assert "a" in names and "b" in names


def test_overwrite_snapshot(tmp_path):
    snapshot_dir = str(tmp_path / "snaps")
    write_snapshot("ow", snapshot_dir, "m1", {}, [], "first")
    write_snapshot("ow", snapshot_dir, "m2", {}, [], "second")
    loaded = read_snapshot("ow", snapshot_dir)
    assert loaded["output"] == "second"
    assert loaded["model"] == "m2"
