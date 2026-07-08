from __future__ import annotations

import json

import pytest

from pathlib import Path

from agentsnap.core.snapshot import (
    input_sha8, last_run_path, list_snapshots,
    read_snapshot, snapshot_path, write_snapshot,
)
from agentsnap.exceptions import SnapshotNotFoundError

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
    from agentsnap.core.snapshot import SNAPSHOT_VERSION
    snapshot_dir = str(tmp_path / "snaps")
    write_snapshot("t1", snapshot_dir, "claude-3", {"input": "hello"}, _TRACE, "hi there")
    loaded = read_snapshot("t1", snapshot_dir)
    assert loaded["version"] == SNAPSHOT_VERSION
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


def test_snapshot_path_no_scenario():
    p = snapshot_path("my_test", "__agent_snapshots__")
    assert p == Path("__agent_snapshots__/my_test.json")


def test_snapshot_path_with_scenario():
    p = snapshot_path("my_test", "__agent_snapshots__", scenario="short_doc")
    assert p == Path("__agent_snapshots__/my_test__short_doc.json")


def test_last_run_path_with_scenario():
    p = last_run_path("my_test", "__agent_snapshots__", scenario="abc12345")
    assert p == Path("__agent_snapshots__/.last_run/my_test__abc12345.json")


def test_snapshot_path_sanitizes_invalid_chars():
    """Scenario strings with path-unsafe chars must be sanitized, not crash."""
    p = snapshot_path("my_test", "__agent_snapshots__", scenario="sub_suite/run_1")
    assert p == Path("__agent_snapshots__/my_test__sub_suite_run_1.json")


def test_snapshot_path_sanitizes_colon():
    p = snapshot_path("my_test", "__agent_snapshots__", scenario="test:first")
    assert p == Path("__agent_snapshots__/my_test__test_first.json")


def test_sanitization_preserves_valid_chars():
    """Alphanumeric, dash, and underscore must pass through unchanged."""
    p = snapshot_path("t", "__agent_snapshots__", scenario="run-1_v2abc")
    assert p == Path("__agent_snapshots__/t__run-1_v2abc.json")


def test_input_sha8_is_deterministic():
    val = {"query": "what is the capital of France?", "model": "gpt-4"}
    assert input_sha8(val) == input_sha8(val)


def test_input_sha8_length():
    assert len(input_sha8("any input")) == 8


def test_input_sha8_differs_for_different_inputs():
    assert input_sha8("input A") != input_sha8("input B")


def test_input_sha8_key_order_independent():
    a = {"x": 1, "y": 2}
    b = {"y": 2, "x": 1}
    assert input_sha8(a) == input_sha8(b)


def test_snapshot_roundtrip_with_scenario(tmp_path):
    snap_dir = str(tmp_path / "snaps")
    write_snapshot("t1", snap_dir, "m", {}, _TRACE, "out", scenario="s1")
    loaded = read_snapshot("t1", snap_dir, scenario="s1")
    assert loaded["output"] == "out"


def test_read_snapshot_wrong_scenario_raises(tmp_path):
    from agentsnap.exceptions import SnapshotNotFoundError
    snap_dir = str(tmp_path / "snaps")
    write_snapshot("t1", snap_dir, "m", {}, _TRACE, "out", scenario="s1")
    with pytest.raises(SnapshotNotFoundError):
        read_snapshot("t1", snap_dir, scenario="s2")


def test_input_sha8_non_serializable_does_not_raise():
    """Verifies Issue 7 fix: default=str handles types json.dumps can't serialize natively.

    dataclasses, datetimes, numpy arrays, and custom objects must not crash input_sha8.
    """
    from datetime import datetime

    class _CustomObj:
        def __str__(self):
            return "custom"

    # datetime is not JSON-serializable without default=str
    result_dt = input_sha8({"ts": datetime(2024, 1, 1), "value": 42})
    assert len(result_dt) == 8

    # Custom object falls back to str()
    result_obj = input_sha8({"obj": _CustomObj()})
    assert len(result_obj) == 8

    # Determinism: same custom repr → same hash
    assert input_sha8({"ts": datetime(2024, 1, 1)}) == input_sha8({"ts": datetime(2024, 1, 1)})


def test_snapshot_version_is_1_1(tmp_path):
    from agentsnap.core.snapshot import SNAPSHOT_VERSION, write_snapshot, read_snapshot
    assert SNAPSHOT_VERSION == "1.1"
    write_snapshot("v11", str(tmp_path), "m", None, [], "out")
    assert read_snapshot("v11", str(tmp_path))["version"] == "1.1"
