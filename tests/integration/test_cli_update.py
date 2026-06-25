from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from agentsnap.cli import cli
from agentsnap.core.snapshot import last_run_path, snapshot_path


def _write_snap(path: Path, output: str, tools: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    trace = [{"type": "tool_call", "name": t, "args": {}, "result": "r", "step": i}
             for i, t in enumerate(tools)]
    data = {"output": output, "trace": trace, "model": "test", "version": "1.0",
            "recorded_at": "2026-01-01T00:00:00+00:00", "input": None}
    path.write_text(json.dumps(data), encoding="utf-8")


def test_update_shows_diff_and_confirms(tmp_path):
    runner = CliRunner()
    snap_dir = str(tmp_path / "snaps")
    name = "my_test"

    _write_snap(snapshot_path(name, snap_dir), output="old output", tools=["search"])
    _write_snap(last_run_path(name, snap_dir), output="new output", tools=["fetch"])

    result = runner.invoke(cli, ["update", name, f"--snapshot-dir={snap_dir}", "--yes"])

    assert result.exit_code == 0, result.output
    assert "old output" in result.output
    assert "new output" in result.output
    assert "fetch" in result.output


def test_update_aborts_without_yes(tmp_path):
    runner = CliRunner()
    snap_dir = str(tmp_path / "snaps")
    name = "abort_test"

    _write_snap(snapshot_path(name, snap_dir), output="old", tools=[])
    _write_snap(last_run_path(name, snap_dir), output="new", tools=[])

    # Simulate user typing 'n'
    result = runner.invoke(cli, ["update", name, f"--snapshot-dir={snap_dir}"], input="n\n")

    assert result.exit_code != 0
    # Golden should be unchanged
    data = json.loads(snapshot_path(name, snap_dir).read_text())
    assert data["output"] == "old"


def test_update_no_last_run_exits_nonzero(tmp_path):
    runner = CliRunner()
    result = runner.invoke(cli, ["update", "missing", f"--snapshot-dir={tmp_path}"])
    assert result.exit_code != 0


def test_update_no_existing_golden_creates_new(tmp_path):
    runner = CliRunner()
    snap_dir = str(tmp_path / "snaps")
    name = "new_golden"

    _write_snap(last_run_path(name, snap_dir), output="first output", tools=["lookup"])

    result = runner.invoke(cli, ["update", name, f"--snapshot-dir={snap_dir}", "--yes"])
    assert result.exit_code == 0
    assert snapshot_path(name, snap_dir).exists()
