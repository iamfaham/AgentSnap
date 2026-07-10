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


def _write_last_run_with_result(path: Path, output: str, tools: list[str], result: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    trace = [{"type": "tool_call", "name": t, "args": {}, "result": "r", "step": i}
             for i, t in enumerate(tools)]
    data = {"output": output, "trace": trace, "model": "test", "version": "1.0",
            "recorded_at": "2026-01-01T00:00:00+00:00", "input": None, "result": result}
    path.write_text(json.dumps(data), encoding="utf-8")


def test_update_neither_arg_errors(tmp_path):
    runner = CliRunner()
    result = runner.invoke(cli, ["update", f"--snapshot-dir={tmp_path}"])
    assert result.exit_code != 0
    assert "TEST_NAME" in result.output or "--all" in result.output


def test_update_both_arg_and_all_errors(tmp_path):
    runner = CliRunner()
    result = runner.invoke(cli, ["update", "some_name", "--all", f"--snapshot-dir={tmp_path}"])
    assert result.exit_code != 0
    assert "TEST_NAME" in result.output or "--all" in result.output


def test_update_all_promotes_only_candidates(tmp_path):
    runner = CliRunner()
    snap_dir = str(tmp_path / "snaps")

    # 1. Failing result -> candidate
    failing = "failing_test"
    _write_snap(snapshot_path(failing, snap_dir), output="old", tools=["search"])
    _write_last_run_with_result(
        last_run_path(failing, snap_dir), output="new", tools=["fetch"],
        result={"passed": False, "failed_checks": ["output"]},
    )

    # 2. Passing result -> NOT a candidate, must be untouched
    passing = "passing_test"
    _write_snap(snapshot_path(passing, snap_dir), output="stable", tools=["search"])
    _write_last_run_with_result(
        last_run_path(passing, snap_dir), output="stable", tools=["search"],
        result={"passed": True},
    )

    # 3. New golden (no existing golden) -> candidate
    new_one = "new_test"
    _write_snap(last_run_path(new_one, snap_dir), output="brand new", tools=["lookup"])

    # 4. No result field, but differs from golden -> candidate
    differing = "differing_test"
    _write_snap(snapshot_path(differing, snap_dir), output="old diff", tools=["a"])
    _write_snap(last_run_path(differing, snap_dir), output="new diff", tools=["b"])

    result = runner.invoke(cli, ["update", "--all", "--yes", f"--snapshot-dir={snap_dir}"])

    assert result.exit_code == 0, result.output
    assert f"--- {failing}.json ---" in result.output
    assert f"--- {new_one}.json ---" in result.output
    assert f"--- {differing}.json ---" in result.output
    assert f"--- {passing}.json ---" not in result.output

    # Golden files updated for candidates
    assert json.loads(snapshot_path(failing, snap_dir).read_text())["output"] == "new"
    assert json.loads(snapshot_path(new_one, snap_dir).read_text())["output"] == "brand new"
    assert json.loads(snapshot_path(differing, snap_dir).read_text())["output"] == "new diff"

    # Passing golden left untouched
    assert json.loads(snapshot_path(passing, snap_dir).read_text())["output"] == "stable"


def test_update_all_shows_single_confirm_prompt(tmp_path):
    runner = CliRunner()
    snap_dir = str(tmp_path / "snaps")

    a = "cand_a"
    _write_last_run_with_result(last_run_path(a, snap_dir), output="a", tools=[], result=None)
    b = "cand_b"
    _write_last_run_with_result(last_run_path(b, snap_dir), output="b", tools=[], result=None)

    result = runner.invoke(cli, ["update", "--all", f"--snapshot-dir={snap_dir}"], input="y\n")
    assert "Approve and update 2 snapshot(s)?" in result.output
    assert result.exit_code == 0, result.output


def test_update_all_no_candidates_reports_up_to_date(tmp_path):
    runner = CliRunner()
    snap_dir = str(tmp_path / "snaps")
    name = "all_good"

    _write_snap(snapshot_path(name, snap_dir), output="same", tools=["x"])
    _write_last_run_with_result(
        last_run_path(name, snap_dir), output="same", tools=["x"], result={"passed": True},
    )

    result = runner.invoke(cli, ["update", "--all", "--yes", f"--snapshot-dir={snap_dir}"])
    assert result.exit_code == 0, result.output
    assert "All snapshots are up to date." in result.output


def test_update_all_skips_corrupt_last_run(tmp_path):
    runner = CliRunner()
    snap_dir = str(tmp_path / "snaps")

    corrupt = last_run_path("broken_test", snap_dir)
    corrupt.parent.mkdir(parents=True, exist_ok=True)
    corrupt.write_text("{not valid json", encoding="utf-8")

    result = runner.invoke(cli, ["update", "--all", "--yes", f"--snapshot-dir={snap_dir}"])
    assert result.exit_code == 0, result.output
    assert "All snapshots are up to date." in result.output
