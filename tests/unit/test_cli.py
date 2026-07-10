import json
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from agentsnap.cli import _ensure_gitignore_entry, _write_example_test, cli
from agentsnap.core.diff import DiffReport

_SNAP = {"output": "hello", "trace": [], "model": "m", "input": {}, "version": "1.0", "recorded_at": "2026-01-01T00:00:00+00:00"}
_SNAP2 = {"output": "world", "trace": [], "model": "m", "input": {}, "version": "1.0", "recorded_at": "2026-01-01T00:00:00+00:00"}


def _write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def test_update_promotes_plain_snapshot(tmp_path):
    snap_dir = str(tmp_path)
    last_run = tmp_path / ".last_run" / "my_test.json"
    golden = tmp_path / "my_test.json"
    _write(last_run, _SNAP)
    _write(golden, _SNAP)

    runner = CliRunner()
    result = runner.invoke(cli, ["update", "my_test", f"--snapshot-dir={snap_dir}", "--yes"])
    assert result.exit_code == 0
    assert "my_test.json" in result.output


def test_update_promotes_all_scenario_variants(tmp_path):
    snap_dir = str(tmp_path)
    # Create two scenario last_run files
    _write(tmp_path / ".last_run" / "my_test__aabbccdd.json", _SNAP)
    _write(tmp_path / ".last_run" / "my_test__11223344.json", _SNAP2)
    # Create existing goldens so diff is shown
    _write(tmp_path / "my_test__aabbccdd.json", _SNAP)
    _write(tmp_path / "my_test__11223344.json", _SNAP2)

    runner = CliRunner()
    result = runner.invoke(cli, ["update", "my_test", f"--snapshot-dir={snap_dir}", "--yes"])
    assert result.exit_code == 0
    assert "my_test__aabbccdd.json" in result.output
    assert "my_test__11223344.json" in result.output
    assert (tmp_path / "my_test__aabbccdd.json").exists()
    assert (tmp_path / "my_test__11223344.json").exists()


def test_update_no_last_run_exits_with_error(tmp_path):
    runner = CliRunner()
    result = runner.invoke(cli, ["update", "nonexistent", f"--snapshot-dir={str(tmp_path)}", "--yes"])
    assert result.exit_code != 0


def test_update_promotes_mixed_plain_and_scenario(tmp_path):
    snap_dir = str(tmp_path)
    _write(tmp_path / ".last_run" / "my_test.json", _SNAP)
    _write(tmp_path / ".last_run" / "my_test__aabbccdd.json", _SNAP2)
    _write(tmp_path / "my_test.json", _SNAP)
    _write(tmp_path / "my_test__aabbccdd.json", _SNAP2)

    runner = CliRunner()
    result = runner.invoke(cli, ["update", "my_test", f"--snapshot-dir={snap_dir}", "--yes"])
    assert result.exit_code == 0
    assert (tmp_path / "my_test.json").exists()
    assert (tmp_path / "my_test__aabbccdd.json").exists()


def test_diff_shows_passed_summary_when_report_passes(tmp_path):
    snap_dir = str(tmp_path)
    _write(tmp_path / "my_test.json", _SNAP)
    _write(tmp_path / ".last_run" / "my_test.json", _SNAP)

    with patch("agentsnap.core.diff.compute_diff",
               return_value=DiffReport(passed=True, semantic_scores={"output": 0.97})):
        with patch("agentsnap.config.judge_from_env", return_value=None):
            runner = CliRunner()
            result = runner.invoke(cli, ["diff", "my_test", f"--snapshot-dir={snap_dir}"])

    assert result.exit_code == 0
    assert "PASSED" in result.output
    # Clean summary line, not the error header
    assert "Agent regression" not in result.output
    assert "97%" in result.output


def test_diff_shows_structural_ok_when_tolerance_absorbs_change(tmp_path):
    snap_dir = str(tmp_path)
    _write(tmp_path / "my_test.json", _SNAP)
    _write(tmp_path / ".last_run" / "my_test.json", _SNAP)

    with patch("agentsnap.core.diff.compute_diff",
               return_value=DiffReport(passed=True,
                                       structural_diff="Tool sequence changed (edit distance 1): ...",
                                       semantic_scores={"output": 0.97})):
        with patch("agentsnap.config.judge_from_env", return_value=None):
            runner = CliRunner()
            result = runner.invoke(cli, ["diff", "my_test", f"--snapshot-dir={snap_dir}"])

    assert result.exit_code == 0
    assert "PASSED" in result.output
    assert "structural: ok" in result.output
    assert "mismatch" not in result.output


def test_diff_shows_failed_and_exits_1_when_report_fails(tmp_path):
    snap_dir = str(tmp_path)
    _write(tmp_path / "my_test.json", _SNAP)
    _write(tmp_path / ".last_run" / "my_test.json", _SNAP2)

    with patch("agentsnap.core.diff.compute_diff",
               return_value=DiffReport(passed=False, semantic_scores={"output": 0.30},
                                       failed_checks=["semantic:output"])):
        with patch("agentsnap.config.judge_from_env", return_value=None):
            runner = CliRunner()
            result = runner.invoke(cli, ["diff", "my_test", f"--snapshot-dir={snap_dir}"])

    assert result.exit_code == 1
    assert "FAILED" in result.output


def test_diff_exits_1_when_no_golden(tmp_path):
    runner = CliRunner()
    result = runner.invoke(cli, ["diff", "nonexistent", f"--snapshot-dir={str(tmp_path)}"])
    assert result.exit_code == 1
    assert "No golden snapshot" in result.output


def test_diff_exits_1_when_no_last_run(tmp_path):
    _write(tmp_path / "my_test.json", _SNAP)
    runner = CliRunner()
    result = runner.invoke(cli, ["diff", "my_test", f"--snapshot-dir={str(tmp_path)}"])
    assert result.exit_code == 1
    assert "No last run" in result.output


def test_diff_graceful_error_when_no_backend_configured(tmp_path):
    snap_dir = str(tmp_path)
    _write(tmp_path / "my_test.json", _SNAP)
    _write(tmp_path / ".last_run" / "my_test.json", _SNAP)

    with patch("agentsnap.core.diff.compute_diff",
               side_effect=RuntimeError("No semantic backend configured.")):
        with patch("agentsnap.config.judge_from_env", return_value=None):
            runner = CliRunner()
            result = runner.invoke(cli, ["diff", "my_test", f"--snapshot-dir={snap_dir}"])

    assert result.exit_code == 1
    assert "agentsnap init" in result.output


def test_status_no_snapshots_mirrors_list_message(tmp_path):
    runner = CliRunner()
    result = runner.invoke(cli, ["status", f"--snapshot-dir={tmp_path}"])
    assert result.exit_code == 0
    assert "No snapshots found" in result.output


def test_status_covers_every_state_and_exits_1_on_fail(tmp_path):
    snap_dir = tmp_path
    old_ts = "2026-01-01T00:00:00+00:00"
    new_ts = "2026-01-02T00:00:00+00:00"

    def golden(name, ts=old_ts):
        return {**_SNAP, "recorded_at": ts}

    # PASS: last_run newer than golden, result.passed True
    _write(snap_dir / "passing_agent.json", golden("passing_agent"))
    _write(
        snap_dir / ".last_run" / "passing_agent.json",
        {**_SNAP, "recorded_at": new_ts, "result": {"passed": True, "failed_checks": [], "mode": "live"}},
    )

    # FAIL: last_run newer than golden, result.passed False
    _write(snap_dir / "failing_agent.json", golden("failing_agent"))
    _write(
        snap_dir / ".last_run" / "failing_agent.json",
        {
            **_SNAP,
            "recorded_at": new_ts,
            "result": {"passed": False, "failed_checks": ["semantic:output"], "mode": "replay"},
        },
    )

    # no run: golden with no matching last_run file
    _write(snap_dir / "no_run_agent.json", golden("no_run_agent"))

    # approved (re-run tests): last_run older/equal to golden's recorded_at
    _write(snap_dir / "stale_agent.json", golden("stale_agent", ts=new_ts))
    _write(
        snap_dir / ".last_run" / "stale_agent.json",
        {**_SNAP, "recorded_at": old_ts, "result": {"passed": True, "failed_checks": [], "mode": "live"}},
    )

    # unknown (re-run tests): last_run newer, no result key
    _write(snap_dir / "no_result_agent.json", golden("no_result_agent"))
    _write(snap_dir / ".last_run" / "no_result_agent.json", {**_SNAP, "recorded_at": new_ts})

    # orphan last_run with no matching golden
    _write(
        snap_dir / ".last_run" / "orphan_agent.json",
        {**_SNAP, "recorded_at": new_ts, "result": {"passed": True, "failed_checks": [], "mode": "live"}},
    )

    runner = CliRunner()
    result = runner.invoke(cli, ["status", f"--snapshot-dir={snap_dir}"])

    assert result.exit_code == 1
    assert "passing_agent" in result.output and "PASS" in result.output
    assert "failing_agent" in result.output and "FAIL" in result.output
    assert "semantic:output" in result.output
    assert "FAIL   semantic:output (replay)" in result.output
    assert "no_run_agent" in result.output and "no run" in result.output
    assert "stale_agent" in result.output and "approved (re-run tests)" in result.output
    assert "no_result_agent" in result.output and "unknown (re-run tests)" in result.output
    assert "orphan_agent" in result.output and "unapproved new run" in result.output
    assert "Summary:" in result.output
    assert "1 passed" in result.output
    assert "1 failed" in result.output


def test_status_exits_0_when_no_fail_present(tmp_path):
    snap_dir = tmp_path
    _write(snap_dir / "passing_agent.json", _SNAP)
    _write(
        snap_dir / ".last_run" / "passing_agent.json",
        {**_SNAP, "recorded_at": "2026-02-01T00:00:00+00:00", "result": {"passed": True, "failed_checks": [], "mode": "live"}},
    )
    runner = CliRunner()
    result = runner.invoke(cli, ["status", f"--snapshot-dir={snap_dir}"])
    assert result.exit_code == 0


def test_status_handles_unreadable_json_gracefully(tmp_path):
    snap_dir = tmp_path
    bad = snap_dir / "corrupt_agent.json"
    bad.parent.mkdir(parents=True, exist_ok=True)
    bad.write_text("{not valid json", encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(cli, ["status", f"--snapshot-dir={snap_dir}"])
    assert "unknown (unreadable)" in result.output
    assert result.exit_code == 1


def test_show_pretty_prints_json(tmp_path):
    """'show' command replaces the old 'diff' pretty-print behavior."""
    snap_file = tmp_path / "my_test.json"
    _write(snap_file, _SNAP)
    runner = CliRunner()
    result = runner.invoke(cli, ["show", str(snap_file)])
    assert result.exit_code == 0
    assert '"output"' in result.output
    assert '"hello"' in result.output


# ── _ensure_gitignore_entry ────────────────────────────────────────────────

def test_ensure_gitignore_entry_creates_file(tmp_path):
    msg = _ensure_gitignore_entry(tmp_path)
    assert msg == "added to .gitignore"
    content = (tmp_path / ".gitignore").read_text(encoding="utf-8")
    assert "__agent_snapshots__/.last_run/" in content.splitlines()
    assert "# agentsnap" in content


def test_ensure_gitignore_entry_appends_to_existing_file(tmp_path):
    gitignore = tmp_path / ".gitignore"
    gitignore.write_text("*.pyc\n", encoding="utf-8")

    msg = _ensure_gitignore_entry(tmp_path)
    assert msg == "added to .gitignore"
    lines = gitignore.read_text(encoding="utf-8").splitlines()
    assert "*.pyc" in lines
    assert "__agent_snapshots__/.last_run/" in lines


def test_ensure_gitignore_entry_normalized_match_no_slash(tmp_path):
    gitignore = tmp_path / ".gitignore"
    gitignore.write_text("__agent_snapshots__/.last_run\n", encoding="utf-8")

    msg = _ensure_gitignore_entry(tmp_path)
    assert msg == "already in .gitignore"
    lines = gitignore.read_text(encoding="utf-8").splitlines()
    assert lines == ["__agent_snapshots__/.last_run"]


def test_ensure_gitignore_entry_idempotent(tmp_path):
    _ensure_gitignore_entry(tmp_path)
    msg = _ensure_gitignore_entry(tmp_path)
    assert msg == "already in .gitignore"
    lines = (tmp_path / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert lines.count("__agent_snapshots__/.last_run/") == 1


# ── _write_example_test ────────────────────────────────────────────────────

def test_write_example_test_creates_file_and_dir(tmp_path):
    msg = _write_example_test(tmp_path)
    assert "Created" in msg
    example_path = tmp_path / "tests" / "test_agentsnap_example.py"
    assert example_path.exists()
    content = example_path.read_text(encoding="utf-8")
    assert "pytest.mark.skip" in content
    assert "def my_agent" in content
    assert "def test_my_agent(snapshot):" in content


def test_write_example_test_does_not_overwrite_existing(tmp_path):
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    example_path = tests_dir / "test_agentsnap_example.py"
    custom_content = "# custom, don't touch\n"
    example_path.write_text(custom_content, encoding="utf-8")

    msg = _write_example_test(tmp_path)
    assert "already exists" in msg.lower()
    assert example_path.read_text(encoding="utf-8") == custom_content
