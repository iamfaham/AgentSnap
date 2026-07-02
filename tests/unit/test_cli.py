import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from agentsnap.cli import cli

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
