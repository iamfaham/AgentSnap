from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SNAPSHOT_VERSION = "1.0"
_LAST_RUN_DIR = ".last_run"


def snapshot_path(test_name: str, snapshot_dir: str) -> Path:
    return Path(snapshot_dir) / f"{test_name}.json"


def last_run_path(test_name: str, snapshot_dir: str) -> Path:
    return Path(snapshot_dir) / _LAST_RUN_DIR / f"{test_name}.json"


def write_snapshot(
    test_name: str,
    snapshot_dir: str,
    model: str,
    input_data: Any,
    trace: list[dict],
    output: str,
) -> Path:
    path = snapshot_path(test_name, snapshot_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "input": input_data,
        "model": model,
        "output": output,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "trace": trace,
        "version": SNAPSHOT_VERSION,
    }
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    return path


def write_last_run(
    test_name: str,
    snapshot_dir: str,
    model: str,
    input_data: Any,
    trace: list[dict],
    output: str,
) -> Path:
    path = last_run_path(test_name, snapshot_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "input": input_data,
        "model": model,
        "output": output,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "trace": trace,
        "version": SNAPSHOT_VERSION,
    }
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    return path


def read_snapshot(test_name: str, snapshot_dir: str) -> dict:
    from agenttest.exceptions import SnapshotNotFoundError

    path = snapshot_path(test_name, snapshot_dir)
    if not path.exists():
        raise SnapshotNotFoundError(test_name)
    return json.loads(path.read_text(encoding="utf-8"))


def list_snapshots(snapshot_dir: str) -> list[Path]:
    d = Path(snapshot_dir)
    if not d.exists():
        return []
    return sorted(p for p in d.glob("*.json"))
