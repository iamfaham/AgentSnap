from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SNAPSHOT_VERSION = "1.0"
_LAST_RUN_DIR = ".last_run"


def _sanitize_scenario(s: str) -> str:
    """Replace any character that is not alphanumeric, dash, or underscore with '_'."""
    return re.sub(r'[^a-zA-Z0-9_-]', '_', s)


def input_sha8(value: Any) -> str:
    """8-char hex hash of a JSON-serializable value. Key order does not affect the result."""
    serialized = json.dumps(value, sort_keys=True, default=str).encode()
    return hashlib.sha256(serialized).hexdigest()[:8]


def snapshot_path(test_name: str, snapshot_dir: str, scenario: str | None = None) -> Path:
    stem = f"{test_name}__{_sanitize_scenario(scenario)}" if scenario else test_name
    return Path(snapshot_dir) / f"{stem}.json"


def last_run_path(test_name: str, snapshot_dir: str, scenario: str | None = None) -> Path:
    stem = f"{test_name}__{_sanitize_scenario(scenario)}" if scenario else test_name
    return Path(snapshot_dir) / _LAST_RUN_DIR / f"{stem}.json"


def write_snapshot(
    test_name: str,
    snapshot_dir: str,
    model: str,
    input_data: Any,
    trace: list[dict],
    output: str,
    scenario: str | None = None,
) -> Path:
    path = snapshot_path(test_name, snapshot_dir, scenario=scenario)
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
    scenario: str | None = None,
) -> Path:
    path = last_run_path(test_name, snapshot_dir, scenario=scenario)
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


def read_snapshot(test_name: str, snapshot_dir: str, scenario: str | None = None) -> dict:
    from agentsnap.exceptions import SnapshotNotFoundError

    path = snapshot_path(test_name, snapshot_dir, scenario=scenario)
    if not path.exists():
        raise SnapshotNotFoundError(test_name)
    return json.loads(path.read_text(encoding="utf-8"))


def list_snapshots(snapshot_dir: str) -> list[Path]:
    d = Path(snapshot_dir)
    if not d.exists():
        return []
    return sorted(p for p in d.glob("*.json"))
