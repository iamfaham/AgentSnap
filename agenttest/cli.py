from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import click

from agenttest.core.snapshot import list_snapshots, last_run_path, snapshot_path
from agenttest.exceptions import SnapshotNotFoundError

DEFAULT_SNAPSHOT_DIR = "__agent_snapshots__"


@click.group()
def cli() -> None:
    """agenttest — deterministic snapshot testing for AI agents."""


@cli.command("record")
@click.argument("test_file")
@click.option("--snapshot-dir", default=DEFAULT_SNAPSHOT_DIR, show_default=True)
def record_cmd(test_file: str, snapshot_dir: str) -> None:
    """Run a test file and record agent traces as snapshots."""
    import subprocess

    result = subprocess.run(
        [sys.executable, test_file, f"--snapshot-dir={snapshot_dir}", "--mode=record"]
    )
    raise SystemExit(result.returncode)


@cli.command("run")
@click.argument("test_file")
@click.option("--snapshot-dir", default=DEFAULT_SNAPSHOT_DIR, show_default=True)
def run_cmd(test_file: str, snapshot_dir: str) -> None:
    """Run a test file and assert agent traces against snapshots."""
    import subprocess

    result = subprocess.run(
        [sys.executable, test_file, f"--snapshot-dir={snapshot_dir}", "--mode=assert"]
    )
    raise SystemExit(result.returncode)


@cli.command("diff")
@click.argument("snapshot_file")
def diff_cmd(snapshot_file: str) -> None:
    """Pretty-print snapshot contents."""
    path = Path(snapshot_file)
    if not path.exists():
        click.echo(f"Snapshot not found: {snapshot_file}", err=True)
        raise SystemExit(1)
    data = json.loads(path.read_text(encoding="utf-8"))
    click.echo(json.dumps(data, indent=2, sort_keys=True))


@cli.command("update")
@click.argument("test_name")
@click.option("--snapshot-dir", default=DEFAULT_SNAPSHOT_DIR, show_default=True)
def update_cmd(test_name: str, snapshot_dir: str) -> None:
    """Copy the last run trace over the snapshot (approve a regression)."""
    src = last_run_path(test_name, snapshot_dir)
    dst = snapshot_path(test_name, snapshot_dir)
    if not src.exists():
        click.echo(
            f"No last run found for '{test_name}'. Run 'agenttest run' first.", err=True
        )
        raise SystemExit(1)
    shutil.copy2(src, dst)
    click.echo(f"Updated snapshot: {dst}")


@cli.command("list")
@click.option("--snapshot-dir", default=DEFAULT_SNAPSHOT_DIR, show_default=True)
def list_cmd(snapshot_dir: str) -> None:
    """List all snapshots in the snapshot directory."""
    snapshots = list_snapshots(snapshot_dir)
    if not snapshots:
        click.echo(f"No snapshots found in '{snapshot_dir}'.")
        return
    click.echo(f"Snapshots in '{snapshot_dir}':")
    for p in snapshots:
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            recorded = data.get("recorded_at", "unknown")
            model = data.get("model", "unknown")
            steps = len(data.get("trace", []))
            click.echo(f"  {p.stem:<40} model={model}  steps={steps}  recorded={recorded}")
        except Exception:
            click.echo(f"  {p.stem}")


def main() -> None:
    cli()
