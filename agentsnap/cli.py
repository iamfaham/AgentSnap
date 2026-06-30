from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import click

from agentsnap.core.snapshot import list_snapshots, last_run_path, snapshot_path

DEFAULT_SNAPSHOT_DIR = "__agent_snapshots__"


@click.group()
def cli() -> None:
    """agentsnap - deterministic snapshot testing for AI agents."""


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


def _print_update_diff(old: dict, new: dict) -> None:
    """Print a human-readable diff between golden and last-run snapshots."""
    click.echo("\nChanges to approve:")

    old_output = old.get("output", "")
    new_output = new.get("output", "")
    if old_output != new_output:
        click.echo(f"  output:\n    old: {old_output!r}\n    new: {new_output!r}")
    else:
        click.echo(f"  output: unchanged ({old_output!r})")

    old_tools = [s["name"] for s in old.get("trace", []) if s.get("type") == "tool_call"]
    new_tools = [s["name"] for s in new.get("trace", []) if s.get("type") == "tool_call"]
    if old_tools != new_tools:
        click.echo(f"  tool sequence:\n    old: {old_tools}\n    new: {new_tools}")
    else:
        click.echo(f"  tool sequence: unchanged {old_tools}")

    old_steps = len(old.get("trace", []))
    new_steps = len(new.get("trace", []))
    if old_steps != new_steps:
        click.echo(f"  trace steps: {old_steps} → {new_steps}")

    old_model = old.get("model", "unknown")
    new_model = new.get("model", "unknown")
    if old_model != new_model:
        click.echo(f"  model: {old_model!r} → {new_model!r}")


@cli.command("update")
@click.argument("test_name")
@click.option("--snapshot-dir", default=DEFAULT_SNAPSHOT_DIR, show_default=True)
@click.option("--yes", "-y", is_flag=True, default=False, help="Skip confirmation prompt.")
def update_cmd(test_name: str, snapshot_dir: str, yes: bool) -> None:
    """Show what changed and promote the last run to the golden snapshot."""
    src = last_run_path(test_name, snapshot_dir)
    dst = snapshot_path(test_name, snapshot_dir)

    if not src.exists():
        click.echo(
            f"No last run found for '{test_name}'. Run 'agentsnap run' first.", err=True
        )
        raise SystemExit(1)

    if dst.exists():
        old = json.loads(dst.read_text(encoding="utf-8"))
        new = json.loads(src.read_text(encoding="utf-8"))
        _print_update_diff(old, new)
    else:
        click.echo("No existing snapshot - will create a new golden.")

    if not yes:
        if not click.confirm("\nApprove and update snapshot?"):
            click.echo("Aborted.")
            raise SystemExit(1)

    shutil.copy2(src, dst)
    click.echo(f"Updated snapshot: {dst}")


@cli.command("init")
def init_cmd() -> None:
    """Interactive setup wizard - choose LLM judge or offline embeddings."""
    from agentsnap.setup_wizard import (
        _download_model,
        apply_result,
        run_wizard,
        test_judge_connection,
    )

    result = run_wizard()
    project_dir = Path.cwd()
    apply_result(result, project_dir)

    if result.backend == "offline":
        if result.pre_download_model:
            click.echo("\nDownloading all-MiniLM-L6-v2...")
            _download_model()
            click.echo("  Model cached.")
        else:
            click.echo(
                "\nModel will download automatically on first test run (~22 MB)."
            )
        click.echo("\nOffline embeddings configured.")
    else:
        click.echo("\nTesting connection...")
        try:
            latency = test_judge_connection(
                base_url=result.judge_base_url,
                model=result.judge_model,
                api_key=result.api_key,
            )
            click.echo(f"  Connection ok ({latency:.1f}s)")
        except RuntimeError as exc:
            click.echo(f"  Warning: {exc}", err=True)
            click.echo(
                "  Setup saved anyway - fix the key and re-run `agentsnap check`."
            )

        if result.save_key_to_env:
            click.echo(f"  API key written to .env ({result.api_key_env_var})")
        click.echo("\nLLM judge configured.")

    click.echo("Configuration written to pyproject.toml.")
    click.echo("\nRun `pytest` to verify everything works.")


@cli.command("check")
def check_cmd() -> None:
    """Verify current agentsnap setup and backend connectivity."""
    from agentsnap import config
    from agentsnap.setup_wizard import check_offline_model, test_judge_connection

    cfg = config.load(Path.cwd())
    api_key = cfg.get("judge_api_key")

    if api_key:
        click.echo("Backend : LLM judge")
        judge_base_url = cfg.get("judge_base_url", "https://openrouter.ai/api/v1")
        judge_model = cfg.get("judge_model", "openai/gpt-4o-mini")
        click.echo(f"Provider: {judge_base_url}")
        click.echo(f"Model   : {judge_model}")
        click.echo("API key : found")
        try:
            latency = test_judge_connection(
                base_url=judge_base_url,
                model=judge_model,
                api_key=api_key,
            )
            click.echo(f"Status  : ok ({latency:.2f}s)")
        except RuntimeError as exc:
            click.echo(f"Status  : error - {exc}", err=True)
            raise SystemExit(1)
    else:
        cached = check_offline_model()
        click.echo("Backend : offline embeddings (all-MiniLM-L6-v2)")
        if cached:
            click.echo(f"Model   : cached at {cached}")
            click.echo("Status  : ok")
        else:
            click.echo(
                "Model   : not cached - will download (~22 MB) on first test run"
            )
            click.echo("Status  : ok (will download on first run)")


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
