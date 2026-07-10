from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import click

from agentsnap.core.snapshot import list_snapshots

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


@cli.command("show")
@click.argument("snapshot_file")
def show_cmd(snapshot_file: str) -> None:
    """Pretty-print snapshot contents as JSON."""
    path = Path(snapshot_file)
    if not path.exists():
        click.echo(f"Snapshot not found: {snapshot_file}", err=True)
        raise SystemExit(1)
    data = json.loads(path.read_text(encoding="utf-8"))
    click.echo(json.dumps(data, indent=2, sort_keys=True))


@cli.command("diff")
@click.argument("test_name")
@click.option("--snapshot-dir", default=DEFAULT_SNAPSHOT_DIR, show_default=True)
@click.option("--scenario", default=None, help="Scenario variant to compare (optional).")
def diff_cmd(test_name: str, snapshot_dir: str, scenario: str | None) -> None:
    """Compare the last run against the golden snapshot and show what changed.

    Exits 0 if the comparison passed, 1 if it failed or no backend is configured.
    Run 'agentsnap show <file>' to pretty-print a snapshot file as JSON.
    """
    from agentsnap.config import judge_from_env, load
    from agentsnap.core.diff import DiffConfig, compute_diff
    from agentsnap.core.snapshot import last_run_path, snapshot_path

    golden_p = snapshot_path(test_name, snapshot_dir, scenario=scenario)
    last_run_p = last_run_path(test_name, snapshot_dir, scenario=scenario)

    if not golden_p.exists():
        click.echo(f"No golden snapshot found for '{test_name}' in '{snapshot_dir}'.", err=True)
        raise SystemExit(1)
    if not last_run_p.exists():
        click.echo(
            f"No last run found for '{test_name}'. Run your tests first to generate one.", err=True
        )
        raise SystemExit(1)

    golden = json.loads(golden_p.read_text(encoding="utf-8"))
    run_data = json.loads(last_run_p.read_text(encoding="utf-8"))

    judge = judge_from_env()
    cfg = load()
    config = DiffConfig(
        semantic_threshold=float(cfg.get("semantic_threshold", 0.92)),
        llm_threshold=float(cfg.get("llm_threshold", 0.75)),
        structural_tolerance=int(cfg.get("structural_tolerance", 0)),
        judge=judge,
    )

    try:
        report = compute_diff(
            golden,
            run_data.get("trace", []),
            run_data.get("output", ""),
            config=config,
        )
    except RuntimeError as exc:
        click.echo(f"Cannot compare: {exc}", err=True)
        click.echo("Run 'agentsnap init' to configure a comparison backend.", err=True)
        raise SystemExit(1)

    if report.passed:
        scores = report.semantic_scores or {}
        parts = ["structural: ok"]
        for step, score in scores.items():
            parts.append(f"{step}: {int(score * 100)}%")
        click.echo(f"agentsnap diff '{test_name}': PASSED")
        click.echo(f"  {' | '.join(parts)}")
    else:
        from agentsnap.exceptions import AgentRegressionError
        err = AgentRegressionError(
            test_name, report, golden, run_data.get("trace", []), run_data.get("output", "")
        )
        click.echo(f"agentsnap diff '{test_name}': FAILED")
        click.echo(str(err))
        raise SystemExit(1)


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


def _approve_pairs(pairs: list[tuple[Path, Path]], yes: bool) -> None:
    """Shared confirm + copy tail for the update command."""
    if not yes:
        if not click.confirm(f"\nApprove and update {len(pairs)} snapshot(s)?"):
            click.echo("Aborted.")
            raise SystemExit(1)

    for src, dst in pairs:
        shutil.copy2(src, dst)
        click.echo(f"Updated snapshot: {dst}")


def _is_all_candidate(last_run_p: Path, snapshot_dir: str) -> bool:
    """Decide whether a .last_run/*.json file should be promoted by --all.

    Candidate when:
    1. result.passed is False, or
    2. no golden with the same filename exists, or
    3. no result field AND output/trace differ from the golden.
    Returns False (skip) for unreadable/corrupt last_run files — caller warns separately.
    """
    dst = Path(snapshot_dir) / last_run_p.name
    run_data = json.loads(last_run_p.read_text(encoding="utf-8"))

    if not dst.exists():
        return True

    result = run_data.get("result")
    if result is not None:
        return not result.get("passed", False)

    golden = json.loads(dst.read_text(encoding="utf-8"))
    return (
        run_data.get("output", "") != golden.get("output", "")
        or run_data.get("trace", []) != golden.get("trace", [])
    )


@cli.command("update")
@click.argument("test_name", required=False)
@click.option("--all", "update_all", is_flag=True, default=False, help="Approve every failing or new snapshot.")
@click.option("--snapshot-dir", default=DEFAULT_SNAPSHOT_DIR, show_default=True)
@click.option("--yes", "-y", is_flag=True, default=False, help="Skip confirmation prompt.")
def update_cmd(test_name: str | None, update_all: bool, snapshot_dir: str, yes: bool) -> None:
    """Show what changed and promote the last run to the golden snapshot.

    Promotes all scenario variants: {test_name}.json and {test_name}__*.json.
    Pass --all instead of TEST_NAME to batch-approve every failing or new snapshot.
    """
    if bool(test_name) == bool(update_all):
        raise click.UsageError("Provide a TEST_NAME or --all, not both/neither.")

    last_run_dir = Path(snapshot_dir) / ".last_run"

    if update_all:
        if not last_run_dir.exists():
            click.echo("All snapshots are up to date.")
            return

        pairs: list[tuple[Path, Path]] = []
        for src in sorted(last_run_dir.glob("*.json")):
            try:
                is_candidate = _is_all_candidate(src, snapshot_dir)
            except Exception as exc:
                click.echo(f"Skipping unreadable last run '{src.name}': {exc}", err=True)
                continue

            if not is_candidate:
                continue

            dst = Path(snapshot_dir) / src.name
            pairs.append((src, dst))

            if dst.exists():
                old = json.loads(dst.read_text(encoding="utf-8"))
                new = json.loads(src.read_text(encoding="utf-8"))
                click.echo(f"\n--- {src.name} ---")
                _print_update_diff(old, new)
            else:
                click.echo(f"\n--- {src.name} --- (new golden)")

        if not pairs:
            click.echo("All snapshots are up to date.")
            return

        _approve_pairs(pairs, yes)
        return

    # Collect all last_run files for this test_name (plain + all scenario variants)
    candidates: list[Path] = []
    plain = last_run_dir / f"{test_name}.json"
    if plain.exists():
        candidates.append(plain)
    candidates.extend(sorted(last_run_dir.glob(f"{test_name}__*.json")))

    if not candidates:
        click.echo(
            f"No last run found for '{test_name}'. Run 'agentsnap run' first.", err=True
        )
        raise SystemExit(1)

    # Build (src, dst) pairs and show diffs
    pairs = []
    for src in candidates:
        # Strip .last_run/ directory — golden lives directly in snapshot_dir
        dst = Path(snapshot_dir) / src.name
        pairs.append((src, dst))

        if dst.exists():
            old = json.loads(dst.read_text(encoding="utf-8"))
            new = json.loads(src.read_text(encoding="utf-8"))
            click.echo(f"\n--- {src.name} ---")
            _print_update_diff(old, new)
        else:
            click.echo(f"\n--- {src.name} --- (new golden)")

    _approve_pairs(pairs, yes)


_GITIGNORE_ENTRY = "__agent_snapshots__/.last_run/"

_EXAMPLE_TEST_TEMPLATE = '''"""Example agentsnap snapshot test — replace the fake agent with your own.

First run records a golden snapshot in __agent_snapshots__/ (commit it).
Every run after that asserts against it. Docs: https://github.com/iamfaham/AgentSnap
"""

import pytest


def my_agent(question: str) -> str:
    # Replace with your real agent (raw SDK clients are captured automatically
    # when you also request the `agentsnap_instrument` fixture).
    return f"echo: {question}"


@pytest.mark.skip(reason="template — replace my_agent with your agent, then remove this marker")
def test_my_agent(snapshot):
    with snapshot.run("my_agent") as s:
        s.output = my_agent("What is Python?")
'''


def _ensure_gitignore_entry(project_dir: Path) -> str:
    """Ensure __agent_snapshots__/.last_run/ is ignored by git. Idempotent.

    Creates .gitignore if absent. Appends a `# agentsnap` comment above the
    entry when adding. Does nothing if the exact line is already present.
    """
    gitignore_path = project_dir / ".gitignore"

    if gitignore_path.exists():
        content = gitignore_path.read_text(encoding="utf-8")
        existing_lines = content.splitlines()
        if _GITIGNORE_ENTRY in existing_lines:
            return "already in .gitignore"

        new_content = content
        if new_content and not new_content.endswith("\n"):
            new_content += "\n"
        new_content += f"# agentsnap\n{_GITIGNORE_ENTRY}\n"
        gitignore_path.write_text(new_content, encoding="utf-8")
        return "added to .gitignore"

    gitignore_path.write_text(f"# agentsnap\n{_GITIGNORE_ENTRY}\n", encoding="utf-8")
    return "added to .gitignore"


def _write_example_test(project_dir: Path) -> str:
    """Write the example snapshot test template if it doesn't already exist."""
    tests_dir = project_dir / "tests"
    test_path = tests_dir / "test_agentsnap_example.py"

    if test_path.exists():
        return "tests/test_agentsnap_example.py already exists, not overwriting."

    tests_dir.mkdir(parents=True, exist_ok=True)
    test_path.write_text(_EXAMPLE_TEST_TEMPLATE, encoding="utf-8")
    return "Created example test at tests/test_agentsnap_example.py"


@cli.command("init")
def init_cmd() -> None:
    """Interactive setup wizard - choose LLM judge or offline embeddings."""
    from agentsnap.setup_wizard import (
        _download_model,
        apply_result,
        run_wizard,
    )

    result = run_wizard()
    project_dir = Path.cwd()
    apply_result(result, project_dir)

    click.echo(f"\n{_ensure_gitignore_entry(project_dir)}")

    if click.confirm(
        "Create an example snapshot test at tests/test_agentsnap_example.py?", default=False
    ):
        click.echo(_write_example_test(project_dir))

    if result.backend == "offline":
        if result.pre_download_model:
            click.echo("\nDownloading all-MiniLM-L6-v2...")
            try:
                _download_model()
                click.echo("  Model cached.")
            except RuntimeError as exc:
                click.echo(f"  {exc}")
        else:
            click.echo(
                "\nModel will download on first test run (~22 MB)."
                "\nNote: requires pip install agentsnap[offline] if not already installed."
            )
        click.echo("\nOffline embeddings configured.")
    else:
        if result.save_key_to_env:
            click.echo(f"\n  API key written to .env ({result.api_key_env_var})")
        click.echo("\nLLM judge configured.")

    click.echo("Configuration written to pyproject.toml.")
    click.echo("\nRun `pytest` to verify everything works.")


@cli.command("check")
def check_cmd() -> None:
    """Verify current agentsnap setup and backend connectivity."""
    from agentsnap import config
    from agentsnap.setup_wizard import check_offline_model, test_judge_connection

    cfg = config.load(Path.cwd())
    backend = cfg.get("backend")
    api_key = cfg.get("judge_api_key")

    # Neither wizard nor env key -- nothing configured yet
    if not backend and not api_key:
        click.echo("agentsnap is not configured.")
        click.echo("Run 'agentsnap init' to set up your comparison backend.")
        raise SystemExit(1)

    if api_key:
        judge_base_url = cfg.get("judge_base_url")
        judge_model = cfg.get("judge_model")
        click.echo("Backend : LLM judge")
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

    elif backend == "offline":
        try:
            import sentence_transformers  # noqa: F401
        except ImportError:
            click.echo("Backend : offline embeddings (configured)")
            click.echo("Status  : error - sentence-transformers not installed")
            click.echo("Run: pip install agentsnap[offline]")
            raise SystemExit(1)
        cached = check_offline_model()
        click.echo("Backend : offline embeddings (all-MiniLM-L6-v2)")
        if cached:
            click.echo(f"Model   : cached at {cached}")
            click.echo("Status  : ok")
        else:
            click.echo("Model   : not cached - will download (~22 MB) on first test run")
            click.echo("Status  : ok (will download on first run)")

    else:
        # backend == "judge" but no key resolved
        click.echo("Backend : LLM judge (configured)")
        click.echo("API key : not found")
        click.echo("Set AGENTSNAP_JUDGE_API_KEY in your environment or .env file.")
        raise SystemExit(1)


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


@cli.command("status")
@click.option("--snapshot-dir", default=DEFAULT_SNAPSHOT_DIR, show_default=True)
def status_cmd(snapshot_dir: str) -> None:
    """Show pass/fail/stale status for every snapshot (CI-friendly, exits 1 on FAIL)."""
    snapshots = list_snapshots(snapshot_dir)
    if not snapshots:
        click.echo(f"No snapshots found in '{snapshot_dir}'.")
        return

    last_run_dir = Path(snapshot_dir) / ".last_run"
    counts: dict[str, int] = {}
    any_fail = False
    matched_names: set[str] = set()

    click.echo(f"Snapshots in '{snapshot_dir}':")

    for p in snapshots:
        name = p.stem
        matched_names.add(name)
        last_run_p = last_run_dir / f"{name}.json"

        try:
            golden = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            click.secho(f"  {name:<40} unknown (unreadable)", fg="white", dim=True)
            counts["unreadable"] = counts.get("unreadable", 0) + 1
            continue

        if not last_run_p.exists():
            click.secho(f"  {name:<40} no run", fg="white", dim=True)
            counts["no run"] = counts.get("no run", 0) + 1
            continue

        try:
            run_data = json.loads(last_run_p.read_text(encoding="utf-8"))
        except Exception:
            click.secho(f"  {name:<40} unknown (unreadable)", fg="white", dim=True)
            counts["unreadable"] = counts.get("unreadable", 0) + 1
            continue

        golden_recorded = golden.get("recorded_at", "")
        run_recorded = run_data.get("recorded_at", "")

        if run_recorded <= golden_recorded:
            click.secho(f"  {name:<40} approved (re-run tests)", fg="white", dim=True)
            counts["approved"] = counts.get("approved", 0) + 1
            continue

        result = run_data.get("result")
        if result is None:
            click.secho(f"  {name:<40} unknown (re-run tests)", fg="white", dim=True)
            counts["unknown"] = counts.get("unknown", 0) + 1
            continue

        if result.get("passed"):
            mode = result.get("mode", "")
            line = f"PASS   ({mode})" if mode else "PASS"
            click.secho(f"  {name:<40} {line}", fg="green")
            counts["passed"] = counts.get("passed", 0) + 1
        else:
            failed_checks = ",".join(result.get("failed_checks", []))
            click.secho(f"  {name:<40} FAIL   {failed_checks}", fg="red")
            counts["failed"] = counts.get("failed", 0) + 1
            any_fail = True

    if last_run_dir.exists():
        for lr in sorted(last_run_dir.glob("*.json")):
            if lr.stem not in matched_names:
                click.secho(f"  {lr.stem:<40} unapproved new run", fg="white", dim=True)
                counts["unapproved"] = counts.get("unapproved", 0) + 1

    labels = {
        "passed": "passed",
        "failed": "failed",
        "no run": "no run",
        "approved": "approved",
        "unknown": "unknown",
        "unreadable": "unreadable",
        "unapproved": "unapproved new run",
    }
    summary_parts = [
        f"{counts[key]} {labels[key]}"
        for key in ("passed", "failed", "no run", "approved", "unknown", "unreadable", "unapproved")
        if counts.get(key)
    ]
    click.echo(f"Summary: {', '.join(summary_parts)}")

    raise SystemExit(1 if any_fail else 0)


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
