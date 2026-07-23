"""
cli_workflow.py -- The CLI approval loop, driven like a user's terminal.

One aspect only: `agentsnap status` / `agentsnap update` as invoked from a
shell, via `subprocess.run([sys.executable, "-m", "agentsnap.cli", ...])`
with `--snapshot-dir` pointing at a scratch directory. No CLI internals are
imported directly -- this is exactly the command line a developer would type.

Usage:
    python examples/cli_workflow.py             # mock only, no keys/network needed
    python examples/cli_workflow.py --real      # mock, then the same loop with
                                                  # real-recorded snapshots (needs
                                                  # ANTHROPIC_API_KEY, OPENAI_API_KEY,
                                                  # or OPENROUTER_API_KEY; prints a
                                                  # skip hint and exits 0 if none set)
    python examples/cli_workflow.py --keep      # keep the temp snapshot dir, print its path

The journey (mock_demo):
  1. Record a golden, then simulate a drifted run -- the assert fails and
     agentsnap writes a failing `.last_run/*.json` next to the golden.
  2. `agentsnap status` -- shows the failing row, exits 1.
  3. `agentsnap update --all --yes` -- promotes the drifted run to golden.
  4. Re-run against the new baseline, then `agentsnap status` again -- shows
     a passing row, exits 0.

real_demo follows the same recommended pattern as quickstart.py: ONE live
call records the golden, then the "drifted run" is produced deterministically
by asserting a deliberately different prompt in `mode="replay"` -- the
request-side mismatch fails the assert and writes the failing `.last_run`
entry without a second live API call. The rest of the CLI loop
(status/update/status) is identical.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import _common as ex
from agentsnap import PatchSet
from agentsnap.core.asserter import AgentAsserter
from agentsnap.exceptions import AgentRegressionError

NAME = "cli_demo"


def _run_cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "agentsnap.cli", *args],
        capture_output=True,
        text=True,
        timeout=60,
    )


def _print_cli_output(result: subprocess.CompletedProcess) -> None:
    if result.stdout.strip():
        print("  " + result.stdout.strip().replace("\n", "\n  "))
    if result.stderr.strip():
        print("  " + result.stderr.strip().replace("\n", "\n  "))
    print(f"  (exit code: {result.returncode})")


def mock_demo(snapshot_dir: str) -> None:
    ex.header("CLI_WORKFLOW (mock)  --  the approval loop from a shell")
    print("  Every command below runs exactly as a developer would type it.\n")

    original = "The weather in Paris is sunny, 22C."
    drifted = "Completely different weather report."

    ex.subheader("Step 1  Record a golden, then simulate a drifted run")
    with AgentAsserter(NAME, snapshot_dir=snapshot_dir, embed_fn=ex.demo_embed) as a:
        a.output = original
    print(f"  Golden recorded: {NAME}.json")

    try:
        with AgentAsserter(NAME, snapshot_dir=snapshot_dir, embed_fn=ex.demo_embed) as a:
            a.output = drifted
    except AgentRegressionError:
        pass
    print("  Drifted run recorded a FAILING .last_run/cli_demo.json")

    ex.subheader(f"Step 2  $ agentsnap status --snapshot-dir {snapshot_dir}")
    result = _run_cli("status", f"--snapshot-dir={snapshot_dir}")
    _print_cli_output(result)
    assert result.returncode == 1, "status should exit 1 while a snapshot is failing"

    ex.subheader(f"Step 3  $ agentsnap update --all --yes --snapshot-dir {snapshot_dir}")
    result = _run_cli("update", "--all", "--yes", f"--snapshot-dir={snapshot_dir}")
    _print_cli_output(result)
    assert result.returncode == 0, "update --all --yes should exit 0"

    ex.subheader("Step 4  Re-run against the new baseline")
    with AgentAsserter(NAME, snapshot_dir=snapshot_dir, embed_fn=ex.demo_embed) as a:
        a.output = drifted
    print("  PASSED against the newly-approved golden.")

    ex.subheader(f"Step 5  $ agentsnap status --snapshot-dir {snapshot_dir}")
    result = _run_cli("status", f"--snapshot-dir={snapshot_dir}")
    _print_cli_output(result)
    assert result.returncode == 0, "status should exit 0 once the drift is approved and re-verified"

    ex.header("Done -- status/update, exactly as a developer runs them.")


def real_demo(snapshot_dir: str) -> None:
    ex.maybe_load_dotenv()
    detected = ex.detect_real_client()
    if detected.client is None:
        ex.header("CLI_WORKFLOW (real)  --  skipped")
        print(f"  {detected.hint}")
        return

    ex.header(f"CLI_WORKFLOW (real)  --  provider: {detected.provider}, model: {detected.model}")
    print("  ONE live call records the golden; the 'drifted run' below is a deliberately")
    print("  changed prompt caught via mode=\"replay\" -- deterministic, no second API call.\n")

    name = f"{NAME}_real"
    query_v1 = "Summarize agentsnap in five words."
    query_v2 = "Write a haiku about snapshot testing."  # the deliberate 'regression'

    def call(prompt: str) -> str:
        if detected.provider == "anthropic":
            response = detected.client.messages.create(
                model=detected.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=ex.REAL_MAX_TOKENS,
                temperature=ex.REAL_TEMPERATURE,
            )
            text = response.content[0].text
        else:
            response = detected.client.chat.completions.create(
                model=detected.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=ex.REAL_MAX_TOKENS,
                temperature=ex.REAL_TEMPERATURE,
            )
            text = response.choices[0].message.content
        return f"Answer: {text}"

    ex.subheader("Step 1  Record a golden (the only live call), then replay a different prompt")
    with PatchSet():
        with AgentAsserter(name, snapshot_dir=snapshot_dir, embed_fn=ex.demo_embed) as a:
            a.output = call(query_v1)
    print(f"  Golden recorded: {name}.json (with raw_response for replay)")

    try:
        with PatchSet():
            with AgentAsserter(
                name, snapshot_dir=snapshot_dir, mode="replay", embed_fn=ex.demo_embed
            ) as a:
                a.output = call(query_v2)
    except AgentRegressionError:
        pass
    print("  Drifted run recorded a FAILING .last_run entry (no second API call)")

    ex.subheader(f"Step 2  $ agentsnap status --snapshot-dir {snapshot_dir}")
    result = _run_cli("status", f"--snapshot-dir={snapshot_dir}")
    _print_cli_output(result)

    ex.subheader(f"Step 3  $ agentsnap update --all --yes --snapshot-dir {snapshot_dir}")
    result = _run_cli("update", "--all", "--yes", f"--snapshot-dir={snapshot_dir}")
    _print_cli_output(result)

    ex.subheader("Step 4  Re-run against the new baseline -- replayed")
    with PatchSet():
        with AgentAsserter(
            name, snapshot_dir=snapshot_dir, mode="replay", embed_fn=ex.demo_embed
        ) as a:
            a.output = call(query_v2)
    print("  PASSED against the newly-approved golden.")

    ex.subheader(f"Step 5  $ agentsnap status --snapshot-dir {snapshot_dir}")
    result = _run_cli("status", f"--snapshot-dir={snapshot_dir}")
    _print_cli_output(result)


def main() -> None:
    args = ex.parse_args(__doc__)
    with ex.temp_snapshot_dir(keep=args.keep) as snapshot_dir:
        if args.keep:
            print(f"Snapshot dir: {snapshot_dir}")
        mock_demo(snapshot_dir)
        if args.real:
            real_demo(snapshot_dir)
    ex.header("CLI workflow complete")


if __name__ == "__main__":
    main()
