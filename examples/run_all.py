"""
run_all.py -- the matrix runner: runs every example and prints a PASS/FAIL table.

Runs each sibling example as its own subprocess (same interpreter), so a crash
in one example can't take down the run. Useful both as a quick smoke test
("did I break an example while touching shared code?") and as the one-command
way to validate a release against real provider APIs.

Usage:
    python examples/run_all.py                          # mock mode, every example
    python examples/run_all.py --real                    # forwards --real to every example
    python examples/run_all.py --only quickstart,replay  # comma-separated subset (no .py)

Exit code is 1 if any example fails (nonzero returncode); 0 otherwise.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

EXAMPLES_DIR = Path(__file__).parent

# Explicit, not a glob -- run_all should never silently pick up a new script
# before someone deliberately adds it here.
EXAMPLES = [
    "quickstart",
    "replay",
    "streaming",
    "model_tools",
    "async_agents",
    "scenarios",
    "tuning",
    "cli_workflow",
    "pytest_fixture",
    "providers",
]


def _run_one(name: str, real: bool) -> tuple[str, bool, float]:
    script = EXAMPLES_DIR / f"{name}.py"
    cmd = [sys.executable, str(script)]
    if real:
        cmd.append("--real")

    start = time.monotonic()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    elapsed = time.monotonic() - start

    passed = result.returncode == 0
    if not passed:
        print(f"\n--- {name} STDOUT ---\n{result.stdout}")
        print(f"--- {name} STDERR ---\n{result.stderr}")
    return name, passed, elapsed


def _print_matrix(results: list[tuple[str, bool, float]]) -> None:
    width = max(len(name) for name, _, _ in results) + 2
    print("\n" + "=" * (width + 20))
    print(f"  {'EXAMPLE':<{width}}{'RESULT':<10}{'TIME':>8}")
    print("-" * (width + 20))
    for name, passed, elapsed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name:<{width}}{status:<10}{elapsed:>6.1f}s")
    print("=" * (width + 20))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--real",
        action="store_true",
        help="Forward --real to every example (needs provider API keys; degrades to per-example skip hints without them)",
    )
    parser.add_argument(
        "--only",
        default=None,
        help="Comma-separated subset of example names to run (no .py), e.g. quickstart,replay",
    )
    args = parser.parse_args()

    names = EXAMPLES
    if args.only:
        wanted = [n.strip() for n in args.only.split(",") if n.strip()]
        unknown = [n for n in wanted if n not in EXAMPLES]
        if unknown:
            print(f"Unknown example name(s): {', '.join(unknown)}")
            print(f"Available: {', '.join(EXAMPLES)}")
            return 1
        names = wanted

    mode = "real" if args.real else "mock"
    print(f"Running {len(names)} example(s) in {mode} mode...\n")

    results = [_run_one(name, args.real) for name in names]
    _print_matrix(results)

    n_pass = sum(1 for _, passed, _ in results if passed)
    n_fail = len(results) - n_pass
    suffix = f", {n_fail} FAILED" if n_fail else ""
    print(f"\n{n_pass}/{len(results)} passed{suffix}")

    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
