"""
pytest_fixture.py -- The pytest plugin, run the way users actually run it.

One aspect only: the `snapshot` fixture's `snapshot.run()` -- the three-line
pattern most projects use inside real `pytest` test files, not called
directly from Python. This example writes a mini test file to a scratch
directory and runs `python -m pytest` against it via subprocess, twice.

Usage:
    python examples/pytest_fixture.py             # mock only, no keys/network needed
    python examples/pytest_fixture.py --real      # mock, then a mini test whose
                                                    # agent makes one real call on
                                                    # the first pytest run and
                                                    # replays it (zero network) on
                                                    # the second, via
                                                    # `pytest --agentsnap-replay`
                                                    # (needs ANTHROPIC_API_KEY,
                                                    # OPENAI_API_KEY, or
                                                    # OPENROUTER_API_KEY; prints a
                                                    # skip hint and exits 0 if none
                                                    # set)
    python examples/pytest_fixture.py --keep      # keep the temp project dir, print its path

The journey (both mock_demo and real_demo):
  1. `pytest <dir> -q` (run 1) -- no snapshot yet, `snapshot.run()` records a
     golden. The "agentsnap snapshots" terminal-summary section shows RECORDED.
  2. `pytest <dir> -q` (run 2) -- asserts against the golden. The summary
     section shows PASSED.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import _common as ex

_MOCK_TEST_FILE = '''"""Mini test file -- the 3-line snapshot.run() pattern, zero LLM calls."""


def my_agent(query: str) -> str:
    return f"echo: {query}"


def test_my_agent(snapshot):
    with snapshot.run("my_agent") as s:
        s.output = my_agent("hello")
'''


def _run_pytest(project_dir: str, *extra_args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "pytest", project_dir, "-q", *extra_args],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=project_dir,
    )


def _print_summary_section(stdout: str) -> None:
    """Print just the 'agentsnap snapshots' terminal-summary section onward
    (through the final pytest result line)."""
    lines = stdout.splitlines()
    start = next((i for i, line in enumerate(lines) if "agentsnap snapshots" in line), None)
    if start is None:
        print("  (no agentsnap summary section found)")
        return
    for line in lines[start:]:
        print("  " + line)


def _write_project(project_dir: Path, test_file_content: str) -> None:
    (project_dir / "conftest.py").write_text("", encoding="utf-8")
    (project_dir / "test_agentsnap_example.py").write_text(test_file_content, encoding="utf-8")


def mock_demo(snapshot_dir: str) -> None:
    ex.header("PYTEST_FIXTURE (mock)  --  the plugin as users actually run it")
    print("  A real test file, run via `python -m pytest`, twice.\n")

    project_dir = Path(snapshot_dir)
    _write_project(project_dir, _MOCK_TEST_FILE)
    print(f"  Wrote {project_dir / 'test_agentsnap_example.py'}")

    ex.subheader("Run 1  $ python -m pytest -q  (no snapshot yet -- records)")
    result = _run_pytest(str(project_dir))
    assert result.returncode == 0, f"run 1 failed:\n{result.stdout}\n{result.stderr}"
    _print_summary_section(result.stdout)

    ex.subheader("Run 2  $ python -m pytest -q  (snapshot exists -- asserts)")
    result = _run_pytest(str(project_dir))
    assert result.returncode == 0, f"run 2 failed:\n{result.stdout}\n{result.stderr}"
    _print_summary_section(result.stdout)

    ex.header("Done -- RECORDED on first run, PASSED on every run after.")


def _real_test_file(provider: str, model: str) -> str:
    if provider == "anthropic":
        agent_code = f'''import anthropic


def my_agent(query: str) -> str:
    client = anthropic.Anthropic()
    resp = client.messages.create(
        model="{model}",
        messages=[{{"role": "user", "content": query}}],
        max_tokens=100,
        temperature=0,
    )
    return resp.content[0].text
'''
    elif os.getenv("OPENAI_API_KEY"):
        agent_code = f'''import openai


def my_agent(query: str) -> str:
    client = openai.OpenAI()
    resp = client.chat.completions.create(
        model="{model}",
        messages=[{{"role": "user", "content": query}}],
        max_tokens=100,
        temperature=0,
    )
    return resp.choices[0].message.content
'''
    else:
        agent_code = f'''import os
import openai


def my_agent(query: str) -> str:
    client = openai.OpenAI(
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url="https://openrouter.ai/api/v1",
    )
    resp = client.chat.completions.create(
        model="{model}",
        messages=[{{"role": "user", "content": query}}],
        max_tokens=100,
        temperature=0,
    )
    return resp.choices[0].message.content
'''
    return (
        '"""Mini test file -- the 3-line snapshot.run() pattern, one real LLM call."""\n\n'
        + agent_code
        + '''

def test_my_agent(snapshot, agentsnap_instrument):
    with snapshot.run("my_agent_real") as s:
        s.output = my_agent("Summarize agentsnap in five words.")
'''
    )


def real_demo(snapshot_dir: str) -> None:
    ex.maybe_load_dotenv()
    detected = ex.detect_real_client()
    if detected.client is None:
        ex.header("PYTEST_FIXTURE (real)  --  skipped")
        print(f"  {detected.hint}")
        return

    ex.header(f"PYTEST_FIXTURE (real)  --  provider: {detected.provider}, model: {detected.model}")
    print("  Run 1 makes one real call and records it. Run 2 replays it -- the")
    print("  recommended real-world CI pattern: record live occasionally, replay on")
    print("  every PR so tests are deterministic and free.\n")

    project_dir = Path(snapshot_dir)
    _write_project(project_dir, _real_test_file(detected.provider, detected.model))
    print(f"  Wrote {project_dir / 'test_agentsnap_example.py'}")

    ex.subheader("Run 1  $ python -m pytest -q  (records against the real API)")
    result = _run_pytest(str(project_dir))
    assert result.returncode == 0, f"run 1 failed:\n{result.stdout}\n{result.stderr}"
    _print_summary_section(result.stdout)

    ex.subheader("Run 2  $ python -m pytest -q --agentsnap-replay  (ZERO network)")
    result = _run_pytest(str(project_dir), "--agentsnap-replay")
    assert result.returncode == 0, f"run 2 failed:\n{result.stdout}\n{result.stderr}"
    _print_summary_section(result.stdout)


def main() -> None:
    args = ex.parse_args(__doc__)
    with ex.temp_snapshot_dir(keep=args.keep) as snapshot_dir:
        if args.keep:
            print(f"Project dir: {snapshot_dir}")
        mock_demo(snapshot_dir)
        if args.real:
            real_demo(snapshot_dir)
    ex.header("Pytest fixture complete")


if __name__ == "__main__":
    main()
