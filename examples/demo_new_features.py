"""
demo_new_features.py -- Hands-on demo of the three new agentsnap features.

  python examples/demo_new_features.py

No API keys required. Uses mock LLM responses so it runs offline.

Features demonstrated:
  1. Inline text excerpts in regression errors
     -- failing llm_call[N] steps now show was:/now: text, not just a score
  2. structural_tolerance end-to-end
     -- a one-tool-call difference can be tolerated without failing the test
  3. agentsnap diff CLI
     -- the diff command now runs the full comparison engine
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agentsnap.core.asserter import AgentAsserter
from agentsnap.core.recorder import AgentRecorder
from agentsnap.core.snapshot import last_run_path, snapshot_path, write_snapshot, write_last_run
from agentsnap.exceptions import AgentRegressionError

SEP  = "=" * 70
THIN = "-" * 70


def header(title: str) -> None:
    print(f"\n{SEP}\n  {title}\n{SEP}")


def subheader(title: str) -> None:
    print(f"\n{THIN}")
    print(f"  {title}")
    print(THIN)


# ---------------------------------------------------------------------------
# Shared mock infrastructure
# ---------------------------------------------------------------------------

class _Block:
    def __init__(self, text): self.text = text

class _Usage:
    input_tokens = 10
    output_tokens = 20

class _AnthropicResponse:
    def __init__(self, text):
        self.content = [_Block(text)]
        self.usage = _Usage()


# ---------------------------------------------------------------------------
# FEATURE 1: Inline text excerpts in regression errors
# ---------------------------------------------------------------------------
#
# Before this change, a failing llm_call[N] semantic check produced:
#
#   [SEMANTIC] llm_call[0]: 12% FAIL
#
# You had no idea what the model said before vs. now. With this change you see:
#
#   [SEMANTIC] llm_call[0]: 12% FAIL
#     was: "I'll look that up for you."
#     now: "That query is outside my scope."
#
# ---------------------------------------------------------------------------

def demo_inline_excerpts(snap_dir: str) -> None:
    header("FEATURE 1 -- Inline text excerpts in regression errors")

    print("""
  When an intermediate LLM response drifts, agentsnap now shows the actual
  before/after text in the error message. Previously you only saw a score.
""")

    from agentsnap.adapters.anthropic import AnthropicAdapter

    class MockMessages:
        def __init__(self, *resps): self._it = iter(resps)
        def create(self, **kw): return next(self._it)

    class MockClient:
        def __init__(self, *resps): self.messages = MockMessages(*resps)

    name = "demo_excerpt"

    # --- Golden: agent says "I'll look that up." then returns a result
    subheader("Step 1  Record the golden snapshot")
    golden_resp = _AnthropicResponse("I'll look that up for you.")
    client = AnthropicAdapter(MockClient(golden_resp))
    with AgentRecorder(name, snapshot_dir=snap_dir) as rec:
        client.messages.create(model="claude-haiku-4-5", messages=[{"role": "user", "content": "test"}], max_tokens=50)
        rec.output = "The result is: 42"
    print(f"  Golden snapshot written: {name}.json")
    print(f"  LLM response in golden: 'I'll look that up for you.'")
    print(f"  Final output in golden: 'The result is: 42'")

    # --- Regression: the LLM now says something completely different
    subheader("Step 2  Run with drifted LLM response and different output")
    regressed_resp = _AnthropicResponse("That query is outside my current scope and I cannot assist.")
    client2 = AnthropicAdapter(MockClient(regressed_resp))
    try:
        with AgentAsserter(name, snapshot_dir=snap_dir) as a:
            client2.messages.create(model="claude-haiku-4-5", messages=[{"role": "user", "content": "test"}], max_tokens=50)
            a.output = "I cannot help with that."
    except AgentRegressionError as e:
        print()
        print("  === AgentRegressionError raised ===")
        print()
        for line in str(e).splitlines():
            print(f"  {line}")
        print()
        print("  ^ Notice: failing llm_call[0] shows was:/now: text, not just a score.")
        print("    You can immediately see which LLM response changed and why the test failed.")


# ---------------------------------------------------------------------------
# FEATURE 2: structural_tolerance end-to-end
# ---------------------------------------------------------------------------
#
# structural_tolerance lets one (or N) extra/missing tool calls pass without
# failing the test. Useful when a minor agent refactor adds a redundant lookup
# that doesn't change the semantics.
#
# Before: any tool sequence change → immediate structural FAIL.
# After:  set structural_tolerance=1 and a one-edit difference is absorbed.
#
# ---------------------------------------------------------------------------

def demo_structural_tolerance(snap_dir: str) -> None:
    header("FEATURE 2 -- structural_tolerance end-to-end")

    print("""
  structural_tolerance is the maximum allowed Levenshtein edit distance
  between the golden and current tool call sequences. Edit distance 1 means
  one tool was added, removed, or renamed.

  You can set it in:
    - pyproject.toml:  [tool.agentsnap]  structural_tolerance = 1
    - pytest ini:      agentsnap_structural_tolerance = 1
    - per-test:        snapshot.assert_agent("name", structural_tolerance=1)
    - direct:          AgentAsserter("name", structural_tolerance=1)

  This demo uses AgentAsserter directly so no API keys or pytest needed.
""")

    from agentsnap.adapters.tool import ToolAdapter

    def lookup(query): return f"result for {query!r}"
    tool = ToolAdapter(lookup, name="lookup")
    def search(query): return f"search result for {query!r}"

    name = "demo_tolerance"

    # --- Record golden: agent calls ONE tool
    subheader("Step 1  Record golden (one tool call: 'lookup')")
    with AgentRecorder(name, snapshot_dir=snap_dir) as rec:
        result = tool(query="what is agentsnap")
        rec.output = f"Answer: {result}"
    print(f"  Golden trace: [lookup]")
    print(f"  Golden output: 'Answer: result for \"what is agentsnap\"'")

    # --- WITHOUT tolerance: adding a second tool call fails structurally
    subheader("Step 2  Run with TWO tool calls (added 'search') -- tolerance=0 (default)")
    tool2 = ToolAdapter(search, name="search")
    try:
        with AgentAsserter(name, snapshot_dir=snap_dir, structural_tolerance=0) as a:
            r1 = tool(query="what is agentsnap")
            r2 = tool2(query="what is agentsnap")
            a.output = f"Answer: {r1} | also: {r2}"
    except AgentRegressionError as e:
        print()
        print("  === AgentRegressionError raised (expected) ===")
        for line in str(e).splitlines():
            print(f"  {line}")
        print()
        print("  ^ structural_tolerance=0 (default): any tool sequence change is a hard fail.")

    # --- WITH tolerance=1: same structural change passes, but keep the output
    #     identical to the golden so we're testing ONLY structural tolerance
    subheader("Step 3  Same structural change -- tolerance=1, output unchanged")
    try:
        with AgentAsserter(name, snapshot_dir=snap_dir, structural_tolerance=1) as a:
            r1 = tool(query="what is agentsnap")
            tool2(query="what is agentsnap")          # extra call happens but output is the same
            a.output = f"Answer: {r1}"                # same as golden
        print()
        print("  PASSED -- the extra 'search' tool call was within tolerance.")
        print("  structural_tolerance=1: an edit distance of 1 is absorbed as acceptable drift.")
        print("  The agent refactored internally but the output was semantically equivalent.")
    except AgentRegressionError as e:
        print(f"  UNEXPECTED FAIL: {e}")


# ---------------------------------------------------------------------------
# FEATURE 3: agentsnap diff CLI
# ---------------------------------------------------------------------------
#
# Before: `agentsnap diff <file>` just pretty-printed the JSON.
# After:  it loads the golden + .last_run, runs the full comparison engine,
#         and reports a human-readable pass/fail summary.
#
# ---------------------------------------------------------------------------

def demo_cli_diff(snap_dir: str) -> None:
    header("FEATURE 3 -- agentsnap diff CLI (semantic comparison)")

    print("""
  'agentsnap diff <test_name>' now runs the full comparison engine between
  the committed golden snapshot and the most recent test run (.last_run/).

  On PASS it shows a clean summary of similarity scores.
  On FAIL it shows the full regression report with was:/now: text excerpts.
""")

    import subprocess

    def run_cli(*args):
        """Run agentsnap CLI and return (returncode, combined output)."""
        result = subprocess.run(
            [sys.executable, "-m", "agentsnap.cli", *args],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace",
        )
        return result.returncode, result.stdout

    # --- Case A: identical golden and last_run -> PASSED
    subheader("Case A  Golden and last run are identical -> PASSED")
    write_snapshot("demo_diff_pass", snap_dir, model="demo", input_data=None,
                   trace=[], output="The capital of France is Paris.")
    write_last_run("demo_diff_pass", snap_dir, model="demo", input_data=None,
                   trace=[], output="The capital of France is Paris.")
    code, out = run_cli("diff", "demo_diff_pass", f"--snapshot-dir={snap_dir}")
    print()
    print("  Command: agentsnap diff demo_diff_pass")
    print(f"  Exit code: {code}")
    print()
    for line in out.strip().splitlines():
        print(f"  {line}")

    # --- Case B: output drifted significantly -> FAILED with text excerpts
    subheader("Case B  Output drifted significantly -> FAILED with text excerpts")
    write_snapshot("demo_diff_fail", snap_dir, model="demo", input_data=None,
                   trace=[], output="The capital of France is Paris, a beautiful city.")
    write_last_run("demo_diff_fail", snap_dir, model="demo", input_data=None,
                   trace=[], output="I have no idea what you are asking about. Please rephrase.")
    code, out = run_cli("diff", "demo_diff_fail", f"--snapshot-dir={snap_dir}")
    print()
    print("  Command: agentsnap diff demo_diff_fail")
    print(f"  Exit code: {code}")
    print()
    for line in out.strip().splitlines():
        print(f"  {line}")
    print()
    print("  ^ Exit code 1 signals failure so CI pipelines can catch regressions.")

    # --- Case C: agentsnap show (the old pretty-print, now its own command)
    subheader("Case C  agentsnap show (old 'diff' behavior -- pretty-prints the JSON)")
    snap_file = str(snapshot_path("demo_diff_pass", snap_dir))
    code, out = run_cli("show", snap_file)
    print()
    print(f"  Command: agentsnap show demo_diff_pass.json")
    print(f"  Exit code: {code}")
    print()
    for line in out.strip().splitlines():
        print(f"  {line}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="agentsnap_feature_demo_") as snap_dir:
        print(f"Using temporary snapshot dir: {snap_dir}")

        demo_inline_excerpts(snap_dir)
        demo_structural_tolerance(snap_dir)
        demo_cli_diff(snap_dir)

        header("All feature demos complete")
        print("""
  Feature 1 -- Inline excerpts:
    Regression errors now show the actual before/after LLM text for every
    failing step. No more guessing which response drifted.

  Feature 2 -- structural_tolerance:
    Set structural_tolerance=N to absorb up to N tool-call edits. Useful for
    refactors that don't change semantics. Configurable via pyproject.toml,
    pytest ini, or per-test.

  Feature 3 -- agentsnap diff:
    The CLI command now runs the full comparison engine (structural + semantic).
    Returns exit code 0 on pass, 1 on fail -- plugs directly into CI.
    Old pretty-print is preserved as 'agentsnap show'.
""")
