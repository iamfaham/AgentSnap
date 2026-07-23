"""
quickstart.py -- THE golden flow: record, pass, catch a regression, approve, re-pass.

One aspect only: zero-instrumentation recording via `PatchSet` + `AgentAsserter`.
Your agent code stays completely unchanged -- no adapter imports, no wrapping.
Tests just wrap the call with `PatchSet()`.

Usage:
    python examples/quickstart.py             # mock only, no keys/network needed
    python examples/quickstart.py --real       # mock, then the same journey against
                                                # a real LLM (needs ANTHROPIC_API_KEY,
                                                # OPENAI_API_KEY, or OPENROUTER_API_KEY;
                                                # prints a skip hint and exits 0 if none set)
    python examples/quickstart.py --keep       # keep the temp snapshot dir, print its path

The journey (mock_demo):
  1. First run  -- no snapshot yet, golden recorded automatically.
  2. Identical run -- passes with similarity scores.
  3. Regression -- the agent's output drifts; AgentAsserter raises
     AgentRegressionError with a full diff report.
  4. Approve -- promote the failing run to golden (what `agentsnap update` does).
  5. Re-run -- passes against the new baseline.

real_demo follows agentsnap's recommended real-world pattern instead: make
ONE live call to record the golden, then use `mode="replay"` for every
"does it still pass" / "did it regress" check that follows. Replay is
deterministic and free, so the identical-run and re-run checks never depend
on a second live call happening to reproduce the model's wording -- exactly
how real projects should wire this up (record live rarely, replay on every
PR).
"""

from __future__ import annotations

import shutil
import sys
import unittest.mock as mock
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import _common as ex
from agentsnap import PatchSet
from agentsnap.core.asserter import AgentAsserter
from agentsnap.core.snapshot import last_run_path, snapshot_path
from agentsnap.exceptions import AgentRegressionError

NAME = "quickstart"


def _agent(query: str) -> str:
    """A user's agent -- raw anthropic.Anthropic() client, zero agentsnap imports."""
    import anthropic

    client = anthropic.Anthropic(api_key="demo-key-no-real-call")
    client.messages.create(
        model="claude-haiku-4-5",
        messages=[{"role": "user", "content": query}],
        max_tokens=50,
    )
    return f"Result: mock result for '{query}'"


def mock_demo(snapshot_dir: str) -> None:
    ex.header("QUICKSTART (mock)  --  zero instrumentation via PatchSet")
    print("  Your agent code stays unchanged. Tests just wrap it with PatchSet().\n")

    from anthropic.resources.messages.messages import Messages as _AnthMessages

    golden_response = ex.make_anthropic_message("I'll look that up.")
    drifted_response = ex.make_anthropic_message("Completely different answer from the LLM.")
    query = "What is agentsnap?"

    ex.subheader("Step 1  First run -- no snapshot yet, golden recorded automatically")
    with mock.patch.object(_AnthMessages, "create", return_value=golden_response):
        with PatchSet():
            with AgentAsserter(NAME, snapshot_dir=snapshot_dir, embed_fn=ex.demo_embed) as a:
                a.output = _agent(query)
    print(f"  Golden snapshot written: {NAME}.json")

    ex.subheader("Step 2  Identical run -- expect PASS with similarity scores")
    with mock.patch.object(_AnthMessages, "create", return_value=golden_response):
        with PatchSet():
            with AgentAsserter(NAME, snapshot_dir=snapshot_dir, embed_fn=ex.demo_embed) as a:
                a.output = _agent(query)

    ex.subheader("Step 3  Regression -- the agent's output drifts")
    try:
        with mock.patch.object(_AnthMessages, "create", return_value=drifted_response):
            with PatchSet():
                with AgentAsserter(NAME, snapshot_dir=snapshot_dir, embed_fn=ex.demo_embed) as a:
                    a.output = "The agent now produces a completely different final answer."
    except AgentRegressionError as e:
        print(str(e))

    ex.subheader(f"Step 4  Approve the change  (agentsnap update {NAME})")
    src = last_run_path(NAME, snapshot_dir)
    dst = snapshot_path(NAME, snapshot_dir)
    if src.exists():
        shutil.copy2(src, dst)
        print(f"  Approved -- .last_run/{NAME}.json promoted to golden.")
        print(f"  (In real use: agentsnap update {NAME})")

    ex.subheader("Step 5  Re-run after approval -- expect PASS with the new baseline")
    with mock.patch.object(_AnthMessages, "create", return_value=drifted_response):
        with PatchSet():
            with AgentAsserter(NAME, snapshot_dir=snapshot_dir, embed_fn=ex.demo_embed) as a:
                a.output = "The agent now produces a completely different final answer."


def real_demo(snapshot_dir: str) -> None:
    ex.maybe_load_dotenv()
    detected = ex.detect_real_client()
    if detected.client is None:
        ex.header("QUICKSTART (real)  --  skipped")
        print(f"  {detected.hint}")
        return

    ex.header(f"QUICKSTART (real)  --  provider: {detected.provider}, model: {detected.model}")
    print("  The recommended real-world pattern: ONE live call records the golden,")
    print("  then every regression check below replays that recording instead of")
    print("  calling the LLM again -- deterministic, zero flake, zero extra API cost.\n")

    name = f"{NAME}_real"
    query_v1 = "Summarize agentsnap in five words."
    query_v2 = "Write a haiku about snapshot testing."  # the deliberate 'regression'

    def call(query: str) -> str:
        if detected.provider == "anthropic":
            response = detected.client.messages.create(
                model=detected.model,
                messages=[{"role": "user", "content": query}],
                max_tokens=ex.REAL_MAX_TOKENS,
                temperature=ex.REAL_TEMPERATURE,
            )
            text = response.content[0].text
        else:
            response = detected.client.chat.completions.create(
                model=detected.model,
                messages=[{"role": "user", "content": query}],
                max_tokens=ex.REAL_MAX_TOKENS,
                temperature=ex.REAL_TEMPERATURE,
            )
            text = response.choices[0].message.content
        return f"Answer: {text}"

    ex.subheader("Step 1  First run -- the ONLY live call. Golden recorded automatically")
    with PatchSet():
        with AgentAsserter(name, snapshot_dir=snapshot_dir, embed_fn=ex.demo_embed) as a:
            a.output = call(query_v1)
    print(f"  Golden snapshot written: {name}.json (with raw_response for replay)")

    ex.subheader("Step 2  Identical run -- replayed, expect PASS (no API call)")
    with PatchSet():
        with AgentAsserter(
            name, snapshot_dir=snapshot_dir, mode="replay", embed_fn=ex.demo_embed
        ) as a:
            a.output = call(query_v1)

    ex.subheader("Step 3  Regression -- deliberately changed prompt, caught by replay")
    try:
        with PatchSet():
            with AgentAsserter(
                name, snapshot_dir=snapshot_dir, mode="replay", embed_fn=ex.demo_embed
            ) as a:
                a.output = call(query_v2)
    except AgentRegressionError as e:
        print(str(e))

    ex.subheader(f"Step 4  Approve the change  (agentsnap update {name})")
    src = last_run_path(name, snapshot_dir)
    dst = snapshot_path(name, snapshot_dir)
    if src.exists():
        shutil.copy2(src, dst)
        print(f"  Approved -- .last_run/{name}.json promoted to golden.")
        print(f"  (In real use: agentsnap update {name})")

    ex.subheader("Step 5  Re-run after approval -- replayed, expect PASS with the new baseline")
    with PatchSet():
        with AgentAsserter(
            name, snapshot_dir=snapshot_dir, mode="replay", embed_fn=ex.demo_embed
        ) as a:
            a.output = call(query_v2)


def main() -> None:
    args = ex.parse_args(__doc__)
    with ex.temp_snapshot_dir(keep=args.keep) as snapshot_dir:
        if args.keep:
            print(f"Snapshot dir: {snapshot_dir}")
        mock_demo(snapshot_dir)
        if args.real:
            real_demo(snapshot_dir)
    ex.header("Quickstart complete")


if __name__ == "__main__":
    main()
