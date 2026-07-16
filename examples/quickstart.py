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

The journey (both mock_demo and real_demo):
  1. First run  -- no snapshot yet, golden recorded automatically.
  2. Identical run -- passes with similarity scores.
  3. Regression -- the agent's output drifts; AgentAsserter raises
     AgentRegressionError with a full diff report.
  4. Approve -- promote the failing run to golden (what `agentsnap update` does).
  5. Re-run -- passes against the new baseline.
"""

from __future__ import annotations

import shutil
import sys
import unittest.mock as mock
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import _common as ex
from agentsnap import PatchSet
from agentsnap.core.asserter import AgentAsserter
from agentsnap.core.snapshot import last_run_path, snapshot_path
from agentsnap.exceptions import AgentRegressionError

NAME = "quickstart"


def _embed(texts: list[str]) -> list[list[float]]:
    """Deterministic offline embedding stub: hashed bag-of-words.

    Keeps the mock demo free of a sentence-transformers dependency and any
    network call -- identical texts score 1.0, different ones score low,
    which is all the demo needs to show PASS/FAIL.
    """
    vecs = []
    for text in texts:
        v = [0.0] * 256
        for word in text.lower().split():
            v[zlib.crc32(word.encode()) % 256] += 1.0
        vecs.append(v)
    return vecs


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
            with AgentAsserter(NAME, snapshot_dir=snapshot_dir, embed_fn=_embed) as a:
                a.output = _agent(query)
    print(f"  Golden snapshot written: {NAME}.json")

    ex.subheader("Step 2  Identical run -- expect PASS with similarity scores")
    with mock.patch.object(_AnthMessages, "create", return_value=golden_response):
        with PatchSet():
            with AgentAsserter(NAME, snapshot_dir=snapshot_dir, embed_fn=_embed) as a:
                a.output = _agent(query)

    ex.subheader("Step 3  Regression -- the agent's output drifts")
    try:
        with mock.patch.object(_AnthMessages, "create", return_value=drifted_response):
            with PatchSet():
                with AgentAsserter(NAME, snapshot_dir=snapshot_dir, embed_fn=_embed) as a:
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
            with AgentAsserter(NAME, snapshot_dir=snapshot_dir, embed_fn=_embed) as a:
                a.output = "The agent now produces a completely different final answer."


def real_demo(snapshot_dir: str) -> None:
    ex.maybe_load_dotenv()
    detected = ex.detect_real_client()
    if detected.client is None:
        ex.header("QUICKSTART (real)  --  skipped")
        print(f"  {detected.hint}")
        return

    ex.header(f"QUICKSTART (real)  --  provider: {detected.provider}, model: {detected.model}")
    print("  Same journey, against a real LLM. The 'regression' is a deliberate prompt change.\n")

    name = f"{NAME}_real"

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

    ex.subheader("Step 1  First run -- no snapshot yet, golden recorded automatically")
    with PatchSet():
        with AgentAsserter(name, snapshot_dir=snapshot_dir) as a:
            a.output = call("Summarize agentsnap in five words.")
    print(f"  Golden snapshot written: {name}.json")

    ex.subheader("Step 2  Identical run -- expect PASS")
    with PatchSet():
        with AgentAsserter(name, snapshot_dir=snapshot_dir) as a:
            a.output = call("Summarize agentsnap in five words.")

    ex.subheader("Step 3  Regression -- deliberately changed prompt")
    try:
        with PatchSet():
            with AgentAsserter(name, snapshot_dir=snapshot_dir) as a:
                a.output = call("Write a haiku about snapshot testing.")
    except AgentRegressionError as e:
        print(str(e))

    ex.subheader(f"Step 4  Approve the change  (agentsnap update {name})")
    src = last_run_path(name, snapshot_dir)
    dst = snapshot_path(name, snapshot_dir)
    if src.exists():
        shutil.copy2(src, dst)
        print(f"  Approved -- .last_run/{name}.json promoted to golden.")
        print(f"  (In real use: agentsnap update {name})")

    ex.subheader("Step 5  Re-run after approval -- expect PASS with the new baseline")
    with PatchSet():
        with AgentAsserter(name, snapshot_dir=snapshot_dir) as a:
            a.output = call("Write a haiku about snapshot testing.")


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
