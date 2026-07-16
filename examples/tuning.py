"""
tuning.py -- Comparison tuning: thresholds and structural tolerance.

One aspect only: the knobs that decide how strict a comparison is.

  - `semantic_threshold` controls how close a paraphrased final output has to
    be to the golden before it counts as a regression. Loosen it for agents
    whose wording legitimately varies; tighten it for agents that must be
    precise.
  - `structural_tolerance` lets a bounded amount of tool-sequence drift pass
    without failing the test -- including drift in which tool the MODEL
    itself decided to call (`tool_requests`), independent of what your code
    executed.

Usage:
    python examples/tuning.py             # mock only, no keys/network needed
    python examples/tuning.py --real      # mock, then an LLM-judge comparison of
                                            # two real paraphrases (needs
                                            # OPENAI_API_KEY or OPENROUTER_API_KEY;
                                            # ANTHROPIC_API_KEY alone can't serve a
                                            # judge endpoint -- prints a skip hint
                                            # and exits 0 in that case)
    python examples/tuning.py --keep      # keep the temp snapshot dir, print its path

The journey (mock_demo):
  1. A paraphrased output passes with a loose `semantic_threshold` and fails
     with a strict one -- same golden, same run, different knob.
  2. The model swaps which tool it asks for (`tool_requests` drift); the
     default `structural_tolerance=0` catches it, but `structural_tolerance=1`
     absorbs it and the PASSED line says so.
"""

from __future__ import annotations

import sys
import unittest.mock as mock
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import _common as ex
from agentsnap import PatchSet
from agentsnap.adapters.tool import ToolAdapter
from agentsnap.core.asserter import AgentAsserter
from agentsnap.exceptions import AgentRegressionError

NAME_OUTPUT = "tuning_output"
NAME_TOOLS = "tuning_tools"

ORIGINAL = "The capital of France is Paris, a city known for the Eiffel Tower."
PARAPHRASE = "Paris is the capital city of France, famous for the Eiffel Tower."


def _search(query: str) -> str:
    return f"results for '{query}'"


def _agent(tool, question: str) -> str:
    """A tiny agent: one LLM call (the model chooses a tool), then execute search."""
    import anthropic

    client = anthropic.Anthropic(api_key="demo-key-no-real-call")
    client.messages.create(
        model="claude-haiku-4-5",
        messages=[{"role": "user", "content": question}],
        max_tokens=100,
    )
    result = tool(query=question)
    return f"answer based on {result}"


def mock_demo(snapshot_dir: str) -> None:
    ex.header("TUNING (mock)  --  thresholds and structural tolerance")
    print("  Same golden, same run -- the knobs decide pass or fail.\n")

    ex.subheader("Step 1  Record the golden output")
    with AgentAsserter(NAME_OUTPUT, snapshot_dir=snapshot_dir, embed_fn=ex.demo_embed) as a:
        a.output = ORIGINAL
    print(f"  Golden: {ORIGINAL!r}")

    ex.subheader("Step 2  Loose semantic_threshold=0.70 -- paraphrase PASSES")
    with AgentAsserter(
        NAME_OUTPUT, snapshot_dir=snapshot_dir, embed_fn=ex.demo_embed, semantic_threshold=0.70
    ) as a:
        a.output = PARAPHRASE
    print(f"  Paraphrase: {PARAPHRASE!r}")

    ex.subheader("Step 3  Strict semantic_threshold=0.90 -- the SAME paraphrase FAILS")
    try:
        with AgentAsserter(
            NAME_OUTPUT, snapshot_dir=snapshot_dir, embed_fn=ex.demo_embed, semantic_threshold=0.90
        ) as a:
            a.output = PARAPHRASE
        print("  ERROR: should have failed!")
    except AgentRegressionError as e:
        print("  Caught by the strict threshold:\n")
        print("  " + str(e).replace("\n", "\n  "))

    ex.subheader("Step 4  Record a golden: the model requests `search`")
    golden = ex.make_anthropic_message(
        "I'll search for that.", tool_uses=[("search", {"query": "capital of France"})]
    )
    from anthropic.resources.messages.messages import Messages as _AnthMessages

    with mock.patch.object(_AnthMessages, "create", return_value=golden):
        with PatchSet():
            tool = ToolAdapter(_search, name="search")
            with AgentAsserter(NAME_TOOLS, snapshot_dir=snapshot_dir) as a:
                a.output = _agent(tool, "capital of France")

    ex.subheader("Step 5  The model now requests `lookup` instead -- structural_tolerance=0 (default) FAILS")
    hijacked = ex.make_anthropic_message(
        "I'll search for that.", tool_uses=[("lookup", {"query": "capital of France"})]
    )
    try:
        with mock.patch.object(_AnthMessages, "create", return_value=hijacked):
            with PatchSet():
                tool = ToolAdapter(_search, name="search")
                with AgentAsserter(NAME_TOOLS, snapshot_dir=snapshot_dir) as a:
                    a.output = _agent(tool, "capital of France")
        print("  ERROR: should have failed!")
    except AgentRegressionError as e:
        print("  Caught the model's tool-choice drift (code still ran `search`):")
        print("  " + str(e).replace("\n", "\n  "))

    ex.subheader("Step 6  Same drift, structural_tolerance=1 -- ABSORBED, test PASSES")
    with mock.patch.object(_AnthMessages, "create", return_value=hijacked):
        with PatchSet():
            tool = ToolAdapter(_search, name="search")
            with AgentAsserter(NAME_TOOLS, snapshot_dir=snapshot_dir, structural_tolerance=1) as a:
                a.output = _agent(tool, "capital of France")

    ex.header("Done -- semantic_threshold and structural_tolerance shape what counts as drift.")


def real_demo(snapshot_dir: str) -> None:
    ex.maybe_load_dotenv()
    detected = ex.detect_real_client()
    if detected.client is None:
        ex.header("TUNING (real)  --  skipped")
        print(f"  {detected.hint}")
        return

    if detected.provider != "openai":
        # detect_real_client() only reaches "anthropic" when ANTHROPIC_API_KEY was
        # the sole key found. LLMJudge always talks to an OpenAI-compatible
        # chat-completions endpoint, so an Anthropic-only key can't serve it.
        ex.header("TUNING (real)  --  judge segment skipped")
        print("  LLMJudge needs an OpenAI-compatible endpoint.")
        print("  Set OPENAI_API_KEY or OPENROUTER_API_KEY to run this segment")
        print("  (ANTHROPIC_API_KEY alone can drive the mock/embedding story above, but not the judge).")
        return

    import os

    from agentsnap.core.diff import LLMJudge

    if os.getenv("OPENAI_API_KEY"):
        api_key, base_url = os.environ["OPENAI_API_KEY"], "https://api.openai.com/v1"
    else:
        api_key, base_url = os.environ["OPENROUTER_API_KEY"], "https://openrouter.ai/api/v1"
    judge = LLMJudge(api_key=api_key, model=detected.model, base_url=base_url)

    ex.header(f"TUNING (real)  --  LLM judge via provider: {detected.provider}, model: {detected.model}")
    print("  A real paraphrase, scored for equivalence by the judge instead of embeddings.\n")

    def call(prompt: str) -> str:
        response = detected.client.chat.completions.create(
            model=detected.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=ex.REAL_MAX_TOKENS,
            temperature=ex.REAL_TEMPERATURE,
        )
        return response.choices[0].message.content or ""

    original = call("In one sentence, state the capital of France.")
    print(f"  Original:   {original!r}")

    paraphrase = call(
        f"Rewrite this sentence with different words but the same meaning: {original!r}"
    )
    print(f"  Paraphrase: {paraphrase!r}")

    score = judge.score(original, paraphrase, key="output")
    reason = judge.last_reasons().get("output", "")
    print(f"\n  Judge score: {score:.2f}")
    print(f"  Judge reason: {reason}")


def main() -> None:
    args = ex.parse_args(__doc__)
    with ex.temp_snapshot_dir(keep=args.keep) as snapshot_dir:
        if args.keep:
            print(f"Snapshot dir: {snapshot_dir}")
        mock_demo(snapshot_dir)
        if args.real:
            real_demo(snapshot_dir)
    ex.header("Tuning complete")


if __name__ == "__main__":
    main()
