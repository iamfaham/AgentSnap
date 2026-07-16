"""
replay.py -- Replay mode: deterministic asserts with zero live API calls.

One aspect only: `AgentAsserter(mode="replay")` feeds recorded responses back
to your agent instead of calling a real API -- deterministic, free, and fast
enough for every PR. `replay_tools=True` goes one step further and stubs tool
*results* from the recording too, for a fully side-effect-free re-run.

Usage:
    python examples/replay.py             # mock only, no keys/network needed
    python examples/replay.py --real      # mock, then a real-LLM recording
                                            # replayed with the network disabled
                                            # (needs ANTHROPIC_API_KEY, OPENAI_API_KEY,
                                            # or OPENROUTER_API_KEY; prints a skip hint
                                            # and exits 0 if none set)
    python examples/replay.py --keep      # keep the temp snapshot dir, print its path

The journey (both mock_demo and real_demo):
  1. Record a golden snapshot (normally: your real agent + real API).
  2. Replay assert -- recorded responses are fed back to the agent while the
     "live" client is patched to explode if it's ever called. Zero network.
  3. A developer edits the prompt -- replay catches the request-side change
     instantly, still without touching any API.
  4. (mock only) replay_tools=True -- tool RESULTS are stubbed from the
     recording too; an exploding tool function proves it never runs.
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

NAME = "replay"


def _search(query: str) -> str:
    """A tiny local tool -- not an LLM call, just something the agent executes."""
    return f"results for '{query}'"


def _exploding_create(*args, **kwargs):
    raise RuntimeError("NETWORK CALL ATTEMPTED -- replay mode should never do this!")


def _agent(tool, question: str, prompt_template: str) -> str:
    """A tiny agent: one LLM call, one tool call. Raw anthropic client, zero agentsnap imports."""
    import anthropic

    client = anthropic.Anthropic(api_key="demo-key-no-real-call")
    response = client.messages.create(
        model="claude-haiku-4-5",
        messages=[{"role": "user", "content": prompt_template.format(q=question)}],
        max_tokens=50,
    )
    plan = response.content[0].text
    result = tool(query=question)
    return f"{plan} | {result}"


def mock_demo(snapshot_dir: str) -> None:
    ex.header("REPLAY (mock)  --  deterministic asserts with zero live API calls")
    print("  Replay feeds recorded responses back to your agent -- no network, ever.\n")

    from anthropic.resources.messages.messages import Messages as _AnthMessages

    golden_response = ex.make_anthropic_message("I will search for that.")
    prompt_v1 = "Answer concisely: {q}"
    question = "What is Python?"

    ex.subheader("Step 1  Record the golden run (normally hits the real API)")
    with mock.patch.object(_AnthMessages, "create", return_value=golden_response):
        with PatchSet():
            tool = ToolAdapter(_search, name="search")
            with AgentAsserter(NAME, snapshot_dir=snapshot_dir) as a:
                a.output = _agent(tool, question, prompt_v1)
    print(f"  Golden snapshot recorded (with raw_response for replay): {NAME}.json")

    ex.subheader("Step 2  Replay assert -- ZERO live API calls")
    with mock.patch.object(_AnthMessages, "create", side_effect=_exploding_create):
        with PatchSet():
            tool = ToolAdapter(_search, name="search")
            with AgentAsserter(NAME, snapshot_dir=snapshot_dir, mode="replay") as a:
                a.output = _agent(tool, question, prompt_v1)
    print("  PASSED deterministically -- recorded response was replayed.")

    ex.subheader("Step 3  A dev edits the prompt template; replay catches it")
    prompt_v2 = "You are a pirate. Answer: {q}"  # the 'accidental' change
    try:
        with mock.patch.object(_AnthMessages, "create", side_effect=_exploding_create):
            with PatchSet():
                tool = ToolAdapter(_search, name="search")
                with AgentAsserter(NAME, snapshot_dir=snapshot_dir, mode="replay") as a:
                    a.output = _agent(tool, question, prompt_v2)
        print("  ERROR: should have failed!")
    except AgentRegressionError as e:
        print("  Caught the prompt change without any API call:\n")
        print("  " + str(e).replace("\n", "\n  "))

    ex.subheader("Step 4  replay_tools=True -- tool RESULTS are stubbed too")

    def _exploding_tool(query: str) -> str:
        raise RuntimeError("TOOL CALL ATTEMPTED -- replay_tools=True should never do this!")

    with mock.patch.object(_AnthMessages, "create", side_effect=_exploding_create):
        with PatchSet():
            tool = ToolAdapter(_exploding_tool, name="search")
            with AgentAsserter(
                NAME, snapshot_dir=snapshot_dir, mode="replay", replay_tools=True
            ) as a:
                a.output = _agent(tool, question, prompt_v1)
    print("  PASSED -- the tool function never ran; its result came from the recording.")

    ex.header("Done -- replay on PRs, live nightly for drift.")


def real_demo(snapshot_dir: str) -> None:
    ex.maybe_load_dotenv()
    detected = ex.detect_real_client()
    if detected.client is None:
        ex.header("REPLAY (real)  --  skipped")
        print(f"  {detected.hint}")
        return

    ex.header(f"REPLAY (real)  --  provider: {detected.provider}, model: {detected.model}")
    print("  Record once against the real API, then prove replay never touches the network again.\n")

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

    query_v1 = "Summarize agentsnap in five words."

    ex.subheader("Step 1  Record the golden run against the real API")
    with PatchSet():
        with AgentAsserter(name, snapshot_dir=snapshot_dir, embed_fn=ex.demo_embed) as a:
            a.output = call(query_v1)
    print(f"  Golden snapshot recorded: {name}.json (with raw_response for replay)")

    ex.subheader("Step 2  Replay assert -- ZERO network, even though the golden is real")
    with PatchSet():
        with AgentAsserter(name, snapshot_dir=snapshot_dir, mode="replay", embed_fn=ex.demo_embed) as a:
            a.output = call(query_v1)
    print("  PASSED deterministically -- no API call was made this run.")

    ex.subheader("Step 3  A different prompt; replay catches it without any API call")
    query_v2 = "Write a haiku about snapshot testing."
    try:
        with PatchSet():
            with AgentAsserter(name, snapshot_dir=snapshot_dir, mode="replay", embed_fn=ex.demo_embed) as a:
                a.output = call(query_v2)
        print("  ERROR: should have failed!")
    except AgentRegressionError as e:
        print("  Caught the prompt change without any API call:\n")
        print("  " + str(e).replace("\n", "\n  "))


def main() -> None:
    args = ex.parse_args(__doc__)
    with ex.temp_snapshot_dir(keep=args.keep) as snapshot_dir:
        if args.keep:
            print(f"Snapshot dir: {snapshot_dir}")
        mock_demo(snapshot_dir)
        if args.real:
            real_demo(snapshot_dir)
    ex.header("Replay complete")


if __name__ == "__main__":
    main()
