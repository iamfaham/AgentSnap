"""
async_agents.py -- Async agents: PatchSet intercepts AsyncOpenAI/AsyncAnthropic
clients exactly like their sync counterparts.

Your agent code creates a real `openai.AsyncOpenAI()` (or `anthropic.AsyncAnthropic()`)
client -- this is what Pydantic AI, the OpenAI Agents SDK, and LangChain's async
paths do internally. `PatchSet` patches the async `create()` methods for the
duration of the `with` block, so the recorder/asserter captures the call with
zero changes to agent code.

Usage:
    python examples/async_agents.py             # mock only, no keys/network needed
    python examples/async_agents.py --real      # mock, then a real async round trip,
                                                  # then replayed with the network off
                                                  # (needs ANTHROPIC_API_KEY, OPENAI_API_KEY,
                                                  # or OPENROUTER_API_KEY; prints a skip hint
                                                  # and exits 0 if none set)
    python examples/async_agents.py --keep      # keep the temp snapshot dir, print its path

The journey (mock_demo):
  1. Record a golden run through a fake `AsyncCompletions.create` (mirrors
     the real SDK's response shape).
  2. Replay assert -- the fake now RAISES if awaited, proving replay makes
     ZERO live calls; the recorded response is fed back instead.
  3. A developer edits the prompt -- replay catches it without any API call.
"""

from __future__ import annotations

import asyncio
import os
import sys
import unittest.mock as mock
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import _common as ex
from agentsnap import PatchSet
from agentsnap.core.asserter import AgentAsserter
from agentsnap.exceptions import AgentRegressionError

NAME = "async_agents"


async def _agent(prompt_template: str, question: str) -> str:
    """A tiny async agent: one AsyncOpenAI chat call. No agentsnap imports."""
    import openai

    client = openai.AsyncOpenAI(api_key="demo-key-no-real-call")
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt_template.format(q=question)}],
        max_tokens=50,
    )
    return response.choices[0].message.content


async def _exploding_create(self, *args, **kwargs):
    raise RuntimeError("NETWORK CALL ATTEMPTED -- replay mode should never do this!")


async def _mock_demo_async(snapshot_dir: str) -> None:
    ex.header("ASYNC_AGENTS (mock)  --  PatchSet intercepts AsyncOpenAI just like sync clients")
    print("  Your async agent code stays unchanged -- PatchSet patches the async create().\n")

    from openai.resources.chat.completions.completions import AsyncCompletions

    golden = ex.make_openai_chat_completion("I will search for that.")
    prompt_v1 = "Answer concisely: {q}"
    question = "What is Python?"

    ex.subheader("Step 1  Record an async agent (PatchSet intercepts AsyncOpenAI)")
    with mock.patch.object(
        AsyncCompletions, "create", mock.AsyncMock(return_value=golden)
    ):
        with PatchSet():
            with AgentAsserter(NAME, snapshot_dir=snapshot_dir) as a:
                a.output = await _agent(prompt_v1, question)
    print("  golden snapshot recorded (async client, with raw_response for replay)")

    ex.subheader("Step 2  Replay assert: ZERO live async calls")
    with mock.patch.object(AsyncCompletions, "create", _exploding_create):
        with PatchSet():
            with AgentAsserter(NAME, snapshot_dir=snapshot_dir, mode="replay") as a:
                a.output = await _agent(prompt_v1, question)
    print("  PASSED deterministically -- recorded async response was replayed.")

    ex.subheader("Step 3  A dev edits the prompt template; replay catches it")
    prompt_v2 = "You are a pirate. Answer: {q}"  # the 'accidental' change
    try:
        with mock.patch.object(AsyncCompletions, "create", _exploding_create):
            with PatchSet():
                with AgentAsserter(NAME, snapshot_dir=snapshot_dir, mode="replay") as a:
                    a.output = await _agent(prompt_v2, question)
        print("  ERROR: should have failed!")
    except AgentRegressionError as e:
        print("  Caught the prompt change without any API call:\n")
        print("  " + str(e).replace("\n", "\n  "))

    ex.header("Done -- async clients are intercepted exactly like sync ones.")


def mock_demo(snapshot_dir: str) -> None:
    asyncio.run(_mock_demo_async(snapshot_dir))


def _detect_async_client() -> tuple[str | None, object, str | None]:
    """Same key priority as _common.detect_real_client(), but builds async clients."""
    if key := os.getenv("ANTHROPIC_API_KEY"):
        import anthropic

        return "anthropic", anthropic.AsyncAnthropic(api_key=key), "claude-haiku-4-5"
    if key := os.getenv("OPENAI_API_KEY"):
        import openai

        return "openai", openai.AsyncOpenAI(api_key=key), "gpt-4o-mini"
    if key := os.getenv("OPENROUTER_API_KEY"):
        import openai

        return (
            "openai",
            openai.AsyncOpenAI(api_key=key, base_url="https://openrouter.ai/api/v1"),
            "openai/gpt-4o-mini",
        )
    return None, None, None


async def _real_demo_async(snapshot_dir: str) -> None:
    ex.maybe_load_dotenv()
    provider, client, model = _detect_async_client()
    if client is None:
        ex.header("ASYNC_AGENTS (real)  --  skipped")
        print("  no API key found -- set ANTHROPIC_API_KEY, OPENAI_API_KEY, or OPENROUTER_API_KEY to run --real")
        return

    ex.header(f"ASYNC_AGENTS (real)  --  provider: {provider}, model: {model}")
    print("  A real AsyncAnthropic/AsyncOpenAI round trip, then replayed with the network off.\n")

    name = f"{NAME}_real"

    async def call(query: str) -> str:
        if provider == "anthropic":
            response = await client.messages.create(
                model=model,
                messages=[{"role": "user", "content": query}],
                max_tokens=ex.REAL_MAX_TOKENS,
                temperature=ex.REAL_TEMPERATURE,
            )
            return f"Answer: {response.content[0].text}"
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": query}],
            max_tokens=ex.REAL_MAX_TOKENS,
            temperature=ex.REAL_TEMPERATURE,
        )
        return f"Answer: {response.choices[0].message.content}"

    query = "Summarize agentsnap in five words."

    ex.subheader("Step 1  Record an async agent against the real API")
    with PatchSet():
        with AgentAsserter(name, snapshot_dir=snapshot_dir) as a:
            a.output = await call(query)
    print(f"  golden snapshot recorded: {name}.json")

    ex.subheader("Step 2  Replay assert -- ZERO live async calls, even though the golden is real")
    with PatchSet():
        with AgentAsserter(name, snapshot_dir=snapshot_dir, mode="replay") as a:
            a.output = await call(query)
    print("  PASSED deterministically -- no async API call was made this run.")


def real_demo(snapshot_dir: str) -> None:
    asyncio.run(_real_demo_async(snapshot_dir))


def main() -> None:
    args = ex.parse_args(__doc__)
    with ex.temp_snapshot_dir(keep=args.keep) as snapshot_dir:
        if args.keep:
            print(f"Snapshot dir: {snapshot_dir}")
        mock_demo(snapshot_dir)
        if args.real:
            real_demo(snapshot_dir)
    ex.header("Async agents complete")


if __name__ == "__main__":
    main()
