"""
demo_async.py -- Async agents: PatchSet intercepts AsyncOpenAI/AsyncAnthropic
clients exactly like their sync counterparts.

  python examples/demo_async.py

No API keys required. Your agent code creates a real `openai.AsyncOpenAI()`
client (this is what Pydantic AI, the OpenAI Agents SDK, and LangChain's
async paths do internally) -- PatchSet patches
`AsyncCompletions.create` for the duration of the `with` block, so the
recorder/asserter captures the call with zero changes to agent code.

The journey:
  1. Record a golden run through a fake `AsyncCompletions.create` (mirrors
     the real SDK's response shape via `model_dump`).
  2. Replay assert -- the fake now RAISES if awaited, proving replay makes
     ZERO live calls; the recorded response is fed back instead.
  3. A developer edits the prompt -- replay catches it without any API call.
"""

from __future__ import annotations

import asyncio
import shutil
import sys
import tempfile
import unittest.mock as mock
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import openai
from openai.resources.chat.completions.completions import AsyncCompletions

from agentsnap import PatchSet
from agentsnap.core.asserter import AgentAsserter
from agentsnap.core.recorder import AgentRecorder
from agentsnap.exceptions import AgentRegressionError

SEP = "=" * 70


def header(title: str) -> None:
    print(f"\n{SEP}\n  {title}\n{SEP}")


# ── A fake AsyncOpenAI-shaped chat completion response ─────────────────────

class _FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content
        self.role = "assistant"


class _FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = _FakeMessage(content)


class _FakeUsage:
    total_tokens = 30


class _FakeChatResponse:
    def __init__(self, text: str, model: str = "gpt-mock") -> None:
        self.choices = [_FakeChoice(text)]
        self.usage = _FakeUsage()
        self.model = model

    def model_dump(self, mode: str = "python") -> dict:
        """OpenAI-ChatCompletion-shaped dict so recordings are replayable."""
        return {
            "id": "chatcmpl_mock",
            "object": "chat.completion",
            "created": 0,
            "model": self.model,
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": self.choices[0].message.content},
                }
            ],
            "usage": {
                "total_tokens": self.usage.total_tokens,
                "prompt_tokens": 10,
                "completion_tokens": 20,
            },
        }


async def my_async_agent(prompt_template: str, question: str) -> str:
    """A tiny async agent: one AsyncOpenAI chat call. No agentsnap imports."""
    client = openai.AsyncOpenAI(api_key="demo-key-no-real-call")
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt_template.format(q=question)}],
        max_tokens=50,
    )
    return response.choices[0].message.content


async def _run() -> None:
    snapshot_dir = tempfile.mkdtemp(prefix="agentsnap_async_demo_")
    try:
        prompt_v1 = "Answer concisely: {q}"

        header("STEP 1 -- Record an async agent (PatchSet intercepts AsyncOpenAI)")
        with mock.patch.object(
            AsyncCompletions, "create",
            mock.AsyncMock(return_value=_FakeChatResponse("I will search for that.")),
        ):
            with PatchSet():
                with AgentRecorder("demo_async", snapshot_dir=snapshot_dir) as rec:
                    rec.output = await my_async_agent(prompt_v1, "What is Python?")
        print("  golden snapshot recorded (async client, with raw_response for replay)")

        header("STEP 2 -- Replay assert: ZERO live async calls")

        async def _exploding_create(self, *args, **kwargs):
            raise RuntimeError("NETWORK CALL ATTEMPTED -- replay mode should never do this!")

        with mock.patch.object(AsyncCompletions, "create", _exploding_create):
            with PatchSet():
                with AgentAsserter("demo_async", snapshot_dir=snapshot_dir, mode="replay") as a:
                    a.output = await my_async_agent(prompt_v1, "What is Python?")
        print("  PASSED deterministically -- recorded async response was replayed.")

        header("STEP 3 -- A dev edits the prompt template; replay catches it")
        prompt_v2 = "You are a pirate. Answer: {q}"  # the 'accidental' change
        try:
            with mock.patch.object(AsyncCompletions, "create", _exploding_create):
                with PatchSet():
                    with AgentAsserter("demo_async", snapshot_dir=snapshot_dir, mode="replay") as a:
                        a.output = await my_async_agent(prompt_v2, "What is Python?")
            print("  ERROR: should have failed!")
        except AgentRegressionError as e:
            print("  Caught the prompt change without any API call:\n")
            print("  " + str(e).replace("\n", "\n  "))

        header("Done -- async clients are intercepted exactly like sync ones.")
    finally:
        shutil.rmtree(snapshot_dir, ignore_errors=True)


def main() -> None:
    asyncio.run(_run())


if __name__ == "__main__":
    main()
