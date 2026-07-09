"""
demo_replay.py -- Replay mode: deterministic asserts with zero live API calls.

  python examples/demo_replay.py

No API keys required. Uses a mock Anthropic-shaped client for the recording,
then replays it -- the "network" is disabled during replay to prove no live
call happens.

The journey:
  1. Record a golden snapshot (normally: your real agent + real API).
  2. Replay assert -- recorded responses are fed back to the agent.
     Deterministic, free, fast. Perfect for PR CI.
  3. A developer edits the prompt -- replay catches the request-side change
     instantly, without calling any API.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agentsnap.adapters.anthropic import AnthropicAdapter
from agentsnap.adapters.tool import ToolAdapter
from agentsnap.core.asserter import AgentAsserter
from agentsnap.core.recorder import AgentRecorder
from agentsnap.exceptions import AgentRegressionError

sys.path.insert(0, str(Path(__file__).parent.parent / "tests"))
from fixtures.mock_agents import MockAnthropicClient, MockAnthropicResponse  # noqa: E402

SEP = "=" * 70


def header(title: str) -> None:
    print(f"\n{SEP}\n  {title}\n{SEP}")


class NetworkDisabledMessages:
    def create(self, **kwargs):
        raise RuntimeError("NETWORK CALL ATTEMPTED -- replay mode should never do this!")


class NetworkDisabledClient:
    messages = NetworkDisabledMessages()


def my_agent(client, search_tool, question: str, prompt_template: str) -> str:
    """A tiny agent: one LLM call, one tool call."""
    response = client.messages.create(
        model="claude-mock",
        messages=[{"role": "user", "content": prompt_template.format(q=question)}],
        max_tokens=100,
    )
    plan = response.content[0].text
    result = search_tool(query=question)
    return f"{plan} | {result}"


def main() -> None:
    snapshot_dir = tempfile.mkdtemp(prefix="agentsnap_replay_demo_")
    try:
        prompt_v1 = "Answer concisely: {q}"

        header("STEP 1 -- Record the golden run (normally hits the real API)")
        client = AnthropicAdapter(MockAnthropicClient([MockAnthropicResponse("I will search for that.")]))
        tool = ToolAdapter(lambda query: f"results for '{query}'", name="search")
        with AgentRecorder("demo_replay", snapshot_dir=snapshot_dir) as rec:
            rec.output = my_agent(client, tool, "What is Python?", prompt_v1)
        print("  golden snapshot recorded (with raw_response for replay)")

        header("STEP 2 -- Replay assert: same code, ZERO live API calls")
        client = AnthropicAdapter(NetworkDisabledClient())  # proves no live call
        tool = ToolAdapter(lambda query: f"results for '{query}'", name="search")
        with AgentAsserter("demo_replay", snapshot_dir=snapshot_dir, mode="replay") as a:
            a.output = my_agent(client, tool, "What is Python?", prompt_v1)
        print("  PASSED deterministically -- recorded response was replayed.")

        header("STEP 3 -- A dev edits the prompt template; replay catches it")
        prompt_v2 = "You are a pirate. Answer: {q}"  # the 'accidental' change
        client = AnthropicAdapter(NetworkDisabledClient())
        tool = ToolAdapter(lambda query: f"results for '{query}'", name="search")
        try:
            with AgentAsserter("demo_replay", snapshot_dir=snapshot_dir, mode="replay") as a:
                a.output = my_agent(client, tool, "What is Python?", prompt_v2)
            print("  ERROR: should have failed!")
        except AgentRegressionError as e:
            print("  Caught the prompt change without any API call:\n")
            print("  " + str(e).replace("\n", "\n  "))

        header("Done -- replay on PRs, live nightly for drift.")
    finally:
        shutil.rmtree(snapshot_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
