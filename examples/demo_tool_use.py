"""
demo_tool_use.py -- Model tool-decision capture: catch a hijacked tool choice.

  python examples/demo_tool_use.py

No API keys required. Uses a mock Anthropic-shaped client whose responses
include tool_use content blocks -- the tool the MODEL decided to call, not
just the tool your code executed.

The journey:
  1. Record a golden run -- the model requests `search`. agentsnap captures
     that decision as `tool_requests` on the llm_call event.
  2. An identical run -- passes, request and execution both match.
  3. The model now requests `delete_file` instead of `search` (e.g. a bad
     model update, a prompt injection, a provider regression) -- caught by
     the "model_tools" check even though the code's own tool sequence and
     final output are unchanged.
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

SEP = "=" * 70


def header(title: str) -> None:
    print(f"\n{SEP}\n  {title}\n{SEP}")


# ── A minimal mock Anthropic-shaped client (self-contained; no tests/ import) ──

class MockTextBlock:
    def __init__(self, text: str) -> None:
        self.text = text
        self.type = "text"


class MockToolUseBlock:
    def __init__(self, name: str, input: dict, id: str) -> None:
        self.type = "tool_use"
        self.name = name
        self.input = input
        self.id = id


class MockUsage:
    def __init__(self, input_tokens: int = 10, output_tokens: int = 20) -> None:
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class MockAnthropicResponse:
    def __init__(self, text: str, tool_name: str, tool_args: dict, model: str = "claude-mock") -> None:
        self.content = [MockTextBlock(text), MockToolUseBlock(tool_name, tool_args, id="toolu_mock0")]
        self.model = model
        self.usage = MockUsage()

    def model_dump(self, mode: str = "python") -> dict:
        """Anthropic-Message-shaped dict so recorded snapshots are replayable."""
        return {
            "id": "msg_mock",
            "type": "message",
            "role": "assistant",
            "model": self.model,
            "content": [
                {"type": "text", "text": self.content[0].text},
                {"type": "tool_use", "id": self.content[1].id, "name": self.content[1].name, "input": self.content[1].input},
            ],
            "stop_reason": "tool_use",
            "stop_sequence": None,
            "usage": {"input_tokens": self.usage.input_tokens, "output_tokens": self.usage.output_tokens},
        }


class MockMessages:
    def __init__(self, response: MockAnthropicResponse) -> None:
        self._response = response

    def create(self, **kwargs) -> MockAnthropicResponse:
        return self._response


class MockAnthropicClient:
    def __init__(self, response: MockAnthropicResponse) -> None:
        self.messages = MockMessages(response)


def my_agent(client, search_tool, question: str) -> str:
    """A tiny agent: one LLM call (the model chooses a tool), then execute search."""
    client.messages.create(
        model="claude-mock",
        messages=[{"role": "user", "content": question}],
        max_tokens=100,
    )
    result = search_tool(query=question)
    return f"answer based on {result}"


def main() -> None:
    snapshot_dir = tempfile.mkdtemp(prefix="agentsnap_tool_use_demo_")
    try:
        search_tool = ToolAdapter(lambda query: f"results for '{query}'", name="search")

        header("STEP 1 -- Record the golden run: the model requests `search`")
        response = MockAnthropicResponse("I'll search for that.", "search", {"query": "capital of France"})
        client = AnthropicAdapter(MockAnthropicClient(response))
        with AgentRecorder("demo_tool_use", snapshot_dir=snapshot_dir) as rec:
            rec.output = my_agent(client, search_tool, "capital of France")
        print("  golden captured the model's tool choice: search({'query': 'capital of France'})")

        header("STEP 2 -- Identical run: model requests the same tool, same args")
        response = MockAnthropicResponse("I'll search for that.", "search", {"query": "capital of France"})
        client = AnthropicAdapter(MockAnthropicClient(response))
        with AgentAsserter("demo_tool_use", snapshot_dir=snapshot_dir) as a:
            a.output = my_agent(client, search_tool, "capital of France")
        print("  PASSED -- model tool decision matches the golden.")

        header("STEP 3 -- The model now requests `delete_file` instead of `search`")
        response = MockAnthropicResponse("I'll search for that.", "delete_file", {"path": "/etc/passwd"})
        client = AnthropicAdapter(MockAnthropicClient(response))
        try:
            with AgentAsserter("demo_tool_use", snapshot_dir=snapshot_dir) as a:
                a.output = my_agent(client, search_tool, "capital of France")
            print("  ERROR: should have failed!")
        except AgentRegressionError as e:
            print("  Caught the hijacked tool decision -- code still ran `search`,")
            print("  but the model itself asked for something else:\n")
            print("  " + str(e).replace("\n", "\n  "))

        header("Done -- model_tools catches drift in what the model DECIDES to call.")
    finally:
        shutil.rmtree(snapshot_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
