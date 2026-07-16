"""
model_tools.py -- Model tool-decision capture: catch a hijacked tool choice.

Every non-streaming Anthropic/OpenAI `llm_call` event records `tool_requests`
-- the `tool_use` blocks the MODEL decided to call, not just the tool your
code executed. agentsnap compares this sequence on assert, independent of
what your code's own tool-calling logic actually ran.

Usage:
    python examples/model_tools.py             # mock only, no keys/network needed
    python examples/model_tools.py --real      # mock, then a real model given a
                                                 # trivial tool schema (needs
                                                 # ANTHROPIC_API_KEY, OPENAI_API_KEY,
                                                 # or OPENROUTER_API_KEY; prints a skip
                                                 # hint and exits 0 if none set)
    python examples/model_tools.py --keep      # keep the temp snapshot dir, print its path

The journey (mock_demo):
  1. Record a golden run -- the model requests `search`. agentsnap captures
     that decision as `tool_requests` on the llm_call event.
  2. An identical run -- passes, request and execution both match.
  3. The model now requests `delete_file` instead of `search` (e.g. a bad
     model update, a prompt injection, a provider regression) -- caught by
     the "model_tools" check even though the code's own tool sequence and
     final output are unchanged.
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

NAME = "model_tools"


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
    ex.header("MODEL_TOOLS (mock)  --  catch a hijacked tool choice")
    print("  agentsnap captures which tool the MODEL asked for, not just what your code ran.\n")

    from anthropic.resources.messages.messages import Messages as _AnthMessages

    question = "capital of France"

    ex.subheader("Step 1  Record the golden run: the model requests `search`")
    golden = ex.make_anthropic_message(
        "I'll search for that.", tool_uses=[("search", {"query": question})]
    )
    with mock.patch.object(_AnthMessages, "create", return_value=golden):
        with PatchSet():
            tool = ToolAdapter(_search, name="search")
            with AgentAsserter(NAME, snapshot_dir=snapshot_dir) as a:
                a.output = _agent(tool, question)
    print("  golden captured the model's tool choice: search({'query': 'capital of France'})")

    ex.subheader("Step 2  Identical run: model requests the same tool, same args")
    identical = ex.make_anthropic_message(
        "I'll search for that.", tool_uses=[("search", {"query": question})]
    )
    with mock.patch.object(_AnthMessages, "create", return_value=identical):
        with PatchSet():
            tool = ToolAdapter(_search, name="search")
            with AgentAsserter(NAME, snapshot_dir=snapshot_dir) as a:
                a.output = _agent(tool, question)
    print("  PASSED -- model tool decision matches the golden.")

    ex.subheader("Step 3  The model now requests `delete_file` instead of `search`")
    hijacked = ex.make_anthropic_message(
        "I'll search for that.", tool_uses=[("delete_file", {"path": "/etc/passwd"})]
    )
    try:
        with mock.patch.object(_AnthMessages, "create", return_value=hijacked):
            with PatchSet():
                tool = ToolAdapter(_search, name="search")
                with AgentAsserter(NAME, snapshot_dir=snapshot_dir) as a:
                    a.output = _agent(tool, question)
        print("  ERROR: should have failed!")
    except AgentRegressionError as e:
        print("  Caught the hijacked tool decision -- code still ran `search`,")
        print("  but the model itself asked for something else:\n")
        print("  " + str(e).replace("\n", "\n  "))

    ex.header("Done -- model_tools catches drift in what the model DECIDES to call.")


def real_demo(snapshot_dir: str) -> None:
    ex.maybe_load_dotenv()
    detected = ex.detect_real_client()
    if detected.client is None:
        ex.header("MODEL_TOOLS (real)  --  skipped")
        print(f"  {detected.hint}")
        return

    ex.header(f"MODEL_TOOLS (real)  --  provider: {detected.provider}, model: {detected.model}")
    print("  A trivial tool schema so the real model actually emits a tool call.\n")

    name = f"{NAME}_real"
    question = "What is the weather in Paris? Use the get_weather tool to answer."

    def call() -> str:
        if detected.provider == "anthropic":
            tools = [
                {
                    "name": "get_weather",
                    "description": "Get the current weather for a city.",
                    "input_schema": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                }
            ]
            response = detected.client.messages.create(
                model=detected.model,
                messages=[{"role": "user", "content": question}],
                max_tokens=ex.REAL_MAX_TOKENS,
                temperature=ex.REAL_TEMPERATURE,
                tools=tools,
            )
            text = "".join(getattr(b, "text", "") for b in response.content)
            tool_names = [b.name for b in response.content if getattr(b, "type", "") == "tool_use"]
        else:
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get the current weather for a city.",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                            "required": ["city"],
                        },
                    },
                }
            ]
            response = detected.client.chat.completions.create(
                model=detected.model,
                messages=[{"role": "user", "content": question}],
                max_tokens=ex.REAL_MAX_TOKENS,
                temperature=ex.REAL_TEMPERATURE,
                tools=tools,
            )
            message = response.choices[0].message
            text = message.content or ""
            tool_names = [tc.function.name for tc in (message.tool_calls or [])]
        note = f" (model requested tool(s): {tool_names})" if tool_names else " (model returned text, no tool call)"
        return f"Answer: {text}{note}"

    ex.subheader("Step 1  Record the golden run: the real model decides whether to call get_weather")
    with PatchSet():
        with AgentAsserter(name, snapshot_dir=snapshot_dir) as a:
            a.output = call()
    print(f"  golden snapshot recorded: {name}.json -- {a.output}")
    print("  (tool_requests, if any, are captured on the llm_call event)")

    ex.subheader("Step 2  Replay assert -- ZERO network, same tool decision reproduced")
    with PatchSet():
        with AgentAsserter(name, snapshot_dir=snapshot_dir, mode="replay") as a:
            a.output = call()
    print("  PASSED deterministically -- no API call was made this run.")


def main() -> None:
    args = ex.parse_args(__doc__)
    with ex.temp_snapshot_dir(keep=args.keep) as snapshot_dir:
        if args.keep:
            print(f"Snapshot dir: {snapshot_dir}")
        mock_demo(snapshot_dir)
        if args.real:
            real_demo(snapshot_dir)
    ex.header("Model tools complete")


if __name__ == "__main__":
    main()
