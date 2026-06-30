"""
demo_mock.py -- Run agentsnap end-to-end with NO real API keys.

Covers the full user journey:
  1. Zero instrumentation with PatchSet (recommended)
     - First run: auto-records golden snapshot
     - Identical run: passes with similarity scores
     - Regression: shows formatted error report with percentages
     - Approve: promote last run to golden (agentsnap update)
     - Re-run after approval: passes
  2. Adapter-based instrumentation (alternative)
  3. LangGraph callback-based instrumentation
  4. pytest fixture (code snippet -- what CI looks like)

Run:
    python examples/demo_mock.py
    python examples/demo_mock.py --snapshot-dir /tmp/snap
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agentsnap import PatchSet
from agentsnap.adapters.anthropic import AnthropicAdapter
from agentsnap.adapters.cohere import CohereAdapter
from agentsnap.adapters.google import GeminiAdapter
from agentsnap.adapters.groq import GroqAdapter
from agentsnap.adapters.langgraph import LangGraphAdapter
from agentsnap.adapters.mistral import MistralAdapter
from agentsnap.adapters.openai import OpenAIAdapter
from agentsnap.adapters.tool import ToolAdapter
from agentsnap.core.asserter import AgentAsserter
from agentsnap.core.recorder import AgentRecorder
from agentsnap.core.snapshot import last_run_path, snapshot_path
from agentsnap.exceptions import AgentRegressionError

# ---------------------------------------------------------------------------
# Mock response shapes
# ---------------------------------------------------------------------------

class _Block:
    def __init__(self, text): self.text = text

class _Usage:
    def __init__(self): self.input_tokens = 10; self.output_tokens = 20; self.total_tokens = 30

class _Choice:
    def __init__(self, text):
        self.message = type("M", (), {"content": text, "role": "assistant"})()

class _ChatResponse:
    def __init__(self, text): self.choices = [_Choice(text)]; self.usage = _Usage()

class _AnthropicResponse:
    def __init__(self, text): self.content = [_Block(text)]; self.usage = _Usage()

class _GeminiResponse:
    def __init__(self, text): self.text = text; self.usage_metadata = type("U", (), {"total_token_count": 30})()

class _CohereMessage:
    def __init__(self, text): self.content = [_Block(text)]

class _CohereResponse:
    def __init__(self, text):
        self.message = _CohereMessage(text)
        self.usage = type("U", (), {"input_tokens": 10, "output_tokens": 20})()

class _MistralResponse:
    def __init__(self, text): self.choices = [_Choice(text)]; self.usage = _Usage()


# ---------------------------------------------------------------------------
# Mock clients
# ---------------------------------------------------------------------------

class MockAnthropicMessages:
    def __init__(self, responses): self._it = iter(responses)
    def create(self, **kwargs): return next(self._it)

class MockAnthropicClient:
    def __init__(self, *responses): self.messages = MockAnthropicMessages(responses)


class MockOpenAICompletions:
    def __init__(self, responses): self._it = iter(responses)
    def create(self, **kwargs): return next(self._it)

class MockOpenAIChat:
    def __init__(self, *responses): self.completions = MockOpenAICompletions(responses)

class MockOpenAIClient:
    def __init__(self, *responses): self.chat = MockOpenAIChat(*responses)


class MockGeminiModels:
    def __init__(self, responses): self._it = iter(responses)
    def generate_content(self, model, contents, **kwargs): return next(self._it)

class MockGeminiClient:
    def __init__(self, *responses): self.models = MockGeminiModels(responses)


class MockCohereClient:
    def __init__(self, *responses): self._it = iter(responses)
    def chat(self, **kwargs): return next(self._it)


class MockMistralChatComplete:
    def __init__(self, responses): self._it = iter(responses)
    def complete(self, **kwargs): return next(self._it)

class MockMistralChat:
    def __init__(self, *responses): self.complete = MockMistralChatComplete(responses).complete

class MockMistralClient:
    def __init__(self, *responses): self.chat = MockMistralChat(*responses)


class _MockLangGraph:
    """Minimal fake CompiledGraph that fires AgentSnapCallback events."""

    def __init__(self, llm_text: str, tool_name: str = "lookup") -> None:
        self._llm_text = llm_text
        self._tool_name = tool_name

    def invoke(self, input_data, config=None, **kwargs):
        class _Gen:
            def __init__(self, text): self.text = text
        class _Result:
            def __init__(self, text): self.generations = [[_Gen(text)]]

        for cb in (config or {}).get("callbacks", []):
            if hasattr(cb, "on_llm_end"):
                cb.on_llm_end(_Result(self._llm_text))
            if hasattr(cb, "on_tool_end"):
                cb.on_tool_end(lookup(input_data), name=self._tool_name)
        return f"Result: {lookup(input_data)}"

    def stream(self, input_data, **kwargs):
        return iter([])


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def lookup(query: str) -> str:
    return f"mock result for '{query}'"


SEPARATOR = "=" * 60
THIN = "-" * 60


def header(title: str) -> None:
    print(f"\n{SEPARATOR}\n  {title}\n{SEPARATOR}")


def subheader(title: str) -> None:
    print(f"\n{THIN}\n  {title}\n{THIN}")


# ---------------------------------------------------------------------------
# SECTION 1: Zero instrumentation with PatchSet (recommended)
# ---------------------------------------------------------------------------

def patchset_demo(snapshot_dir: str) -> None:
    """
    Full lifecycle using PatchSet -- no changes to agent code needed.

    Your agent creates a raw anthropic.Anthropic() (or openai.OpenAI()) client.
    Tests wrap the call with PatchSet. That's it.
    """
    import anthropic
    import unittest.mock as mock
    from anthropic.resources.messages.messages import Messages as _AnthMessages

    header("ZERO INSTRUMENTATION (PatchSet)  -- recommended")
    print("  Your agent code stays completely unchanged.")
    print("  Just wrap your tests with PatchSet.\n")

    # ----- Your agent function (zero agentsnap imports) ----------------------
    def my_agent(query: str) -> str:
        """Simulates a user's agent -- raw SDK, no agentsnap code."""
        client = anthropic.Anthropic(api_key="demo-key-no-real-call")
        client.messages.create(
            model="claude-haiku-4-5",
            messages=[{"role": "user", "content": query}],
            max_tokens=50,
        )
        result = lookup(query)
        return f"Result: {result}"

    class _FakeContent:
        text = "I'll look that up."

    class _FakeResp:
        content = [_FakeContent()]
        class usage:
            input_tokens = 5; output_tokens = 10

    name = "demo_patchset"

    # -- Step 1: First run -- no snapshot exists, auto-records golden ---------
    subheader("Step 1  First run -- no snapshot yet, golden recorded automatically")
    with mock.patch.object(_AnthMessages, "create", return_value=_FakeResp()):
        with PatchSet():
            with AgentAsserter(name, snapshot_dir=snapshot_dir) as a:
                a.output = my_agent("What is agentsnap?")
    print(f"  Golden snapshot written: {name}.json")

    # -- Step 2: Identical run -- should pass ---------------------------------
    subheader("Step 2  Identical run -- expect PASS with similarity scores")
    with mock.patch.object(_AnthMessages, "create", return_value=_FakeResp()):
        with PatchSet():
            with AgentAsserter(name, snapshot_dir=snapshot_dir) as a:
                a.output = my_agent("What is agentsnap?")

    # -- Step 3: Regression -- LLM response drifted ---------------------------
    subheader("Step 3  Regression -- LLM gives a different response")

    class _DriftedContent:
        text = "Completely different answer from the LLM."

    class _DriftedResp:
        content = [_DriftedContent()]
        class usage:
            input_tokens = 5; output_tokens = 10

    try:
        with mock.patch.object(_AnthMessages, "create", return_value=_DriftedResp()):
            with PatchSet():
                with AgentAsserter(name, snapshot_dir=snapshot_dir) as a:
                    a.output = "The agent now produces a different final answer."
    except AgentRegressionError as e:
        print(str(e))  # full report: percentages, PASS/FAIL per step

    # -- Step 4: Approve -- promote last run to golden (agentsnap update) -----
    subheader("Step 4  Approve the change  (agentsnap update demo_patchset)")
    src = last_run_path(name, snapshot_dir)
    dst = snapshot_path(name, snapshot_dir)
    if src.exists():
        shutil.copy2(src, dst)
        print(f"  Approved -- .last_run/{name}.json promoted to golden.")
        print(f"  (In real use: agentsnap update {name})")

    # -- Step 5: Re-run with new golden -- should pass ------------------------
    subheader("Step 5  Re-run after approval -- expect PASS with new baseline")
    with mock.patch.object(_AnthMessages, "create", return_value=_DriftedResp()):
        with PatchSet():
            with AgentAsserter(name, snapshot_dir=snapshot_dir) as a:
                a.output = "The agent now produces a different final answer."


# ---------------------------------------------------------------------------
# SECTION 2: Adapter-based instrumentation (alternative)
# ---------------------------------------------------------------------------

def run_adapter_demo(provider: str, make_client, call_llm, snapshot_dir: str) -> None:
    """Record, assert, catch regression for one adapter-wrapped provider."""
    tool = ToolAdapter(lookup, name="lookup")
    name = f"demo_{provider}"

    print(f"\n[{provider}] recording...")
    with AgentRecorder(name, snapshot_dir=snapshot_dir) as rec:
        client = make_client()
        rec.output = call_llm(client, tool, "What is agentsnap?")
    print(f"[{provider}] snapshot written -> {name}.json")

    print(f"[{provider}] asserting (identical run)...")
    with AgentAsserter(name, snapshot_dir=snapshot_dir) as a:
        a.output = call_llm(make_client(), ToolAdapter(lookup, name="lookup"), "What is agentsnap?")

    print(f"[{provider}] simulating regression (drifted output)...")
    try:
        with AgentAsserter(name, snapshot_dir=snapshot_dir) as a:
            a.output = "drifted output -- agent changed its answer"
    except AgentRegressionError as e:
        print(f"[{provider}] caught: {e.diff_report.failed_checks}")


def adapters_demo(snapshot_dir: str) -> None:
    header("ADAPTER-BASED INSTRUMENTATION  -- alternative to PatchSet")
    print("  Wrap the SDK client in the agentsnap adapter. Explicit per-provider.\n")

    def _resp(text): return _AnthropicResponse(text)
    def _chat(text): return _ChatResponse(text)
    def _gem(text): return _GeminiResponse(text)
    def _coh(text): return _CohereResponse(text)
    def _mis(text): return _MistralResponse(text)

    run_adapter_demo(
        "anthropic",
        lambda: AnthropicAdapter(MockAnthropicClient(_resp("I'll look that up."), _resp("I'll look that up."))),
        lambda c, t, q: (c.messages.create(model="claude-sonnet-4-6", messages=[{"role": "user", "content": q}], max_tokens=50), t(query=q)) and f"Result: {t(query=q)}",
        snapshot_dir,
    )

    # Simpler lambda approach for remaining providers
    def _make_and_call(adapter_cls, mock_cls, resp_fn, call_fn):
        def make():
            return adapter_cls(mock_cls(resp_fn("I'll look that up."), resp_fn("I'll look that up.")))
        return make, call_fn

    run_adapter_demo(
        "openai",
        lambda: OpenAIAdapter(MockOpenAIClient(_chat("I'll look that up."), _chat("I'll look that up."))),
        lambda c, t, q: (c.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": q}], max_tokens=50), f"Result: {t(query=q)}")[1],
        snapshot_dir,
    )

    run_adapter_demo(
        "gemini",
        lambda: GeminiAdapter(MockGeminiClient(_gem("I'll look that up."), _gem("I'll look that up."))),
        lambda c, t, q: (c.models.generate_content(model="gemini-2.0-flash", contents=q), f"Result: {t(query=q)}")[1],
        snapshot_dir,
    )

    run_adapter_demo(
        "cohere",
        lambda: CohereAdapter(MockCohereClient(_coh("I'll look that up."), _coh("I'll look that up."))),
        lambda c, t, q: (c.chat(model="command-r-plus", messages=[{"role": "user", "content": q}]), f"Result: {t(query=q)}")[1],
        snapshot_dir,
    )

    run_adapter_demo(
        "mistral",
        lambda: MistralAdapter(MockMistralClient(_mis("I'll look that up."), _mis("I'll look that up."))),
        lambda c, t, q: (c.chat.complete(model="mistral-large-latest", messages=[{"role": "user", "content": q}]), f"Result: {t(query=q)}")[1],
        snapshot_dir,
    )

    run_adapter_demo(
        "groq",
        lambda: GroqAdapter(MockOpenAIClient(_chat("I'll look that up."), _chat("I'll look that up."))),
        lambda c, t, q: (c.chat.completions.create(model="llama-3.3-70b-versatile", messages=[{"role": "user", "content": q}], max_tokens=50), f"Result: {t(query=q)}")[1],
        snapshot_dir,
    )


# ---------------------------------------------------------------------------
# SECTION 3: LangGraph callback-based instrumentation
# ---------------------------------------------------------------------------

def langgraph_demo(snapshot_dir: str) -> None:
    header("LANGGRAPH  -- callback-based, node-level capture")

    graph = LangGraphAdapter(_MockLangGraph("I'll look that up.", tool_name="lookup"))
    name = "demo_langgraph"

    print("[langgraph] recording...")
    with AgentRecorder(name, snapshot_dir=snapshot_dir) as rec:
        rec.output = graph.invoke("What is agentsnap?")
    print(f"[langgraph] snapshot written -> {name}.json")

    print("[langgraph] asserting (identical run)...")
    with AgentAsserter(name, snapshot_dir=snapshot_dir) as a:
        a.output = graph.invoke("What is agentsnap?")

    print("[langgraph] simulating regression (different node output)...")
    drifted = LangGraphAdapter(_MockLangGraph("Completely different answer.", tool_name="lookup"))
    try:
        with AgentAsserter(name, snapshot_dir=snapshot_dir) as a:
            a.output = drifted.invoke("What is agentsnap?")
    except AgentRegressionError as e:
        print(f"[langgraph] caught: {e.diff_report.failed_checks}")


# ---------------------------------------------------------------------------
# SECTION 4: pytest fixture (code snippet)
# ---------------------------------------------------------------------------

def pytest_fixture_demo() -> None:
    header("PYTEST FIXTURE  -- what CI tests look like")
    print("""
  No conftest.py needed -- agentsnap registers the fixture automatically.

  Write a test:

    def test_my_agent(snapshot):
        with snapshot.assert_agent("my_agent_test") as a:
            result = my_agent("some query")   # your unchanged agent code
            a.output = result

  Run:
    pytest tests/ -v

  On pass:
    [agentsnap] 'my_agent_test' PASSED | structural: ok | output: 97%

  On regression:
    AgentRegressionError: ...
    -- Diff Report ---
      [STRUCTURAL] 67% tool match  (edit distance 1: ...)
      [SEMANTIC] output: 71% (FAIL)  "responses differ in content"
    -----------------------------------------------

  Approve a change:
    agentsnap update my_agent_test

  Environment:
    AGENTSNAP_JUDGE_API_KEY=sk-...   # use LLM judge instead of embeddings
    pytest tests/ -v
""")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--snapshot-dir",
        default="__agent_snapshots__",
        help="Where to write snapshots (default: __agent_snapshots__)",
    )
    args = parser.parse_args()
    snap_dir = args.snapshot_dir

    header("agentsnap mock demo -- no API keys required")
    print(f"Snapshot dir: {snap_dir}")

    patchset_demo(snap_dir)
    adapters_demo(snap_dir)
    langgraph_demo(snap_dir)
    pytest_fixture_demo()

    header("All demos complete")
    snapshots = list(Path(snap_dir).glob("*.json"))
    print(f"Snapshots written: {len(snapshots)}")
    for p in sorted(snapshots):
        print(f"  {p.name}")
