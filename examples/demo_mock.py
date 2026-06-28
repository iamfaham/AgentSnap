"""
demo_mock.py -- Run agentsnap end-to-end with NO real API keys.

Uses in-process mock clients for every provider so you can try the full
record -> assert -> regression -> approve workflow instantly.

Run:
    python examples/demo_mock.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agentsnap.adapters.anthropic import AnthropicAdapter
from agentsnap.adapters.cohere import CohereAdapter
from agentsnap.adapters.google import GeminiAdapter
from agentsnap.adapters.groq import GroqAdapter
from agentsnap.adapters.mistral import MistralAdapter
from agentsnap.adapters.openai import OpenAIAdapter
from agentsnap.adapters.langgraph import LangGraphAdapter
from agentsnap.adapters.tool import ToolAdapter
from agentsnap.core.asserter import AgentAsserter
from agentsnap.core.recorder import AgentRecorder
from agentsnap.exceptions import AgentRegressionError
from agentsnap import PatchSet

# -- Generic mock response shapes ---------------------------------------------

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


# -- Mock clients (one per provider) ------------------------------------------

class MockAnthropicMessages:
    def __init__(self, responses):
        self._it = iter(responses)
    def create(self, **kwargs): return next(self._it)

class MockAnthropicClient:
    def __init__(self, *responses): self.messages = MockAnthropicMessages(responses)


class MockOpenAICompletions:
    def __init__(self, responses):
        self._it = iter(responses)
    def create(self, **kwargs): return next(self._it)

class MockOpenAIChat:
    def __init__(self, *responses): self.completions = MockOpenAICompletions(responses)

class MockOpenAIClient:
    def __init__(self, *responses): self.chat = MockOpenAIChat(*responses)


class MockGeminiModels:
    def __init__(self, responses):
        self._it = iter(responses)
    def generate_content(self, model, contents, **kwargs): return next(self._it)

class MockGeminiClient:
    def __init__(self, *responses): self.models = MockGeminiModels(responses)


class MockCohereClient:
    def __init__(self, *responses):
        self._it = iter(responses)
    def chat(self, **kwargs): return next(self._it)


class MockMistralChatComplete:
    def __init__(self, responses):
        self._it = iter(responses)
    def complete(self, **kwargs): return next(self._it)

class MockMistralChat:
    def __init__(self, *responses): self.complete = MockMistralChatComplete(responses).complete

class MockMistralClient:
    def __init__(self, *responses): self.chat = MockMistralChat(*responses)


# -- Mock LangGraph graph (fires callbacks, no langchain_core import needed) --

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


# -- Shared tool ---------------------------------------------------------------

def lookup(query: str) -> str:
    return f"mock result for '{query}'"


# -- Demo helpers --------------------------------------------------------------

SEPARATOR = "-" * 60

def header(title: str) -> None:
    print(f"\n{SEPARATOR}\n  {title}\n{SEPARATOR}")

def run_demo(provider: str, make_client, call_llm, snapshot_dir: str) -> None:
    """Record then assert for a single provider."""
    tool = ToolAdapter(lookup, name="lookup")
    name = f"demo_{provider}"

    # -- Record ----------------------------------------------------------------
    print(f"[{provider}] recording...")
    with AgentRecorder(name, snapshot_dir=snapshot_dir) as rec:
        client = make_client()
        result = call_llm(client, tool, "What is agentsnap?")
        rec.output = result
    print(f"[{provider}] snapshot written -> {name}.json")

    # -- Assert (same inputs -> should pass) -----------------------------------
    print(f"[{provider}] asserting (identical run)...")
    with AgentAsserter(name, snapshot_dir=snapshot_dir) as a:
        client = make_client()
        tool2 = ToolAdapter(lookup, name="lookup")
        a.output = call_llm(client, tool2, "What is agentsnap?")
    print(f"[{provider}] OK passed")

    # -- Simulate a regression (wrong tool arg) --------------------------------
    print(f"[{provider}] simulating regression (changed tool arg)...")
    def drifted_lookup(query: str) -> str:
        return f"drifted result for '{query}'"

    try:
        with AgentAsserter(name, snapshot_dir=snapshot_dir) as a:
            client = make_client()
            drifted_tool = ToolAdapter(drifted_lookup, name="lookup")
            # Force a different arg value so argument diff fires
            call_llm(client, drifted_tool, "What is agentsnap?")
            a.output = "drifted output"
    except AgentRegressionError as e:
        print(f"[{provider}] OK regression caught: {e.diff_report.failed_checks}")


# -- Per-provider wiring -------------------------------------------------------

def anthropic_demo(snapshot_dir: str) -> None:
    def make(): return AnthropicAdapter(MockAnthropicClient(
        _AnthropicResponse("I'll look that up."),
        _AnthropicResponse("I'll look that up."),
        _AnthropicResponse("I'll look that up."),
    ))
    def call(client, tool, q):
        client.messages.create(model="claude-sonnet-4-6", messages=[{"role": "user", "content": q}], max_tokens=50)
        result = tool(query=q)
        return f"Result: {result}"
    run_demo("anthropic", make, call, snapshot_dir)

def openai_demo(snapshot_dir: str) -> None:
    def make(): return OpenAIAdapter(MockOpenAIClient(
        _ChatResponse("I'll look that up."),
        _ChatResponse("I'll look that up."),
        _ChatResponse("I'll look that up."),
    ))
    def call(client, tool, q):
        client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": q}], max_tokens=50)
        result = tool(query=q)
        return f"Result: {result}"
    run_demo("openai", make, call, snapshot_dir)

def gemini_demo(snapshot_dir: str) -> None:
    def make(): return GeminiAdapter(MockGeminiClient(
        _GeminiResponse("I'll look that up."),
        _GeminiResponse("I'll look that up."),
        _GeminiResponse("I'll look that up."),
    ))
    def call(client, tool, q):
        client.models.generate_content(model="gemini-2.0-flash", contents=q)
        result = tool(query=q)
        return f"Result: {result}"
    run_demo("gemini", make, call, snapshot_dir)

def cohere_demo(snapshot_dir: str) -> None:
    def make(): return CohereAdapter(MockCohereClient(
        _CohereResponse("I'll look that up."),
        _CohereResponse("I'll look that up."),
        _CohereResponse("I'll look that up."),
    ))
    def call(client, tool, q):
        client.chat(model="command-r-plus", messages=[{"role": "user", "content": q}])
        result = tool(query=q)
        return f"Result: {result}"
    run_demo("cohere", make, call, snapshot_dir)

def mistral_demo(snapshot_dir: str) -> None:
    def make(): return MistralAdapter(MockMistralClient(
        _MistralResponse("I'll look that up."),
        _MistralResponse("I'll look that up."),
        _MistralResponse("I'll look that up."),
    ))
    def call(client, tool, q):
        client.chat.complete(model="mistral-large-latest", messages=[{"role": "user", "content": q}])
        result = tool(query=q)
        return f"Result: {result}"
    run_demo("mistral", make, call, snapshot_dir)

def groq_demo(snapshot_dir: str) -> None:
    def make(): return GroqAdapter(MockOpenAIClient(
        _ChatResponse("I'll look that up."),
        _ChatResponse("I'll look that up."),
        _ChatResponse("I'll look that up."),
    ))
    def call(client, tool, q):
        client.chat.completions.create(model="llama-3.3-70b-versatile", messages=[{"role": "user", "content": q}], max_tokens=50)
        result = tool(query=q)
        return f"Result: {result}"
    run_demo("groq", make, call, snapshot_dir)


def zero_instrumentation_demo(snapshot_dir: str) -> None:
    """Shows that raw SDK clients work without any adapter wrapping."""
    import anthropic
    import unittest.mock as mock
    from anthropic.resources.messages.messages import Messages as _AnthMessages

    name = "demo_zero_instrument"

    class _FakeContent:
        text = "I'll look that up."

    class _FakeResp:
        content = [_FakeContent()]
        class usage:
            input_tokens = 5
            output_tokens = 10

    def _agent(query: str) -> str:
        # Raw client — no AnthropicAdapter wrapping
        client = anthropic.Anthropic(api_key="demo-key-no-real-call")
        client.messages.create(
            model="claude-haiku-4-5",
            messages=[{"role": "user", "content": query}],
            max_tokens=50,
        )
        result = lookup(query)
        return f"Result: {result}"

    # -- Record ----------------------------------------------------------------
    print("[zero-instrumentation] recording (raw client, no adapter)...")
    with mock.patch.object(_AnthMessages, "create", return_value=_FakeResp()):
        with PatchSet():
            with AgentRecorder(name, snapshot_dir=snapshot_dir) as rec:
                rec.output = _agent("What is agentsnap?")
    print(f"[zero-instrumentation] snapshot written -> {name}.json")

    # -- Assert (same inputs -> should pass) -----------------------------------
    print("[zero-instrumentation] asserting (identical run)...")
    with mock.patch.object(_AnthMessages, "create", return_value=_FakeResp()):
        with PatchSet():
            with AgentAsserter(name, snapshot_dir=snapshot_dir) as a:
                a.output = _agent("What is agentsnap?")
    print("[zero-instrumentation] OK passed")

    # -- Simulate a regression -------------------------------------------------
    print("[zero-instrumentation] simulating regression (different LLM response)...")

    class _DriftedContent:
        text = "Completely different answer."

    class _DriftedResp:
        content = [_DriftedContent()]
        class usage:
            input_tokens = 5
            output_tokens = 10

    try:
        with mock.patch.object(_AnthMessages, "create", return_value=_DriftedResp()):
            with PatchSet():
                with AgentAsserter(name, snapshot_dir=snapshot_dir) as a:
                    a.output = _agent("What is agentsnap?")
    except AgentRegressionError as e:
        print(f"[zero-instrumentation] OK regression caught: {e.diff_report.failed_checks}")


def langgraph_demo(snapshot_dir: str) -> None:
    """Demonstrates node-level capture via callbacks — no langchain_core needed."""
    graph = LangGraphAdapter(_MockLangGraph("I'll look that up.", tool_name="lookup"))
    name = "demo_langgraph"

    # -- Record ----------------------------------------------------------------
    print("[langgraph] recording...")
    with AgentRecorder(name, snapshot_dir=snapshot_dir) as rec:
        result = graph.invoke("What is agentsnap?")
        rec.output = result
    print(f"[langgraph] snapshot written -> {name}.json")

    # -- Assert (same inputs -> should pass) -----------------------------------
    print("[langgraph] asserting (identical run)...")
    with AgentAsserter(name, snapshot_dir=snapshot_dir) as a:
        a.output = graph.invoke("What is agentsnap?")
    print("[langgraph] OK passed")

    # -- Simulate a regression (different node output) -------------------------
    print("[langgraph] simulating regression (changed LLM response in node)...")
    drifted_graph = LangGraphAdapter(_MockLangGraph("Completely different answer.", tool_name="lookup"))
    try:
        with AgentAsserter(name, snapshot_dir=snapshot_dir) as a:
            a.output = drifted_graph.invoke("What is agentsnap?")
    except AgentRegressionError as e:
        print(f"[langgraph] OK regression caught: {e.diff_report.failed_checks}")


# -- Main ----------------------------------------------------------------------

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
    print("Snapshot dir:", snap_dir)

    anthropic_demo(snap_dir)
    openai_demo(snap_dir)
    gemini_demo(snap_dir)
    cohere_demo(snap_dir)
    mistral_demo(snap_dir)
    groq_demo(snap_dir)
    langgraph_demo(snap_dir)
    zero_instrumentation_demo(snap_dir)

    header("All providers complete")
    snapshots = list(Path(snap_dir).glob("*.json"))
    print(f"Snapshots written: {len(snapshots)}")
    for p in sorted(snapshots):
        print(f"  {p.name}")
