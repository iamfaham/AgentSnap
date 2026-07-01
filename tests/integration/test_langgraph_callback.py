from __future__ import annotations

import pytest

from agentsnap.adapters.langgraph import AgentSnapCallback, LangGraphAdapter
from agentsnap.adapters.anthropic import AnthropicAdapter
from agentsnap.adapters.tool import ToolAdapter
from agentsnap.core.recorder import AgentRecorder, TraceAccumulator, _accumulator_var
from agentsnap.core.asserter import AgentAsserter
from tests.fixtures.mock_agents import MockAnthropicClient, MockAnthropicResponse

import numpy as np

_DIM = 8

def _identical_embed(texts):
    v = np.ones(_DIM, dtype=float)
    v /= np.linalg.norm(v)
    return [v.copy() for _ in texts]


# ── Mock LangChain result (duck-typed, no langchain_core import needed) ──────

class _Gen:
    def __init__(self, text: str):
        self.text = text

class _LLMResult:
    def __init__(self, text: str):
        self.generations = [[_Gen(text)]]


# ── Mock graph that fires callbacks (simulates LangGraph runtime) ─────────────

class _MockGraph:
    """Fake compiled graph: fires on_llm_end then on_tool_end on each callback."""

    def __init__(self, llm_text: str = "I'll look that up.", tool_name: str = "search"):
        self._llm_text = llm_text
        self._tool_name = tool_name

    def invoke(self, input_data, config=None, **kwargs):
        for cb in (config or {}).get("callbacks", []):
            if hasattr(cb, "on_llm_end"):
                cb.on_llm_end(_LLMResult(self._llm_text))
            if hasattr(cb, "on_tool_end"):
                cb.on_tool_end("mock tool result", name=self._tool_name)
        return f"Final: {self._llm_text}"

    def stream(self, input_data, **kwargs):
        return iter([])


# ── Unit: AgentSnapCallback pushes correct event types ───────────────────────

def test_callback_on_llm_end_pushes_llm_call_event():
    acc = TraceAccumulator()
    token = _accumulator_var.set(acc)
    try:
        cb = AgentSnapCallback()
        cb.on_llm_end(_LLMResult("hello world"))
    finally:
        _accumulator_var.reset(token)

    events = acc.trace
    assert len(events) == 1
    assert events[0]["type"] == "llm_call"
    assert events[0]["response"] == "hello world"


def test_callback_on_tool_end_pushes_tool_call_event():
    acc = TraceAccumulator()
    token = _accumulator_var.set(acc)
    try:
        cb = AgentSnapCallback()
        cb.on_tool_end("search result", name="my_tool")
    finally:
        _accumulator_var.reset(token)

    events = acc.trace
    assert len(events) == 1
    assert events[0]["type"] == "tool_call"
    assert events[0]["name"] == "my_tool"
    assert events[0]["result"] == "search result"


def test_callback_noop_outside_accumulator_context():
    """Callbacks must be no-ops when no TraceAccumulator is active."""
    assert TraceAccumulator.current() is None
    cb = AgentSnapCallback()
    cb.on_llm_end(_LLMResult("text"))  # should not raise
    cb.on_tool_end("result", name="tool")  # should not raise


# ── Integration: LangGraphAdapter injects callback into invoke ────────────────

def test_langgraph_adapter_captures_node_events(tmp_path):
    """LangGraphAdapter must capture LLM + tool events from inside the graph."""
    graph = LangGraphAdapter(_MockGraph(llm_text="Node response", tool_name="lookup"))
    snap_dir = str(tmp_path / "snaps")

    with AgentRecorder("lg_node_events", snapshot_dir=snap_dir) as rec:
        result = graph.invoke("What is agentsnap?")
        rec.output = result

    import json
    data = json.loads((tmp_path / "snaps" / "lg_node_events.json").read_text())
    trace = data["trace"]

    llm_events  = [e for e in trace if e["type"] == "llm_call"]
    tool_events = [e for e in trace if e["type"] == "tool_call"]
    assert len(llm_events)  == 1, f"Expected 1 llm_call, got {llm_events}"
    assert len(tool_events) == 1, f"Expected 1 tool_call, got {tool_events}"
    assert llm_events[0]["response"] == "Node response"
    assert tool_events[0]["name"] == "lookup"


def test_langgraph_adapter_record_then_assert_passes(tmp_path):
    """Full record → assert cycle with a mock LangGraph graph."""
    graph = LangGraphAdapter(_MockGraph())
    snap_dir = str(tmp_path / "snaps")

    with AgentRecorder("lg_cycle", snapshot_dir=snap_dir) as rec:
        rec.output = graph.invoke("hello")

    with AgentAsserter("lg_cycle", snapshot_dir=snap_dir, embed_fn=_identical_embed) as a:
        a.output = graph.invoke("hello")


def test_langgraph_adapter_passthrough_without_accumulator():
    """LangGraphAdapter must be transparent when no recorder is active."""
    graph = LangGraphAdapter(_MockGraph(llm_text="direct"))
    assert TraceAccumulator.current() is None
    result = graph.invoke("test input")
    assert result == "Final: direct"


def test_langgraph_adapter_passes_existing_callbacks(tmp_path):
    """User-supplied callbacks in config must be preserved, not replaced."""
    seen = []

    class _Spy:
        def on_llm_end(self, response, **kwargs):
            seen.append("spy_llm")
        def on_tool_end(self, output, *, name="", **kwargs):
            seen.append(f"spy_tool:{name}")

    graph = LangGraphAdapter(_MockGraph())
    snap_dir = str(tmp_path / "snaps")

    with AgentRecorder("lg_spy", snapshot_dir=snap_dir) as rec:
        rec.output = graph.invoke("q", config={"callbacks": [_Spy()]})

    assert "spy_llm" in seen
    assert "spy_tool:search" in seen


import uuid as _uuid

class _MockGraphWithArgs:
    """Fires on_tool_start then on_tool_end with matching run_id."""

    def invoke(self, input_data, config=None, **kwargs):
        run_id = str(_uuid.uuid4())
        for cb in (config or {}).get("callbacks", []):
            if hasattr(cb, "on_tool_start"):
                cb.on_tool_start(
                    {"name": "search"},
                    '{"query": "agentsnap docs"}',
                    run_id=run_id,
                )
            if hasattr(cb, "on_tool_end"):
                cb.on_tool_end("Found 3 results.", name="search", run_id=run_id)
        return "Done"

    def stream(self, input_data, **kwargs):
        return iter([])


def test_callback_captures_tool_args_via_on_tool_start(tmp_path):
    graph = LangGraphAdapter(_MockGraphWithArgs())
    snap_dir = str(tmp_path / "snaps")

    with AgentRecorder("lg_args", snapshot_dir=snap_dir) as rec:
        rec.output = graph.invoke("query")

    import json
    data = json.loads((tmp_path / "snaps" / "lg_args.json").read_text())
    tool_events = [e for e in data["trace"] if e["type"] == "tool_call"]
    assert len(tool_events) == 1
    assert tool_events[0]["args"] == {"query": "agentsnap docs"}
    assert tool_events[0]["result"] == "Found 3 results."


def test_tool_end_without_prior_start_still_records_event(tmp_path):
    """on_tool_end must not crash if on_tool_start was never called."""
    graph = LangGraphAdapter(_MockGraph())  # fires on_tool_end without on_tool_start
    snap_dir = str(tmp_path / "snaps")
    with AgentRecorder("lg_no_start", snapshot_dir=snap_dir) as rec:
        rec.output = graph.invoke("query")
    import json
    data = json.loads((tmp_path / "snaps" / "lg_no_start.json").read_text())
    tool_events = [e for e in data["trace"] if e["type"] == "tool_call"]
    assert len(tool_events) == 1
    assert tool_events[0]["args"] == {}


import asyncio


class _MockAsyncGraph:
    """Async graph that fires callbacks synchronously (mirrors real LangGraph behavior)."""

    def __init__(self, llm_text: str = "async response", tool_name: str = "async_tool"):
        self._llm_text = llm_text
        self._tool_name = tool_name

    async def ainvoke(self, input_data, config=None, **kwargs):
        import uuid as _uuid
        run_id = str(_uuid.uuid4())
        for cb in (config or {}).get("callbacks", []):
            if hasattr(cb, "on_tool_start"):
                cb.on_tool_start(
                    {"name": self._tool_name},
                    '{"q": "test"}',
                    run_id=run_id,
                )
            if hasattr(cb, "on_llm_end"):
                cb.on_llm_end(_LLMResult(self._llm_text))
            if hasattr(cb, "on_tool_end"):
                cb.on_tool_end("async result", name=self._tool_name, run_id=run_id)
        return f"Final: {self._llm_text}"

    def stream(self, *a, **kw):
        return iter([])

    async def astream(self, *a, **kw):
        return
        yield  # make it an async generator


def test_langgraph_adapter_ainvoke_captures_trace(tmp_path):
    """ainvoke must capture LLM + tool events just like sync invoke."""
    async def _run():
        graph = LangGraphAdapter(_MockAsyncGraph())
        snap_dir = str(tmp_path / "snaps_async")
        with AgentRecorder("lg_async", snapshot_dir=snap_dir) as rec:
            rec.output = await graph.ainvoke("What is agentsnap?")

    asyncio.run(_run())

    import json
    data = json.loads((tmp_path / "snaps_async" / "lg_async.json").read_text())
    trace = data["trace"]
    assert any(e["type"] == "llm_call" for e in trace)
    assert any(e["type"] == "tool_call" for e in trace)
    tool_events = [e for e in trace if e["type"] == "tool_call"]
    assert tool_events[0]["args"] == {"q": "test"}


def test_langgraph_adapter_ainvoke_passthrough_without_accumulator():
    """ainvoke must be transparent when no recorder is active."""
    async def _run():
        graph = LangGraphAdapter(_MockAsyncGraph(llm_text="direct"))
        assert TraceAccumulator.current() is None
        result = await graph.ainvoke("input")
        assert result == "Final: direct"

    asyncio.run(_run())


def test_async_context_manager_with_recorder(tmp_path):
    """async with AgentRecorder must work in async test code."""
    async def _run():
        graph = LangGraphAdapter(_MockAsyncGraph())
        snap_dir = str(tmp_path / "snaps_actx")
        async with AgentRecorder("lg_async_ctx", snapshot_dir=snap_dir) as rec:
            rec.output = await graph.ainvoke("question")

    asyncio.run(_run())
    import json
    snap = json.loads((tmp_path / "snaps_actx" / "lg_async_ctx.json").read_text())
    assert snap["output"].startswith("Final:")
