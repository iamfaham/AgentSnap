from __future__ import annotations

import numpy as np
import pytest

from agentsnap.adapters.anthropic import AnthropicAdapter
from agentsnap.adapters.tool import ToolAdapter
from agentsnap.core.asserter import AgentAsserter
from agentsnap.core.recorder import AgentRecorder
from agentsnap.exceptions import AgentRegressionError
from tests.fixtures.mock_agents import (
    MockAnthropicClient,
    MockAnthropicResponse,
    MultiStepAgent,
    SimpleToolAgent,
)

# ── Embed stubs ────────────────────────────────────────────────────────────────

_DIM = 8


def _identical_embed(texts):
    v = np.ones(_DIM, dtype=float)
    v /= np.linalg.norm(v)
    return [v.copy() for _ in texts]


def _orthogonal_embed(texts):
    vecs = []
    for i in range(len(texts)):
        v = np.zeros(_DIM, dtype=float)
        v[i % _DIM] = 1.0
        vecs.append(v)
    return vecs


# ── Helpers ────────────────────────────────────────────────────────────────────

def _simple_client():
    return MockAnthropicClient([MockAnthropicResponse("I'll search for that.")])


def _multi_client():
    return MockAnthropicClient(
        [
            MockAnthropicResponse("I'll fetch that."),
            MockAnthropicResponse("Now I'll summarize."),
        ]
    )


def _search_fn(query: str) -> str:
    return f"search_result_for_{query}"


def _fetch_fn(query: str) -> str:
    return f"fetch_result_for_{query}"


def _summarize_fn(content: str) -> str:
    return f"summary_of_{content}"


# ── Tests ──────────────────────────────────────────────────────────────────────

def test_record_then_assert_passes(tmp_path):
    snapshot_dir = str(tmp_path / "snaps")
    name = "simple_pass"

    with AgentRecorder(name, snapshot_dir=snapshot_dir) as rec:
        client = AnthropicAdapter(_simple_client())
        tool = ToolAdapter(_search_fn, name="search")
        rec.output = SimpleToolAgent(client, tool, "hello")

    with AgentAsserter(name, snapshot_dir=snapshot_dir, embed_fn=_identical_embed) as asserter:
        client2 = AnthropicAdapter(_simple_client())
        tool2 = ToolAdapter(_search_fn, name="search")
        asserter.output = SimpleToolAgent(client2, tool2, "hello")


def test_mutated_tool_order_fails(tmp_path):
    snapshot_dir = str(tmp_path / "snaps")
    name = "multi_order"

    with AgentRecorder(name, snapshot_dir=snapshot_dir) as rec:
        client = AnthropicAdapter(_multi_client())
        fetch = ToolAdapter(_fetch_fn, name="fetch")
        summarize = ToolAdapter(_summarize_fn, name="summarize")
        rec.output = MultiStepAgent(client, fetch, summarize, "data")

    # Replay with reversed tool order: summarize → fetch
    def _rev_multi(client, fetch_tool, summarize_tool, input_text):
        client.messages.create(
            model="claude-mock",
            messages=[{"role": "user", "content": input_text}],
            max_tokens=100,
        )
        # Call summarize first (wrong order)
        s = summarize_tool(content=input_text)
        client.messages.create(
            model="claude-mock",
            messages=[{"role": "user", "content": input_text}],
            max_tokens=100,
        )
        f = fetch_tool(query=input_text)
        return f"{f} | {s}"

    with pytest.raises(AgentRegressionError) as exc_info:
        with AgentAsserter(name, snapshot_dir=snapshot_dir, embed_fn=_identical_embed) as asserter:
            client2 = AnthropicAdapter(_multi_client())
            fetch2 = ToolAdapter(_fetch_fn, name="fetch")
            summarize2 = ToolAdapter(_summarize_fn, name="summarize")
            asserter.output = _rev_multi(client2, fetch2, summarize2, "data")

    assert "structural" in exc_info.value.diff_report.failed_checks


def test_semantic_drift_above_threshold_fails(tmp_path):
    snapshot_dir = str(tmp_path / "snaps")
    name = "semantic_fail"

    with AgentRecorder(name, snapshot_dir=snapshot_dir) as rec:
        client = AnthropicAdapter(_simple_client())
        tool = ToolAdapter(_search_fn, name="search")
        rec.output = SimpleToolAgent(client, tool, "hello")

    with pytest.raises(AgentRegressionError) as exc_info:
        with AgentAsserter(
            name,
            snapshot_dir=snapshot_dir,
            semantic_threshold=0.92,
            embed_fn=_orthogonal_embed,
        ) as asserter:
            client2 = AnthropicAdapter(_simple_client())
            tool2 = ToolAdapter(_search_fn, name="search")
            asserter.output = SimpleToolAgent(client2, tool2, "hello")

    failed = exc_info.value.diff_report.failed_checks
    assert any("semantic" in f for f in failed)


def test_semantic_drift_below_threshold_passes(tmp_path):
    snapshot_dir = str(tmp_path / "snaps")
    name = "semantic_pass"

    with AgentRecorder(name, snapshot_dir=snapshot_dir) as rec:
        client = AnthropicAdapter(_simple_client())
        tool = ToolAdapter(_search_fn, name="search")
        rec.output = SimpleToolAgent(client, tool, "hello")

    # threshold = 0.0 means any similarity passes
    with AgentAsserter(
        name,
        snapshot_dir=snapshot_dir,
        semantic_threshold=0.0,
        embed_fn=_orthogonal_embed,
    ) as asserter:
        client2 = AnthropicAdapter(_simple_client())
        tool2 = ToolAdapter(_search_fn, name="search")
        asserter.output = SimpleToolAgent(client2, tool2, "hello")


def test_different_tool_args_detected(tmp_path):
    snapshot_dir = str(tmp_path / "snaps")
    name = "arg_drift"

    with AgentRecorder(name, snapshot_dir=snapshot_dir) as rec:
        client = AnthropicAdapter(_simple_client())
        tool = ToolAdapter(_search_fn, name="search")
        rec.output = SimpleToolAgent(client, tool, "hello")

    def _different_query(query: str) -> str:
        return f"result_for_different"

    with pytest.raises(AgentRegressionError) as exc_info:
        with AgentAsserter(name, snapshot_dir=snapshot_dir, embed_fn=_identical_embed) as asserter:
            client2 = AnthropicAdapter(_simple_client())
            # Wrap search with a different query keyword arg to cause arg drift
            def _search_different(query: str) -> str:
                return f"result_for_{query}"

            class _ForcedQueryTool:
                def __call__(self, **kwargs):
                    acc = __import__("agentsnap.core.recorder", fromlist=["TraceAccumulator"]).TraceAccumulator.current()
                    result = _search_different(query="different_query")
                    if acc:
                        acc.push({"type": "tool_call", "name": "search", "args": {"query": "different_query"}, "result": str(result)})
                    return result

            tool2 = _ForcedQueryTool()
            client2.messages.create(
                model="claude-mock",
                messages=[{"role": "user", "content": "hello"}],
                max_tokens=100,
            )
            asserter.output = f"Result: {tool2()}"

    assert "arguments" in exc_info.value.diff_report.failed_checks


def test_snapshot_not_found_raises(tmp_path):
    from agentsnap.exceptions import SnapshotNotFoundError

    with pytest.raises(SnapshotNotFoundError):
        with AgentAsserter("no_such_snapshot", snapshot_dir=str(tmp_path)):
            pass


def test_last_run_written_on_assert(tmp_path):
    from agentsnap.core.snapshot import last_run_path

    snapshot_dir = str(tmp_path / "snaps")
    name = "last_run_test"

    with AgentRecorder(name, snapshot_dir=snapshot_dir) as rec:
        client = AnthropicAdapter(_simple_client())
        tool = ToolAdapter(_search_fn, name="search")
        rec.output = SimpleToolAgent(client, tool, "q")

    try:
        with AgentAsserter(name, snapshot_dir=snapshot_dir, embed_fn=_identical_embed) as asserter:
            client2 = AnthropicAdapter(_simple_client())
            tool2 = ToolAdapter(_search_fn, name="search")
            asserter.output = SimpleToolAgent(client2, tool2, "q")
    except AgentRegressionError:
        pass

    assert last_run_path(name, snapshot_dir).exists()
