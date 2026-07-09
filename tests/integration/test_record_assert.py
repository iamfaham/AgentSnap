from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path

from agentsnap.adapters.anthropic import AnthropicAdapter
from agentsnap.adapters.tool import ToolAdapter
from agentsnap.core.asserter import AgentAsserter
from agentsnap.core.recorder import AgentRecorder
from agentsnap.core.snapshot import input_sha8, snapshot_path
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

    # Output text differs by one character from the recorded run (via a tool fn
    # that appends "!") so it is not byte-identical — this keeps the exact-match
    # short-circuit in semantic_scores() from bypassing the orthogonal embed stub,
    # which simulates semantic drift for this test.
    def _search_fn_drifted(query: str) -> str:
        return f"{_search_fn(query)}!"

    with pytest.raises(AgentRegressionError) as exc_info:
        with AgentAsserter(
            name,
            snapshot_dir=snapshot_dir,
            semantic_threshold=0.92,
            embed_fn=_orthogonal_embed,
        ) as asserter:
            client2 = AnthropicAdapter(_simple_client())
            tool2 = ToolAdapter(_search_fn_drifted, name="search")
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

    # both thresholds = 0.0 means any similarity passes
    with AgentAsserter(
        name,
        snapshot_dir=snapshot_dir,
        semantic_threshold=0.0,
        llm_threshold=0.0,
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


def test_auto_records_on_first_miss(tmp_path):
    """AgentAsserter should auto-record when no snapshot exists, not raise."""
    snapshot_dir = str(tmp_path / "snaps")
    name = "auto_record"
    snap_file = Path(snapshot_dir) / f"{name}.json"

    assert not snap_file.exists()

    # First use: no snapshot → should auto-record, not raise
    with AgentAsserter(name, snapshot_dir=snapshot_dir, embed_fn=_identical_embed) as a:
        client = AnthropicAdapter(_simple_client())
        tool = ToolAdapter(_search_fn, name="search")
        a.output = SimpleToolAgent(client, tool, "hello")

    assert snap_file.exists(), "snapshot should have been written on first miss"


def test_auto_record_then_second_run_asserts(tmp_path):
    """After auto-record, a second run should assert against the written golden."""
    snapshot_dir = str(tmp_path / "snaps")
    name = "auto_then_assert"

    # First use: auto-records
    with AgentAsserter(name, snapshot_dir=snapshot_dir, embed_fn=_identical_embed) as a:
        client = AnthropicAdapter(_simple_client())
        tool = ToolAdapter(_search_fn, name="search")
        a.output = SimpleToolAgent(client, tool, "hello")

    # Second use: asserts (identical run → passes)
    with AgentAsserter(name, snapshot_dir=snapshot_dir, embed_fn=_identical_embed) as a:
        client = AnthropicAdapter(_simple_client())
        tool = ToolAdapter(_search_fn, name="search")
        a.output = SimpleToolAgent(client, tool, "hello")


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


# ── Scenario / input-binding tests ────────────────────────────────────────────

def test_recorder_explicit_scenario_namespaces_file(tmp_path):
    snap_dir = str(tmp_path)
    with AgentRecorder("agent", snapshot_dir=snap_dir, scenario="query_a") as r:
        r.output = "answer A"

    assert snapshot_path("agent", snap_dir, scenario="query_a").exists()
    assert not snapshot_path("agent", snap_dir).exists()


def test_recorder_auto_hash_from_input_data(tmp_path):
    snap_dir = str(tmp_path)
    inp = {"query": "what is 2+2?"}
    with AgentRecorder("agent", snapshot_dir=snap_dir) as r:
        r.input_data = inp
        r.output = "4"

    sha = input_sha8(inp)
    assert snapshot_path("agent", snap_dir, scenario=sha).exists()


def test_recorder_no_input_uses_plain_path(tmp_path):
    snap_dir = str(tmp_path)
    with AgentRecorder("agent", snapshot_dir=snap_dir) as r:
        r.output = "result"

    assert snapshot_path("agent", snap_dir).exists()


def test_asserter_explicit_scenario_roundtrip(tmp_path):
    snap_dir = str(tmp_path)
    with AgentRecorder("agent", snapshot_dir=snap_dir, scenario="s1") as r:
        r.output = "the answer"

    with AgentAsserter("agent", snapshot_dir=snap_dir, scenario="s1",
                       semantic_threshold=0.0, llm_threshold=0.0,
                       embed_fn=_identical_embed) as a:
        a.output = "the answer"
    # No exception = pass


def test_asserter_auto_hash_from_input_roundtrip(tmp_path):
    """a.input set inside the with block must drive scenario resolution."""
    snap_dir = str(tmp_path)
    inp = {"query": "test question"}

    # Record using explicit scenario matching what asserter will auto-hash to
    sha = input_sha8(inp)
    with AgentRecorder("agent", snapshot_dir=snap_dir, scenario=sha) as r:
        r.output = "test answer"

    # Assert: set a.input inside the with block; asserter auto-hashes it
    with AgentAsserter("agent", snapshot_dir=snap_dir,
                       semantic_threshold=0.0, llm_threshold=0.0,
                       embed_fn=_identical_embed) as a:
        a.input = inp
        a.output = "test answer"
    # No exception = pass


def test_input_binding_warning_on_mismatch(tmp_path, capsys):
    snap_dir = str(tmp_path)

    # Record with explicit scenario and original input stored in snapshot
    with AgentRecorder("agent", snapshot_dir=snap_dir, scenario="fixed") as r:
        r.input_data = {"q": "original query"}
        r.output = "result"

    # Assert with same scenario but set a.input to a different value
    with AgentAsserter("agent", snapshot_dir=snap_dir, scenario="fixed",
                       semantic_threshold=0.0, llm_threshold=0.0,
                       embed_fn=_identical_embed) as a:
        a.input = {"q": "completely different query"}
        a.output = "result"

    captured = capsys.readouterr()
    assert "WARNING" in captured.out
    assert "input changed" in captured.out


def test_input_binding_no_warning_when_inputs_match(tmp_path, capsys):
    snap_dir = str(tmp_path)

    with AgentRecorder("agent", snapshot_dir=snap_dir, scenario="fixed") as r:
        r.input_data = {"q": "same query"}
        r.output = "result"

    with AgentAsserter("agent", snapshot_dir=snap_dir, scenario="fixed",
                       semantic_threshold=0.0, llm_threshold=0.0,
                       embed_fn=_identical_embed) as a:
        a.input = {"q": "same query"}
        a.output = "result"

    captured = capsys.readouterr()
    assert "WARNING" not in captured.out
