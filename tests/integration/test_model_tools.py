"""End-to-end coverage: the model's own tool_use requests flow from the
AnthropicAdapter through recording, live assert, and replay — a tool-swap
requested by the MODEL (not a ToolAdapter call) fails the snapshot check.
"""
from __future__ import annotations

import json

import pytest

from agentsnap.adapters.anthropic import AnthropicAdapter
from agentsnap.core.asserter import AgentAsserter
from agentsnap.core.recorder import AgentRecorder
from agentsnap.core.snapshot import snapshot_path
from agentsnap.exceptions import AgentRegressionError
from tests.fixtures.mock_agents import MockAnthropicClient, MockAnthropicResponse


class ExplodingMessages:
    def create(self, **kwargs):
        raise AssertionError("live API called during replay")


class ExplodingClient:
    messages = ExplodingMessages()


def _identical_embed(texts):
    return [[1.0, 0.0] for _ in texts]


def ModelToolAgent(client, input_text: str) -> str:
    """Makes exactly one LLM call and returns a constant output. Any tool_use
    blocks the mock model attaches to its response are captured on the
    llm_call event as tool_requests — no ToolAdapter involved."""
    client.messages.create(
        model="claude-mock",
        messages=[{"role": "user", "content": input_text}],
        max_tokens=100,
    )
    return "the answer"


def _record_golden(tmp_path, tool_uses):
    client = AnthropicAdapter(MockAnthropicClient([MockAnthropicResponse("the answer", tool_uses=tool_uses)]))
    with AgentRecorder("model_tools_it", snapshot_dir=str(tmp_path)) as rec:
        rec.output = ModelToolAgent(client, "hello")


def test_identical_model_tool_requests_pass(tmp_path):
    _record_golden(tmp_path, [("search", {"q": "x"})])

    client = AnthropicAdapter(MockAnthropicClient([MockAnthropicResponse("the answer", tool_uses=[("search", {"q": "x"})])]))
    with AgentAsserter("model_tools_it", snapshot_dir=str(tmp_path), embed_fn=_identical_embed) as a:
        a.output = ModelToolAgent(client, "hello")


def test_model_tool_swap_raises_regression_with_model_tools_check(tmp_path):
    _record_golden(tmp_path, [("search", {"q": "x"})])

    client = AnthropicAdapter(
        MockAnthropicClient([MockAnthropicResponse("the answer", tool_uses=[("delete_file", {"path": "/etc"})])])
    )
    with pytest.raises(AgentRegressionError) as exc_info:
        with AgentAsserter("model_tools_it", snapshot_dir=str(tmp_path), embed_fn=_identical_embed) as a:
            a.output = ModelToolAgent(client, "hello")

    err = exc_info.value
    assert "model_tools" in err.diff_report.failed_checks
    assert "[MODEL TOOLS]" in str(err)


def test_model_tool_replay_round_trip_passes_and_carries_tool_requests(tmp_path):
    _record_golden(tmp_path, [("search", {"q": "x"})])

    replay_client = AnthropicAdapter(ExplodingClient())
    with AgentAsserter("model_tools_it", snapshot_dir=str(tmp_path), mode="replay") as a:
        a.output = ModelToolAgent(replay_client, "hello")
    assert a.output == "the answer"

    last_run_path = snapshot_path("model_tools_it", str(tmp_path)).parent / ".last_run" / "model_tools_it.json"
    data = json.loads(last_run_path.read_text(encoding="utf-8"))
    llm_calls = [e for e in data["trace"] if e["type"] == "llm_call"]
    assert len(llm_calls) == 1
    assert llm_calls[0]["tool_requests"] == [{"name": "search", "args": {"q": "x"}}]
