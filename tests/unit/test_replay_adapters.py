import pytest

from agentsnap.adapters.anthropic import AnthropicAdapter, reconstruct as reconstruct_anthropic
from agentsnap.core.recorder import TraceAccumulator, _accumulator_var
from agentsnap.core.replay import ReplaySession
from agentsnap.exceptions import ReplayError
from tests.fixtures.mock_agents import MockAnthropicClient, MockAnthropicResponse


RAW = MockAnthropicResponse("recorded answer").model_dump()

GOLDEN_TRACE = [
    {"step": 0, "type": "llm_call", "messages": [{"role": "user", "content": "q"}],
     "response": "recorded answer", "tokens": 30, "raw_response": RAW},
]


class ExplodingMessages:
    def create(self, **kwargs):
        raise AssertionError("live API called during replay")


class ExplodingClient:
    messages = ExplodingMessages()


def _with_replay_acc(trace):
    acc = TraceAccumulator(model="unknown", replay=ReplaySession(trace))
    token = _accumulator_var.set(acc)
    return acc, token


def test_reconstruct_returns_real_anthropic_message():
    msg = reconstruct_anthropic(RAW)
    assert msg.content[0].text == "recorded answer"
    assert msg.role == "assistant"


def test_adapter_replay_does_not_call_sdk():
    acc, token = _with_replay_acc(GOLDEN_TRACE)
    try:
        client = AnthropicAdapter(ExplodingClient())
        resp = client.messages.create(model="m", messages=[{"role": "user", "content": "q"}], max_tokens=10)
        assert resp.content[0].text == "recorded answer"
    finally:
        _accumulator_var.reset(token)


def test_adapter_replay_pushes_observed_messages():
    acc, token = _with_replay_acc(GOLDEN_TRACE)
    try:
        client = AnthropicAdapter(ExplodingClient())
        client.messages.create(model="m", messages=[{"role": "user", "content": "CHANGED"}], max_tokens=10)
    finally:
        _accumulator_var.reset(token)
    event = acc.trace[0]
    assert event["type"] == "llm_call"
    assert event["messages"] == [{"role": "user", "content": "CHANGED"}]
    assert event["response"] == "recorded answer"


def test_adapter_replay_exhausted_raises():
    acc, token = _with_replay_acc(GOLDEN_TRACE)
    try:
        client = AnthropicAdapter(ExplodingClient())
        client.messages.create(model="m", messages=[], max_tokens=10)
        with pytest.raises(ReplayError, match="more LLM calls"):
            client.messages.create(model="m", messages=[], max_tokens=10)
    finally:
        _accumulator_var.reset(token)


def test_adapter_passthrough_without_accumulator():
    client = AnthropicAdapter(MockAnthropicClient([MockAnthropicResponse("live")]))
    resp = client.messages.create(model="m", messages=[], max_tokens=10)
    assert resp.content[0].text == "live"
