import pytest

from agentsnap.adapters.anthropic import AnthropicAdapter, reconstruct as reconstruct_anthropic
from agentsnap.adapters.openai import OpenAIAdapter
from agentsnap.core.recorder import TraceAccumulator, _accumulator_var
from agentsnap.core.replay import ReplaySession
from agentsnap.exceptions import ReplayError
from tests.fixtures.mock_agents import MockAnthropicClient, MockAnthropicResponse


RAW = MockAnthropicResponse("recorded answer").model_dump()

GOLDEN_TRACE = [
    {"step": 0, "type": "llm_call", "messages": [{"role": "user", "content": "q"}],
     "response": "recorded answer", "tokens": 30, "raw_response": RAW},
]

CORRUPT_TRACE = [
    {"step": 0, "type": "llm_call", "messages": [{"role": "user", "content": "q"}],
     "response": "recorded answer", "tokens": 30, "raw_response": {"nonsense": True}},
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


def test_tool_adapter_executes_real_tool_in_replay_by_default():
    from agentsnap.adapters.tool import ToolAdapter
    trace = GOLDEN_TRACE + [
        {"step": 1, "type": "tool_call", "name": "search", "args": {"q": "x"}, "result": "recorded result"},
    ]
    acc = TraceAccumulator(replay=ReplaySession(trace, replay_tools=False))
    token = _accumulator_var.set(acc)
    try:
        tool = ToolAdapter(lambda **kw: "real result", name="search")
        assert tool(q="x") == "real result"
    finally:
        _accumulator_var.reset(token)


def test_tool_adapter_replays_recorded_result_when_enabled():
    from agentsnap.adapters.tool import ToolAdapter
    trace = GOLDEN_TRACE + [
        {"step": 1, "type": "tool_call", "name": "search", "args": {"q": "x"}, "result": "recorded result"},
    ]
    acc = TraceAccumulator(replay=ReplaySession(trace, replay_tools=True))
    token = _accumulator_var.set(acc)
    try:
        def explode(**kw):
            raise AssertionError("real tool executed during replay_tools")
        tool = ToolAdapter(explode, name="search")
        assert tool(q="x") == "recorded result"
        assert acc.trace[0]["result"] == "recorded result"
    finally:
        _accumulator_var.reset(token)


def test_tool_adapter_replay_wrong_name_raises():
    from agentsnap.adapters.tool import ToolAdapter
    trace = [{"step": 0, "type": "tool_call", "name": "search", "args": {}, "result": "r"}]
    acc = TraceAccumulator(replay=ReplaySession(trace, replay_tools=True))
    token = _accumulator_var.set(acc)
    try:
        tool = ToolAdapter(lambda **kw: "x", name="fetch")
        with pytest.raises(ReplayError, match="expected 'search'"):
            tool()
    finally:
        _accumulator_var.reset(token)


def test_anthropic_adapter_replay_corrupt_raw_response_raises_clear_replay_error():
    acc, token = _with_replay_acc(CORRUPT_TRACE)
    try:
        client = AnthropicAdapter(ExplodingClient())
        with pytest.raises(ReplayError, match="reconstruct") as exc_info:
            client.messages.create(model="m", messages=[{"role": "user", "content": "q"}], max_tokens=10)
        assert "agentsnap-record" in str(exc_info.value)
    finally:
        _accumulator_var.reset(token)


def test_openai_adapter_replay_corrupt_raw_response_raises_clear_replay_error():
    class ExplodingCompletions:
        def create(self, **kwargs):
            raise AssertionError("live API called during replay")

    class ExplodingChat:
        completions = ExplodingCompletions()

    class ExplodingOpenAIClient:
        chat = ExplodingChat()

    acc = TraceAccumulator(model="unknown", replay=ReplaySession(CORRUPT_TRACE))
    token = _accumulator_var.set(acc)
    try:
        client = OpenAIAdapter(ExplodingOpenAIClient())
        with pytest.raises(ReplayError, match="reconstruct") as exc_info:
            client.chat.completions.create(model="m", messages=[{"role": "user", "content": "q"}])
        assert "agentsnap-record" in str(exc_info.value)
    finally:
        _accumulator_var.reset(token)
