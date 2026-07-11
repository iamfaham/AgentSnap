from __future__ import annotations

import pytest

from agentsnap.adapters.anthropic import (
    AnthropicAdapter,
    extract_tool_requests as extract_anthropic_tool_requests,
)
from agentsnap.adapters.openai import (
    extract_tool_requests as extract_openai_tool_requests,
)
from agentsnap.core.recorder import AgentRecorder, TraceAccumulator, _accumulator_var
from agentsnap.core.replay import ReplaySession
from tests.fixtures.mock_agents import MockAnthropicClient, MockAnthropicResponse


# ── Anthropic extractor ─────────────────────────────────────────────────────

def test_anthropic_extract_text_only_returns_empty_list():
    resp = MockAnthropicResponse("hello")
    assert extract_anthropic_tool_requests(resp) == []


def test_anthropic_extract_two_tool_requests_in_order():
    resp = MockAnthropicResponse(
        "using tools",
        tool_uses=[("search", {"q": "x"}), ("fetch", {"url": "y"})],
    )
    result = extract_anthropic_tool_requests(resp)
    assert result == [
        {"name": "search", "args": {"q": "x"}},
        {"name": "fetch", "args": {"url": "y"}},
    ]


def test_anthropic_extract_missing_attrs_returns_empty_list_no_exception():
    class Bare:
        pass

    assert extract_anthropic_tool_requests(Bare()) == []


# ── OpenAI extractor ─────────────────────────────────────────────────────────

class _Function:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments


class _ToolCall:
    def __init__(self, name, arguments):
        self.function = _Function(name, arguments)


class _Message:
    def __init__(self, tool_calls=None):
        self.tool_calls = tool_calls


class _Choice:
    def __init__(self, message):
        self.message = message


class _Response:
    def __init__(self, choices):
        self.choices = choices


def test_openai_extract_text_only_returns_empty_list():
    resp = _Response([_Choice(_Message(tool_calls=None))])
    assert extract_openai_tool_requests(resp) == []


def test_openai_extract_two_tool_requests_in_order():
    resp = _Response(
        [
            _Choice(
                _Message(
                    tool_calls=[
                        _ToolCall("search", '{"q": "x"}'),
                        _ToolCall("fetch", '{"url": "y"}'),
                    ]
                )
            )
        ]
    )
    result = extract_openai_tool_requests(resp)
    assert result == [
        {"name": "search", "args": {"q": "x"}},
        {"name": "fetch", "args": {"url": "y"}},
    ]


def test_openai_extract_unparseable_arguments_keeps_raw_string():
    resp = _Response(
        [_Choice(_Message(tool_calls=[_ToolCall("search", "not json")]))]
    )
    result = extract_openai_tool_requests(resp)
    assert result == [{"name": "search", "args": "not json"}]


def test_openai_extract_missing_attrs_returns_empty_list_no_exception():
    class Bare:
        pass

    assert extract_openai_tool_requests(Bare()) == []


def test_openai_extract_no_choices_returns_empty_list():
    resp = _Response([])
    assert extract_openai_tool_requests(resp) == []


# ── End-to-end: AgentRecorder + AnthropicAdapter ────────────────────────────

def test_recorder_captures_tool_requests_on_llm_call_event(tmp_path):
    client = AnthropicAdapter(
        MockAnthropicClient(
            [MockAnthropicResponse("t", tool_uses=[("search", {"q": "x"})])]
        )
    )
    with AgentRecorder(
        "test_tool_requests_capture", snapshot_dir=str(tmp_path)
    ) as rec:
        client.messages.create(
            model="claude-mock",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=10,
        )
        rec.output = "done"

    event = rec.accumulator.trace[0]
    assert event["type"] == "llm_call"
    assert event["tool_requests"] == [{"name": "search", "args": {"q": "x"}}]


def test_recorder_records_empty_tool_requests_when_no_tools_used(tmp_path):
    client = AnthropicAdapter(MockAnthropicClient([MockAnthropicResponse("t")]))
    with AgentRecorder(
        "test_tool_requests_empty", snapshot_dir=str(tmp_path)
    ) as rec:
        client.messages.create(
            model="claude-mock",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=10,
        )
        rec.output = "done"

    event = rec.accumulator.trace[0]
    assert "tool_requests" in event
    assert event["tool_requests"] == []


# ── Replay pass-through ──────────────────────────────────────────────────────

class ExplodingMessages:
    def create(self, **kwargs):
        raise AssertionError("live API called during replay")


class ExplodingClient:
    messages = ExplodingMessages()


def _with_replay_acc(trace):
    acc = TraceAccumulator(model="unknown", replay=ReplaySession(trace))
    token = _accumulator_var.set(acc)
    return acc, token


def test_replay_carries_tool_requests_when_recorded():
    raw = MockAnthropicResponse(
        "recorded answer", tool_uses=[("search", {"q": "x"})]
    ).model_dump()
    trace = [
        {
            "step": 0,
            "type": "llm_call",
            "messages": [{"role": "user", "content": "q"}],
            "response": "recorded answer",
            "tokens": 30,
            "raw_response": raw,
            "tool_requests": [{"name": "search", "args": {"q": "x"}}],
        }
    ]
    acc, token = _with_replay_acc(trace)
    try:
        client = AnthropicAdapter(ExplodingClient())
        client.messages.create(
            model="m", messages=[{"role": "user", "content": "q"}], max_tokens=10
        )
    finally:
        _accumulator_var.reset(token)

    event = acc.trace[0]
    assert event["tool_requests"] == [{"name": "search", "args": {"q": "x"}}]


def test_replay_omits_tool_requests_key_for_old_format_events():
    raw = MockAnthropicResponse("recorded answer").model_dump()
    trace = [
        {
            "step": 0,
            "type": "llm_call",
            "messages": [{"role": "user", "content": "q"}],
            "response": "recorded answer",
            "tokens": 30,
            "raw_response": raw,
        }
    ]
    acc, token = _with_replay_acc(trace)
    try:
        client = AnthropicAdapter(ExplodingClient())
        client.messages.create(
            model="m", messages=[{"role": "user", "content": "q"}], max_tokens=10
        )
    finally:
        _accumulator_var.reset(token)

    event = acc.trace[0]
    assert "tool_requests" not in event


# ── model_dump validity against real anthropic SDK types ───────────────────

def test_mock_response_with_tool_use_model_dump_validates_against_real_sdk():
    anthropic_types = pytest.importorskip("anthropic.types")
    Message = anthropic_types.Message

    resp = MockAnthropicResponse("t", tool_uses=[("search", {"q": "x"})])
    dumped = resp.model_dump()
    msg = Message.model_validate(dumped)

    tool_blocks = [b for b in msg.content if b.type == "tool_use"]
    assert len(tool_blocks) == 1
    assert tool_blocks[0].name == "search"
    assert tool_blocks[0].input == {"q": "x"}
