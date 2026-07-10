import pytest

from agentsnap.adapters.anthropic import AnthropicAdapter
from agentsnap.adapters.tool import ToolAdapter
from agentsnap.core.asserter import AgentAsserter
from agentsnap.core.recorder import AgentRecorder
from agentsnap.core.snapshot import snapshot_path
from agentsnap.exceptions import AgentRegressionError, ReplayError, SnapshotFormatError
from tests.fixtures.mock_agents import (
    MockAnthropicClient,
    MockAnthropicResponse,
    SimpleToolAgent,
)


# ── Streaming round trip: record streamed chunks, replay as real SDK objects ────

class FakeAnthDelta:
    def __init__(self, text=None):
        self.text = text


class FakeAnthUsage:
    def __init__(self, input_tokens=0, output_tokens=0):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class FakeAnthMessage:
    def __init__(self, usage):
        self.usage = usage


class FakeAnthStreamEvent:
    def __init__(self, type, message=None, delta=None, usage=None):
        self.type = type
        self.message = message
        self.delta = delta
        self.usage = usage

    def model_dump(self, mode="json"):
        d = {"type": self.type}
        if self.message is not None:
            d["message"] = {
                "id": "msg_mock", "type": "message", "role": "assistant",
                "model": "claude-mock", "content": [], "stop_reason": None,
                "stop_sequence": None,
                "usage": {
                    "input_tokens": self.message.usage.input_tokens,
                    "output_tokens": self.message.usage.output_tokens,
                },
            }
        if self.delta is not None:
            d["index"] = 0
            d["delta"] = {"type": "text_delta", "text": self.delta.text}
        if self.usage is not None:
            d["delta"] = {"stop_reason": "end_turn", "stop_sequence": None}
            d["usage"] = {
                "input_tokens": self.usage.input_tokens,
                "output_tokens": self.usage.output_tokens,
            }
        return d


def _stream_events(text_parts):
    events = [FakeAnthStreamEvent("message_start", message=FakeAnthMessage(FakeAnthUsage(11, 0)))]
    for part in text_parts:
        events.append(FakeAnthStreamEvent("content_block_delta", delta=FakeAnthDelta(part)))
    events.append(FakeAnthStreamEvent("message_delta", usage=FakeAnthUsage(0, 9)))
    return events


class FakeAnthStream:
    def __init__(self, events):
        self._events = events

    def __iter__(self):
        return iter(self._events)

    def close(self):
        pass


class StreamingMessages:
    def __init__(self, events):
        self._events = events

    def create(self, **kwargs):
        assert kwargs.get("stream") is True
        return FakeAnthStream(self._events)


class StreamingClient:
    def __init__(self, events):
        self.messages = StreamingMessages(events)


def StreamingAgent(client, input_text: str) -> str:
    """Streams one LLM call and joins the delta text as its output."""
    stream = client.messages.create(
        model="claude-mock",
        messages=[{"role": "user", "content": input_text}],
        max_tokens=100,
        stream=True,
    )
    parts = []
    for event in stream:
        if getattr(event, "type", "") == "content_block_delta":
            parts.append(event.delta.text)
    return "".join(parts)


def test_replay_streams_recorded_chunks_as_real_sdk_objects_without_live_call(tmp_path):
    client = AnthropicAdapter(StreamingClient(_stream_events(["Hello, ", "world!"])))
    with AgentRecorder("replay_stream_it", snapshot_dir=str(tmp_path)) as rec:
        rec.output = StreamingAgent(client, "hello")
    assert rec.output == "Hello, world!"

    replay_client = AnthropicAdapter(ExplodingClient())
    with AgentAsserter("replay_stream_it", snapshot_dir=str(tmp_path), mode="replay") as a:
        a.output = StreamingAgent(replay_client, "hello")
    assert a.output == "Hello, world!"


class ExplodingMessages:
    def create(self, **kwargs):
        raise AssertionError("live API called during replay")


class ExplodingClient:
    messages = ExplodingMessages()


def _record_golden(tmp_path, text="the answer"):
    client = AnthropicAdapter(MockAnthropicClient([MockAnthropicResponse(text)]))
    tool = ToolAdapter(lambda query: f"results for {query}", name="search")
    with AgentRecorder("replay_it", snapshot_dir=str(tmp_path)) as rec:
        rec.output = SimpleToolAgent(client, tool, "hello")
    return tool


def test_replay_round_trip_passes_without_live_calls(tmp_path):
    _record_golden(tmp_path)
    tool = ToolAdapter(lambda query: f"results for {query}", name="search")
    client = AnthropicAdapter(ExplodingClient())
    with AgentAsserter("replay_it", snapshot_dir=str(tmp_path), mode="replay") as a:
        a.output = SimpleToolAgent(client, tool, "hello")


def test_replay_fails_on_prompt_change(tmp_path):
    _record_golden(tmp_path)
    tool = ToolAdapter(lambda query: f"results for {query}", name="search")
    client = AnthropicAdapter(ExplodingClient())
    with pytest.raises(AgentRegressionError) as exc_info:
        with AgentAsserter("replay_it", snapshot_dir=str(tmp_path), mode="replay") as a:
            a.output = SimpleToolAgent(client, tool, "DIFFERENT INPUT")
    assert "llm_requests" in exc_info.value.diff_report.failed_checks


def test_replay_extra_llm_call_raises_replay_error(tmp_path):
    _record_golden(tmp_path)
    client = AnthropicAdapter(ExplodingClient())
    with pytest.raises(ReplayError, match="more LLM calls"):
        with AgentAsserter("replay_it", snapshot_dir=str(tmp_path), mode="replay") as a:
            client.messages.create(model="m", messages=[{"role": "user", "content": "hello"}], max_tokens=10)
            client.messages.create(model="m", messages=[{"role": "user", "content": "again"}], max_tokens=10)
            a.output = "x"


def test_replay_on_v10_snapshot_raises_format_error(tmp_path):
    _record_golden(tmp_path)
    # Strip raw_response to simulate a v1.0 file
    import json
    path = snapshot_path("replay_it", str(tmp_path))
    data = json.loads(path.read_text(encoding="utf-8"))
    for event in data["trace"]:
        event.pop("raw_response", None)
    data["version"] = "1.0"
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(SnapshotFormatError, match="agentsnap-record"):
        with AgentAsserter("replay_it", snapshot_dir=str(tmp_path), mode="replay"):
            pass


def test_replay_missing_snapshot_auto_records_live(tmp_path):
    client = AnthropicAdapter(MockAnthropicClient([MockAnthropicResponse("first run")]))
    tool = ToolAdapter(lambda query: "r", name="search")
    with AgentAsserter("brand_new", snapshot_dir=str(tmp_path), mode="replay") as a:
        a.output = SimpleToolAgent(client, tool, "hello")
    assert snapshot_path("brand_new", str(tmp_path)).exists()


def test_invalid_mode_raises_value_error(tmp_path):
    with pytest.raises(ValueError, match="mode"):
        AgentAsserter("x", snapshot_dir=str(tmp_path), mode="cassette")


def test_replay_scenario_not_found_at_enter_raises_instead_of_running_live(tmp_path):
    """Regression: if the scenario snapshot only exists at an input-derived path that
    isn't known until inside the with-block, replay mode must not silently fall
    through to a live run — it must raise ReplayError instead."""
    input_data = {"query": "hello"}

    # Record a golden that lands at an input-hash scenario path (no explicit scenario).
    client = AnthropicAdapter(MockAnthropicClient([MockAnthropicResponse("the answer")]))
    tool = ToolAdapter(lambda query: f"results for {query}", name="search")
    with AgentRecorder("replay_scenario_it", snapshot_dir=str(tmp_path)) as rec:
        rec.input_data = input_data
        rec.output = SimpleToolAgent(client, tool, "hello")

    # Sanity: the golden was written under an input-hash-suffixed filename, not the bare name.
    from agentsnap.core.snapshot import input_sha8
    scenario = input_sha8(input_data)
    assert snapshot_path("replay_scenario_it", str(tmp_path), scenario=scenario).exists()
    assert not snapshot_path("replay_scenario_it", str(tmp_path)).exists()

    live_client = AnthropicAdapter(MockAnthropicClient([MockAnthropicResponse("the answer")]))
    live_tool = ToolAdapter(lambda query: f"results for {query}", name="search")
    with pytest.raises(ReplayError, match="scenario"):
        with AgentAsserter("replay_scenario_it", snapshot_dir=str(tmp_path), mode="replay") as a:
            a.input = input_data
            a.output = SimpleToolAgent(live_client, live_tool, "hello")


def test_replay_tools_stubs_tool_results(tmp_path):
    _record_golden(tmp_path)

    def exploding_tool(query):
        raise AssertionError("real tool executed")

    tool = ToolAdapter(exploding_tool, name="search")
    client = AnthropicAdapter(ExplodingClient())
    with AgentAsserter("replay_it", snapshot_dir=str(tmp_path), mode="replay", replay_tools=True) as a:
        a.output = SimpleToolAgent(client, tool, "hello")
