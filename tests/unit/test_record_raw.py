from agentsnap.adapters.anthropic import AnthropicAdapter, dump_raw
from agentsnap.core.recorder import AgentRecorder
from agentsnap.core.snapshot import read_snapshot
from tests.fixtures.mock_agents import MockAnthropicClient, MockAnthropicResponse


def test_dump_raw_uses_model_dump():
    resp = MockAnthropicResponse("hello")
    raw = dump_raw(resp)
    assert raw is not None
    assert raw["content"][0]["text"] == "hello"
    assert raw["role"] == "assistant"


def test_dump_raw_returns_none_without_model_dump():
    class Bare:
        pass
    assert dump_raw(Bare()) is None


def test_dump_raw_returns_none_when_model_dump_raises():
    class Bad:
        def model_dump(self, mode="python"):
            raise RuntimeError("boom")
    assert dump_raw(Bad()) is None


def test_dump_raw_returns_none_when_fallback_raises():
    class Bad:
        def model_dump(self):  # no mode kwarg -> TypeError, then fallback raises
            raise RuntimeError("boom")
    assert dump_raw(Bad()) is None


def test_recorder_captures_raw_response(tmp_path):
    client = AnthropicAdapter(MockAnthropicClient([MockAnthropicResponse("hi there")]))
    with AgentRecorder("raw_rec", snapshot_dir=str(tmp_path)) as rec:
        client.messages.create(model="m", messages=[{"role": "user", "content": "q"}], max_tokens=10)
        rec.output = "done"
    snap = read_snapshot("raw_rec", str(tmp_path))
    llm = [e for e in snap["trace"] if e["type"] == "llm_call"][0]
    assert llm["raw_response"]["content"][0]["text"] == "hi there"
