"""
demo_streaming.py -- Streaming agents: record by teeing, replay from recorded chunks.

  python examples/demo_streaming.py

No API keys required. Uses a fake Anthropic-shaped streaming client whose
events are valid `anthropic.types.RawMessageStreamEvent` dicts, so the
recorded snapshot can be replayed back into real SDK objects.

The journey:
  1. Record an agent that consumes a live `stream=True` call. The adapter
     tees the stream: chunks flow through to the agent unmodified while the
     assembled response is recorded for replay.
  2. Replay -- the "network" is disabled. The recorded chunks are rebuilt as
     real anthropic stream-event objects and yielded back incrementally,
     producing the same output with ZERO live API calls.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent.parent))

from agentsnap.adapters.anthropic import AnthropicAdapter
from agentsnap.core.asserter import AgentAsserter
from agentsnap.core.recorder import AgentRecorder

SEP = "=" * 70


def header(title: str) -> None:
    print(f"\n{SEP}\n  {title}\n{SEP}")


# ── A fake Anthropic-shaped streaming client ───────────────────────────────

def _event_dicts() -> list[dict]:
    """Valid anthropic RawMessageStreamEvent-shaped dicts."""
    return [
        {
            "type": "message_start",
            "message": {
                "id": "msg_1", "type": "message", "role": "assistant",
                "model": "claude-mock", "content": [], "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 11, "output_tokens": 0},
            },
        },
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Python is "}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "a great language."}},
        {"type": "content_block_stop", "index": 0},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": 9},
        },
        {"type": "message_stop"},
    ]


class FakeStreamEvent:
    """Mimics a real anthropic stream event: real attributes + model_dump()."""

    def __init__(self, raw: dict) -> None:
        self._raw = raw
        self.type = raw["type"]
        if self.type == "message_start":
            usage = raw["message"]["usage"]
            self.message = SimpleNamespace(usage=SimpleNamespace(input_tokens=usage["input_tokens"]))
        elif self.type == "content_block_delta":
            self.delta = SimpleNamespace(text=raw["delta"]["text"])
        elif self.type == "message_delta":
            self.usage = SimpleNamespace(output_tokens=raw["usage"]["output_tokens"])

    def model_dump(self, mode: str = "json") -> dict:
        return self._raw


class FakeAnthropicStream:
    def __init__(self, events: list[FakeStreamEvent]) -> None:
        self._events = events
        self.closed = False

    def __iter__(self):
        return iter(self._events)

    def close(self) -> None:
        self.closed = True


class FakeMessages:
    def __init__(self, stream: FakeAnthropicStream) -> None:
        self._stream = stream

    def create(self, **kwargs):
        assert kwargs.get("stream") is True, "demo expects a streaming call"
        return self._stream


class FakeAnthropicClient:
    def __init__(self, stream: FakeAnthropicStream) -> None:
        self.messages = FakeMessages(stream)


class NetworkDisabledMessages:
    def create(self, **kwargs):
        raise RuntimeError("NETWORK CALL ATTEMPTED -- replay mode should never do this!")


class NetworkDisabledClient:
    messages = NetworkDisabledMessages()


def my_streaming_agent(client, question: str) -> str:
    """A tiny agent that consumes a live/replayed stream chunk by chunk."""
    chunks_seen = 0
    text_parts: list[str] = []
    stream = client.messages.create(
        model="claude-mock",
        messages=[{"role": "user", "content": question}],
        max_tokens=100,
        stream=True,
    )
    for event in stream:
        chunks_seen += 1
        if event.type == "content_block_delta":
            text_parts.append(event.delta.text)
    print(f"    ({chunks_seen} chunks arrived incrementally)")
    return "".join(text_parts)


def main() -> None:
    snapshot_dir = tempfile.mkdtemp(prefix="agentsnap_streaming_demo_")
    try:
        header("STEP 1 -- Record a streaming agent (the adapter tees the stream)")
        events = [FakeStreamEvent(e) for e in _event_dicts()]
        client = AnthropicAdapter(FakeAnthropicClient(FakeAnthropicStream(events)))
        with AgentRecorder("demo_streaming", snapshot_dir=snapshot_dir) as rec:
            rec.output = my_streaming_agent(client, "What is Python?")
        print(f"  golden snapshot recorded -- output: {rec.output!r}")

        header("STEP 2 -- Replay: recorded chunks rebuilt as real SDK objects")
        client = AnthropicAdapter(NetworkDisabledClient())  # proves no live call
        with AgentAsserter("demo_streaming", snapshot_dir=snapshot_dir, mode="replay") as a:
            a.output = my_streaming_agent(client, "What is Python?")
        print(f"  PASSED deterministically -- output: {a.output!r}")
        print("  ZERO live API calls were made during replay.")

        header("Done -- streaming agents record and replay just like non-streaming ones.")
    finally:
        shutil.rmtree(snapshot_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
