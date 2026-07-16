"""
streaming.py -- Streaming agents: record by teeing, replay from recorded chunks.

`AnthropicAdapter`/`OpenAIAdapter` (and PatchSet, used here) tee a `stream=True`
call: chunks flow through to the agent unmodified while the assembled response
is recorded for replay. A stream that's abandoned mid-iteration -- or never
closed at all -- is still finalized automatically when the recorder/asserter
context exits.

Usage:
    python examples/streaming.py             # mock only, no keys/network needed
    python examples/streaming.py --real      # mock, then a real streamed call,
                                               # then replayed with the network off
                                               # (needs ANTHROPIC_API_KEY, OPENAI_API_KEY,
                                               # or OPENROUTER_API_KEY; prints a skip hint
                                               # and exits 0 if none set)
    python examples/streaming.py --keep      # keep the temp snapshot dir, print its path

The journey (mock_demo):
  1. Record an agent that consumes a live `stream=True` call -- chunks arrive
     incrementally and are teed into a recorded snapshot.
  2. Replay -- the "network" is disabled. Recorded chunks are rebuilt as real
     SDK stream-event objects and yielded back, producing the same output
     with ZERO live API calls.
  3. Abandoned stream -- the agent stops consuming chunks early (no explicit
     close()). The partial call is still recorded when the context exits.
"""

from __future__ import annotations

import sys
import unittest.mock as mock
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import _common as ex
from agentsnap import PatchSet
from agentsnap.core.asserter import AgentAsserter

NAME = "streaming"


# ── A fake Anthropic-shaped streaming client (real SDK event objects) ──────

def _stream_event_dicts() -> list[dict]:
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


def _build_stream_events() -> list:
    """Validate the raw dicts into real anthropic SDK stream-event objects."""
    from anthropic.types import RawMessageStreamEvent
    from pydantic import TypeAdapter

    adapter = TypeAdapter(RawMessageStreamEvent)
    return [adapter.validate_python(d) for d in _stream_event_dicts()]


class _MockStream:
    """Minimal iterable + closeable stream -- what client.messages.create(stream=True) returns."""

    def __init__(self, events: list) -> None:
        self._events = events

    def __iter__(self):
        return iter(self._events)

    def close(self) -> None:
        pass


def _streaming_agent(question: str) -> str:
    """A tiny agent that consumes a live/replayed stream chunk by chunk. Zero agentsnap imports."""
    import anthropic

    client = anthropic.Anthropic(api_key="demo-key-no-real-call")
    chunks_seen = 0
    text_parts: list[str] = []
    stream = client.messages.create(
        model="claude-haiku-4-5",
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


def _abandoning_agent(question: str) -> str:
    """Consumes only the first two chunks then stops -- no explicit close()."""
    import anthropic

    client = anthropic.Anthropic(api_key="demo-key-no-real-call")
    chunks_seen = 0
    text_parts: list[str] = []
    stream = client.messages.create(
        model="claude-haiku-4-5",
        messages=[{"role": "user", "content": question}],
        max_tokens=100,
        stream=True,
    )
    for event in stream:
        chunks_seen += 1
        if event.type == "content_block_delta":
            text_parts.append(event.delta.text)
        if chunks_seen == 2:
            break
    return f"(abandoned after {chunks_seen} chunks) " + "".join(text_parts)


def mock_demo(snapshot_dir: str) -> None:
    ex.header("STREAMING (mock)  --  record by teeing, replay from recorded chunks")
    print("  Chunks flow through to your agent unmodified while the tee records them.\n")

    from anthropic.resources.messages.messages import Messages as _AnthMessages

    question = "What is Python?"

    ex.subheader("Step 1  Record a streaming agent (the adapter tees the stream)")
    events = _build_stream_events()
    with mock.patch.object(_AnthMessages, "create", return_value=_MockStream(events)):
        with PatchSet():
            with AgentAsserter(NAME, snapshot_dir=snapshot_dir) as a:
                a.output = _streaming_agent(question)
    print(f"  golden snapshot recorded -- output: {a.output!r}")

    def _exploding_create(*args, **kwargs):
        raise RuntimeError("NETWORK CALL ATTEMPTED -- replay mode should never do this!")

    ex.subheader("Step 2  Replay: recorded chunks rebuilt as real SDK objects")
    with mock.patch.object(_AnthMessages, "create", side_effect=_exploding_create):
        with PatchSet():
            with AgentAsserter(NAME, snapshot_dir=snapshot_dir, mode="replay") as a:
                a.output = _streaming_agent(question)
    print(f"  PASSED deterministically -- output: {a.output!r}")
    print("  ZERO live API calls were made during replay.")

    ex.subheader("Step 3  Abandoned stream -- still recorded automatically at context exit")
    events = _build_stream_events()
    with mock.patch.object(_AnthMessages, "create", return_value=_MockStream(events)):
        with PatchSet():
            with AgentAsserter(f"{NAME}_abandoned", snapshot_dir=snapshot_dir) as a:
                a.output = _abandoning_agent(question)
    print(f"  golden recorded even though only 2 of 7 chunks were consumed -- output: {a.output!r}")
    print("  TraceAccumulator.finalize_streams() closes any never-iterated-to-completion")
    print("  stream when the recorder/asserter context exits, so the event is never lost.")

    ex.header("Done -- streaming agents record and replay just like non-streaming ones.")


def real_demo(snapshot_dir: str) -> None:
    ex.maybe_load_dotenv()
    detected = ex.detect_real_client()
    if detected.client is None:
        ex.header("STREAMING (real)  --  skipped")
        print(f"  {detected.hint}")
        return

    ex.header(f"STREAMING (real)  --  provider: {detected.provider}, model: {detected.model}")
    print("  A real streamed call, chunks printed as they arrive, then replayed with the network off.\n")

    name = f"{NAME}_real"

    def call_stream(query: str) -> str:
        text_parts: list[str] = []
        chunk_count = 0
        if detected.provider == "anthropic":
            stream = detected.client.messages.create(
                model=detected.model,
                messages=[{"role": "user", "content": query}],
                max_tokens=ex.REAL_MAX_TOKENS,
                temperature=ex.REAL_TEMPERATURE,
                stream=True,
            )
            for event in stream:
                if getattr(event, "type", "") == "content_block_delta":
                    text = getattr(getattr(event, "delta", None), "text", None)
                    if text:
                        chunk_count += 1
                        text_parts.append(text)
                        if chunk_count <= 3:
                            print(f"    chunk {chunk_count}: {text!r}")
        else:
            stream = detected.client.chat.completions.create(
                model=detected.model,
                messages=[{"role": "user", "content": query}],
                max_tokens=ex.REAL_MAX_TOKENS,
                temperature=ex.REAL_TEMPERATURE,
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content if chunk.choices else None
                if delta:
                    chunk_count += 1
                    text_parts.append(delta)
                    if chunk_count <= 3:
                        print(f"    chunk {chunk_count}: {delta!r}")
        return f"Answer: {''.join(text_parts)}"

    query = "Summarize agentsnap in five words."

    ex.subheader("Step 1  Record a real streamed call (chunks arrive incrementally)")
    with PatchSet():
        with AgentAsserter(name, snapshot_dir=snapshot_dir) as a:
            a.output = call_stream(query)
    print(f"  golden snapshot recorded: {name}.json -- output: {a.output!r}")

    ex.subheader("Step 2  Replay -- ZERO network, even though the golden is real")
    with PatchSet():
        with AgentAsserter(name, snapshot_dir=snapshot_dir, mode="replay") as a:
            a.output = call_stream(query)
    print(f"  PASSED deterministically -- output: {a.output!r}")


def main() -> None:
    args = ex.parse_args(__doc__)
    with ex.temp_snapshot_dir(keep=args.keep) as snapshot_dir:
        if args.keep:
            print(f"Snapshot dir: {snapshot_dir}")
        mock_demo(snapshot_dir)
        if args.real:
            real_demo(snapshot_dir)
    ex.header("Streaming complete")


if __name__ == "__main__":
    main()
