from __future__ import annotations

from agentsnap.adapters._raw_response import (
    RawResponseStreamShim,
    ReplayLegacyResponse,
    reconstruct_event_with_clear_errors,
    unwrap_legacy_response,
    wants_raw_response,
)
from agentsnap.core.recorder import TraceAccumulator
from agentsnap.exceptions import ReplayError

__all__ = [
    "RawResponseStreamShim",
    "ReplayLegacyResponse",
    "unwrap_legacy_response",
    "wants_raw_response",
]


def extract_tool_requests(response) -> list[dict]:
    """Extract model-requested tool_use blocks from an Anthropic response."""
    requests = []
    for block in getattr(response, "content", []) or []:
        if getattr(block, "type", "") != "tool_use":
            continue
        args = dict(block.input) if isinstance(block.input, dict) else block.input
        requests.append({"name": block.name, "args": args})
    return requests


def dump_raw(response) -> dict | None:
    """Serialize a provider response for replay. None if the object can't dump."""
    dump = getattr(response, "model_dump", None)
    if dump is None:
        return None
    try:
        return dump(mode="json")
    except TypeError:
        try:
            return dump()
        except Exception:
            return None
    except Exception:
        return None


def reconstruct(raw: dict):
    """Rebuild an anthropic Message object from a recorded raw_response dict."""
    from anthropic.types import Message

    return Message.model_validate(raw)


def reconstruct_event(event: dict):
    """Rebuild the recorded response for a replayed event, with a clear error on failure."""
    return reconstruct_event_with_clear_errors(reconstruct, event, this_api="Anthropic")


def replay_stream(chunk_dicts):
    """Rebuild a recorded stream as a generator of real anthropic stream event objects."""
    from anthropic.types import RawMessageStreamEvent
    from pydantic import TypeAdapter

    adapter = TypeAdapter(RawMessageStreamEvent)

    events = []
    for i, raw in enumerate(chunk_dicts):
        try:
            events.append(adapter.validate_python(raw))
        except Exception as e:
            raise ReplayError(
                f"Failed to reconstruct recorded stream chunk {i} — the snapshot may "
                f"be corrupt or recorded under a different SDK version ({e}). "
                "Re-record the golden: pytest --agentsnap-record"
            ) from e

    return _ReplayedStream(events)


def replay_stream_async(chunk_dicts):
    """Rebuild a recorded stream as an async generator of real anthropic stream event objects."""
    from anthropic.types import RawMessageStreamEvent
    from pydantic import TypeAdapter

    adapter = TypeAdapter(RawMessageStreamEvent)

    events = []
    for i, raw in enumerate(chunk_dicts):
        try:
            events.append(adapter.validate_python(raw))
        except Exception as e:
            raise ReplayError(
                f"Failed to reconstruct recorded stream chunk {i} — the snapshot may "
                f"be corrupt or recorded under a different SDK version ({e}). "
                "Re-record the golden: pytest --agentsnap-record"
            ) from e

    return _AsyncReplayedStream(events)


class _ReplayedStream:
    """Replayed stream: iterable + context manager, mirroring the SDK Stream surface."""

    def __init__(self, items) -> None:
        self._items = items

    def __iter__(self):
        yield from self._items

    def close(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args) -> None:
        self.close()


class _AsyncReplayedStream:
    """Replayed async stream: async-iterable + async context manager."""

    def __init__(self, items) -> None:
        self._items = items

    async def __aiter__(self):
        for item in self._items:
            yield item

    async def aclose(self) -> None:
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args) -> None:
        await self.aclose()


class AnthropicRecordingStream:
    """Tees a streaming response: yields events unchanged, records the assembled call."""

    def __init__(self, inner, messages, acc) -> None:
        self._inner = inner
        self._messages = messages
        self._acc = acc
        self._chunks: list = []
        self._text: list[str] = []
        self._input_tokens = 0
        self._output_tokens = 0
        self._recorded = False
        acc.register_stream(self)

    def __iter__(self):
        # finally covers natural exhaustion, consumer break (GeneratorExit),
        # and mid-stream exceptions — the partial call is always recorded.
        # Note: record mode pushes the llm_call at stream exhaustion/close, while
        # replay pushes it at create() time — with a single active stream the
        # relative order of llm_call vs interleaved tool_call events can differ
        # between record and replay. Structural diff (tool names) and
        # llm_request_diffs (llm order) are unaffected.
        try:
            for event in self._inner:
                self._capture(event)
                yield event
        finally:
            self._record()

    def _capture(self, event) -> None:
        self._chunks.append(event)
        etype = getattr(event, "type", "")
        if etype == "content_block_delta":
            delta = getattr(event, "delta", None)
            text = getattr(delta, "text", None)
            if text:
                self._text.append(text)
        elif etype == "message_start":
            usage = getattr(getattr(event, "message", None), "usage", None)
            if usage is not None:
                self._input_tokens = getattr(usage, "input_tokens", 0) or 0
        elif etype == "message_delta":
            usage = getattr(event, "usage", None)
            if usage is not None:
                # Anthropic's message_delta.usage.output_tokens is cumulative
                # across the stream, so the latest value replaces (not adds to)
                # the running total rather than accumulating.
                self._output_tokens = getattr(usage, "output_tokens", 0) or 0

    def _record(self) -> None:
        if self._recorded:
            return
        self._recorded = True
        self._acc.push(
            {
                "type": "llm_call",
                "messages": self._messages,
                "response": "".join(self._text),
                "tokens": self._input_tokens + self._output_tokens,
                "raw_response": {
                    "__stream__": True,
                    "chunks": [dump_raw(c) for c in self._chunks],
                },
            }
        )

    def close(self) -> None:
        try:
            self._inner.close()
        finally:
            self._record()

    def __enter__(self):
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def __getattr__(self, name):
        return getattr(self._inner, name)


class AsyncAnthropicRecordingStream(AnthropicRecordingStream):
    """Async tee: yields events unchanged, records the assembled call.

    Subclasses the sync tee to reuse ``_capture``/``_record``/``__init__``
    (including ``acc.register_stream(self)``) and only overrides the
    iteration/close surface for the async protocol.
    """

    async def __aiter__(self):
        # Mirrors AnthropicRecordingStream.__iter__: try/finally covers
        # natural exhaustion, consumer abandonment (GeneratorExit via
        # aclose()), and mid-stream exceptions — the partial call is
        # always recorded.
        try:
            async for event in self._inner:
                self._capture(event)
                yield event
        finally:
            self._record()

    async def aclose(self) -> None:
        try:
            inner_aclose = getattr(self._inner, "aclose", None)
            if inner_aclose is not None:
                await inner_aclose()
        finally:
            self._record()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args) -> None:
        await self.aclose()

    def __iter__(self):
        raise TypeError(
            "AsyncAnthropicRecordingStream wraps an async stream — use 'async for' / 'async with'"
        )

    def __enter__(self):
        raise TypeError(
            "AsyncAnthropicRecordingStream wraps an async stream — use 'async for' / 'async with'"
        )

    def close(self) -> None:
        # TraceAccumulator.finalize_streams() runs synchronously at context
        # exit and cannot await the inner stream's aclose(); recording the
        # partial event here matters more than closing the inner stream, so
        # this sync close() only records — it never touches self._inner.
        self._record()


class _MessagesProxy:
    def __init__(self, original) -> None:
        self._original = original

    def create(self, **kwargs):
        acc = TraceAccumulator.current()
        if acc is None:
            return self._original.create(**kwargs)

        messages = kwargs.get("messages", [])

        if acc.replay is not None:
            event = acc.replay.next_llm_event()
            pushed = {
                "type": "llm_call",
                "messages": messages,
                "response": event.get("response", ""),
                "tokens": event.get("tokens", 0),
                "raw_response": event.get("raw_response"),
            }
            if "tool_requests" in event:
                pushed["tool_requests"] = event["tool_requests"]
            acc.push(pushed)
            raw = event.get("raw_response")
            is_stream_recording = isinstance(raw, dict) and raw.get("__stream__")
            wants_stream = bool(kwargs.get("stream"))
            if wants_stream != bool(is_stream_recording):
                raise ReplayError(
                    f"Replay shape mismatch at llm_call step {event.get('step', '?')}: "
                    f"the snapshot recorded a {'streaming' if is_stream_recording else 'non-streaming'} call "
                    f"but the agent requested {'streaming' if wants_stream else 'non-streaming'}. "
                    "Re-record the golden: pytest --agentsnap-record"
                )
            if is_stream_recording:
                return replay_stream(raw["chunks"])
            return reconstruct_event(event)

        if kwargs.get("stream"):
            response = self._original.create(**kwargs)
            return AnthropicRecordingStream(response, messages, acc)

        response = self._original.create(**kwargs)

        response_text = ""
        tokens = 0
        if hasattr(response, "content"):
            for block in response.content:
                if hasattr(block, "text"):
                    response_text += block.text
        if hasattr(response, "usage"):
            tokens = getattr(response.usage, "input_tokens", 0) + getattr(
                response.usage, "output_tokens", 0
            )

        acc.push(
            {
                "type": "llm_call",
                "messages": messages,
                "response": response_text,
                "tokens": tokens,
                "raw_response": dump_raw(response),
                "tool_requests": extract_tool_requests(response),
            }
        )
        return response

    def __getattr__(self, name: str):
        return getattr(self._original, name)


class AnthropicAdapter:
    """Wraps an anthropic.Anthropic() client to intercept .messages.create()."""

    def __init__(self, client) -> None:
        self._client = client
        self.messages = _MessagesProxy(client.messages)

    def __getattr__(self, name: str):
        return getattr(self._client, name)
