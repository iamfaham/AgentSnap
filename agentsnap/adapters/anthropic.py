from __future__ import annotations

from agentsnap.core.recorder import TraceAccumulator


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
    from agentsnap.exceptions import ReplayError

    try:
        return reconstruct(event["raw_response"])
    except ReplayError:
        raise
    except Exception as e:
        raise ReplayError(
            f"Failed to reconstruct the recorded response for llm_call step "
            f"{event.get('step', '?')} — the snapshot may be corrupt or recorded "
            f"under a different SDK version ({e}). "
            "Re-record the golden: pytest --agentsnap-record"
        ) from e


def replay_stream(chunk_dicts):
    """Rebuild a recorded stream as a generator of real anthropic stream event objects."""
    from anthropic.types import RawMessageStreamEvent
    from pydantic import TypeAdapter

    from agentsnap.exceptions import ReplayError

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


class AnthropicRecordingStream:
    """Tees a streaming response: yields events unchanged, records the assembled call."""

    def __init__(self, inner, messages, acc) -> None:
        self._inner = inner
        self._messages = messages
        self._acc = acc
        self._chunks: list = []
        self._text: list[str] = []
        self._tokens = 0
        self._recorded = False

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
                self._tokens += getattr(usage, "input_tokens", 0) or 0
        elif etype == "message_delta":
            usage = getattr(event, "usage", None)
            if usage is not None:
                self._tokens += getattr(usage, "output_tokens", 0) or 0

    def _record(self) -> None:
        if self._recorded:
            return
        self._recorded = True
        self._acc.push(
            {
                "type": "llm_call",
                "messages": self._messages,
                "response": "".join(self._text),
                "tokens": self._tokens,
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
            acc.push(
                {
                    "type": "llm_call",
                    "messages": messages,
                    "response": event.get("response", ""),
                    "tokens": event.get("tokens", 0),
                    "raw_response": event.get("raw_response"),
                }
            )
            raw = event.get("raw_response")
            is_stream_recording = isinstance(raw, dict) and raw.get("__stream__")
            wants_stream = bool(kwargs.get("stream"))
            if wants_stream != bool(is_stream_recording):
                from agentsnap.exceptions import ReplayError

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
