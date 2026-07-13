from __future__ import annotations

import json

from agentsnap.adapters._raw_response import (
    RawResponseStreamShim,
    ReplayLegacyResponse,
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
    """Extract model-requested tool_calls from an OpenAI ChatCompletion response."""
    requests = []
    choices = getattr(response, "choices", None) or []
    if not choices:
        return requests
    message = getattr(choices[0], "message", None)
    tool_calls = getattr(message, "tool_calls", None) or []
    for tc in tool_calls:
        function = getattr(tc, "function", None)
        name = getattr(function, "name", None)
        arguments = getattr(function, "arguments", None)
        try:
            args = json.loads(arguments)
        except (TypeError, ValueError):
            args = arguments
        requests.append({"name": name, "args": args})
    return requests


def extract_responses_text(response) -> str:
    """Extract the assembled output text from a Responses API response."""
    return getattr(response, "output_text", "") or ""


def extract_responses_tool_requests(response) -> list[dict]:
    """Extract model-requested function calls from a Responses API response."""
    requests = []
    output = getattr(response, "output", None) or []
    for item in output:
        if getattr(item, "type", "") != "function_call":
            continue
        name = getattr(item, "name", None)
        arguments = getattr(item, "arguments", None)
        try:
            args = json.loads(arguments)
        except (TypeError, ValueError):
            args = arguments
        requests.append({"name": name, "args": args})
    return requests


def normalize_responses_input(kwargs: dict) -> list[dict]:
    """Normalize Responses API kwargs into a chat-style messages list."""
    raw_input = kwargs.get("input")
    if isinstance(raw_input, str):
        messages = [{"role": "user", "content": raw_input}]
    elif isinstance(raw_input, list):
        messages = list(raw_input)
    else:
        messages = []
    instructions = kwargs.get("instructions")
    if instructions:
        messages = [{"role": "system", "content": instructions}] + messages
    return messages


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


def _lenient_reconstruct(model_type, raw: dict, object_marker: str):
    """Validate strictly first; fall back to the SDK's own lenient wire parsing.

    SDK schema evolution can make a recorded dump fail strict validation
    (e.g. a usage field that became required after the snapshot was taken).
    `construct_type` is what the openai SDK itself uses to parse wire
    responses, so it tolerates the same drift. The fallback engages only
    when the payload carries the expected `object` marker — genuinely
    corrupt data still raises, preserving the ReplayError path.
    """
    try:
        return model_type.model_validate(raw)
    except Exception:
        if not (isinstance(raw, dict) and raw.get("object") == object_marker):
            raise
        from openai._models import construct_type

        return construct_type(type_=model_type, value=raw)


def reconstruct(raw: dict):
    """Rebuild an openai ChatCompletion object from a recorded raw_response dict."""
    from openai.types.chat import ChatCompletion

    return _lenient_reconstruct(ChatCompletion, raw, "chat.completion")


def reconstruct_event(event: dict):
    """Rebuild the recorded response for a replayed event, with a clear error on failure."""
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


def reconstruct_response(raw: dict):
    """Rebuild an openai Responses API Response object from a recorded raw_response dict."""
    from openai.types.responses import Response

    return _lenient_reconstruct(Response, raw, "response")


def reconstruct_response_event(event: dict):
    """Rebuild the recorded Responses API response for a replayed event, with a clear error on failure."""
    try:
        return reconstruct_response(event["raw_response"])
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
    """Rebuild a recorded stream as a generator of real openai ChatCompletionChunk objects."""
    from openai.types.chat import ChatCompletionChunk

    chunks = []
    for i, raw in enumerate(chunk_dicts):
        try:
            chunks.append(ChatCompletionChunk.model_validate(raw))
        except Exception as e:
            raise ReplayError(
                f"Failed to reconstruct recorded stream chunk {i} — the snapshot may "
                f"be corrupt or recorded under a different SDK version ({e}). "
                "Re-record the golden: pytest --agentsnap-record"
            ) from e

    return _ReplayedStream(chunks)


def replay_stream_async(chunk_dicts):
    """Rebuild a recorded stream as an async generator of real openai ChatCompletionChunk objects."""
    from openai.types.chat import ChatCompletionChunk

    chunks = []
    for i, raw in enumerate(chunk_dicts):
        try:
            chunks.append(ChatCompletionChunk.model_validate(raw))
        except Exception as e:
            raise ReplayError(
                f"Failed to reconstruct recorded stream chunk {i} — the snapshot may "
                f"be corrupt or recorded under a different SDK version ({e}). "
                "Re-record the golden: pytest --agentsnap-record"
            ) from e

    return _AsyncReplayedStream(chunks)


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


class OpenAIRecordingStream:
    """Tees a streaming response: yields chunks unchanged, records the assembled call."""

    def __init__(self, inner, messages, acc) -> None:
        self._inner = inner
        self._messages = messages
        self._acc = acc
        self._chunks: list = []
        self._text: list[str] = []
        self._tokens = 0
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
            for chunk in self._inner:
                self._capture(chunk)
                yield chunk
        finally:
            self._record()

    def _capture(self, chunk) -> None:
        self._chunks.append(chunk)
        if getattr(chunk, "choices", None):
            delta = chunk.choices[0].delta
            if getattr(delta, "content", None):
                self._text.append(delta.content)
        usage = getattr(chunk, "usage", None)
        if usage is not None:
            self._tokens = getattr(usage, "total_tokens", 0) or 0

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


class AsyncOpenAIRecordingStream(OpenAIRecordingStream):
    """Async tee: yields chunks unchanged, records the assembled call.

    Subclasses the sync tee to reuse ``_capture``/``_record``/``__init__``
    (including ``acc.register_stream(self)``) and only overrides the
    iteration/close surface for the async protocol.
    """

    async def __aiter__(self):
        # Mirrors OpenAIRecordingStream.__iter__: try/finally covers
        # natural exhaustion, consumer abandonment (GeneratorExit via
        # aclose()), and mid-stream exceptions — the partial call is
        # always recorded.
        try:
            async for chunk in self._inner:
                self._capture(chunk)
                yield chunk
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

    def close(self) -> None:
        # TraceAccumulator.finalize_streams() runs synchronously at context
        # exit and cannot await the inner stream's aclose(); recording the
        # partial event here matters more than closing the inner stream, so
        # this sync close() only records — it never touches self._inner.
        self._record()


class _CompletionsProxy:
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
            return OpenAIRecordingStream(response, messages, acc)

        # Force non-streaming so we always get a complete ChatCompletion object.
        # Streaming responses expose deltas, not the full message content.
        kwargs["stream"] = False
        response = self._original.create(**kwargs)

        response_text = ""
        tokens = 0
        if hasattr(response, "choices") and response.choices:
            msg = response.choices[0].message
            response_text = msg.content or ""
        if hasattr(response, "usage"):
            tokens = getattr(response.usage, "total_tokens", 0)

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


class _ChatProxy:
    def __init__(self, chat) -> None:
        self._chat = chat
        self.completions = _CompletionsProxy(chat.completions)

    def __getattr__(self, name: str):
        return getattr(self._chat, name)


class OpenAIAdapter:
    """Wraps an openai.OpenAI() client to intercept .chat.completions.create()."""

    def __init__(self, client) -> None:
        self._client = client
        self.chat = _ChatProxy(client.chat)

    def __getattr__(self, name: str):
        return getattr(self._client, name)
