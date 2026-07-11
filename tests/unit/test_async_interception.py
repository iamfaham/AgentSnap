from __future__ import annotations

import asyncio
import unittest.mock as mock

import anthropic
import openai
import pytest

from agentsnap.core.recorder import TraceAccumulator, _accumulator_var
from agentsnap.exceptions import ReplayError
from agentsnap.patches import PatchSet
from anthropic.resources.messages.messages import AsyncMessages as _AnthAsyncMessages
from openai.resources.chat.completions.completions import (
    AsyncCompletions as _OAIAsyncCompletions,
)


# ── Fake response shapes (non-stream) ─────────────────────────────────────────

class _AnthContent:
    text = "anthropic async patched response"


class _AnthResp:
    content = [_AnthContent()]

    class usage:
        input_tokens = 5
        output_tokens = 10

    def model_dump(self, mode="json"):
        return {"content": [{"type": "text", "text": self.content[0].text}]}


class _OAIMessage:
    content = "openai async patched response"


class _OAIChoice:
    message = _OAIMessage()


class _OAIResp:
    choices = [_OAIChoice()]

    class usage:
        total_tokens = 20

    def model_dump(self, mode="json"):
        return {"choices": [{"message": {"content": self.choices[0].message.content}}]}


# ── Fake async stream chunk shapes ────────────────────────────────────────────

class _AnthUsageStart:
    input_tokens = 4


class _AnthMessageStart:
    usage = _AnthUsageStart()


class _AnthDelta:
    def __init__(self, text):
        self.text = text


class _AnthUsageDelta:
    output_tokens = 6


class _AnthEvent:
    def __init__(self, type, message=None, delta=None, usage=None):
        self.type = type
        self.message = message
        self.delta = delta
        self.usage = usage

    def model_dump(self, mode="json"):
        return {"type": self.type}


class _OAIDelta:
    def __init__(self, content):
        self.content = content


class _OAIStreamChoice:
    def __init__(self, content):
        self.delta = _OAIDelta(content)


class _OAIChunk:
    def __init__(self, content=None, tokens=None):
        self.choices = [_OAIStreamChoice(content)]
        self.usage = type("U", (), {"total_tokens": tokens})() if tokens is not None else None

    def model_dump(self, mode="json"):
        return {"content": self.choices[0].delta.content}


class _FakeAsyncStream:
    """An async-iterable fake stream with an aclose(), mirroring SDK streams."""

    def __init__(self, items) -> None:
        self._items = list(items)
        self.aclose_called = False
        self._abandon_after: int | None = None

    def abandon_after(self, n: int) -> "_FakeAsyncStream":
        self._abandon_after = n
        return self

    async def __aiter__(self):
        yielded = 0
        try:
            for item in self._items:
                if self._abandon_after is not None and yielded >= self._abandon_after:
                    return
                yield item
                yielded += 1
        finally:
            pass

    async def aclose(self) -> None:
        self.aclose_called = True


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_acc(replay=None):
    acc = TraceAccumulator(replay=replay)
    token = _accumulator_var.set(acc)
    return acc, token


class _FakeReplaySession:
    def __init__(self, events):
        self._events = list(events)

    def next_llm_event(self):
        return self._events.pop(0)


# ── PatchSet restores async originals on exit ────────────────────────────────

def test_patchset_restores_async_anthropic_on_exit():
    original = _AnthAsyncMessages.create
    with PatchSet():
        assert _AnthAsyncMessages.create is not original
    assert _AnthAsyncMessages.create is original


def test_patchset_restores_async_openai_on_exit():
    original = _OAIAsyncCompletions.create
    with PatchSet():
        assert _OAIAsyncCompletions.create is not original
    assert _OAIAsyncCompletions.create is original


# ── Anthropic async: record, non-stream ──────────────────────────────────────

def test_anthropic_async_patcher_captures_llm_call():
    async def _run():
        acc, token = _make_acc()
        try:
            with mock.patch.object(
                _AnthAsyncMessages, "create", mock.AsyncMock(return_value=_AnthResp())
            ):
                with PatchSet():
                    client = anthropic.AsyncAnthropic(api_key="test-key-12345")
                    await client.messages.create(
                        model="claude-haiku-4-5",
                        messages=[{"role": "user", "content": "hello"}],
                        max_tokens=10,
                    )
        finally:
            _accumulator_var.reset(token)
        return acc.trace

    events = asyncio.run(_run())
    assert len(events) == 1
    assert events[0]["type"] == "llm_call"
    assert events[0]["response"] == "anthropic async patched response"
    assert events[0]["tokens"] == 15
    assert events[0]["messages"] == [{"role": "user", "content": "hello"}]
    assert events[0]["raw_response"] is not None
    assert events[0]["tool_requests"] == []


def test_anthropic_async_patcher_noop_without_accumulator():
    async def _run():
        assert TraceAccumulator.current() is None
        with mock.patch.object(
            _AnthAsyncMessages, "create", mock.AsyncMock(return_value=_AnthResp())
        ):
            with PatchSet():
                client = anthropic.AsyncAnthropic(api_key="test-key")
                return await client.messages.create(
                    model="test", messages=[{"role": "user", "content": "hi"}], max_tokens=5
                )

    result = asyncio.run(_run())
    assert result is not None


# ── Anthropic async: record, stream ───────────────────────────────────────────

def test_anthropic_async_patcher_stream_tees_in_order():
    from agentsnap.adapters.anthropic import AsyncAnthropicRecordingStream

    events = [
        _AnthEvent("message_start", message=_AnthMessageStart()),
        _AnthEvent("content_block_delta", delta=_AnthDelta("streamed ")),
        _AnthEvent("content_block_delta", delta=_AnthDelta("reply")),
        _AnthEvent("message_delta", usage=_AnthUsageDelta()),
        _AnthEvent("message_stop"),
    ]
    captured_kwargs = {}

    async def _spy_create(self, **kwargs):
        captured_kwargs.update(kwargs)
        return _FakeAsyncStream(events)

    async def _run():
        acc, token = _make_acc()
        seen = []
        try:
            with mock.patch.object(_AnthAsyncMessages, "create", _spy_create):
                with PatchSet():
                    client = anthropic.AsyncAnthropic(api_key="test-key")
                    result = await client.messages.create(
                        model="claude-haiku-4-5",
                        messages=[{"role": "user", "content": "hi"}],
                        max_tokens=10,
                        stream=True,
                    )
                    assert isinstance(result, AsyncAnthropicRecordingStream)
                    async for chunk in result:
                        seen.append(chunk)
        finally:
            _accumulator_var.reset(token)
        return acc.trace, seen

    trace, seen = asyncio.run(_run())
    assert captured_kwargs.get("stream") is True
    assert seen == events  # chunks reached the consumer in order
    assert len(trace) == 1
    assert trace[0]["response"] == "streamed reply"
    assert trace[0]["tokens"] == 10
    assert trace[0]["raw_response"]["__stream__"] is True


def test_anthropic_async_patcher_abandoned_iteration_still_records():
    """Breaking out of `async for` early still records a partial event."""
    events = [
        _AnthEvent("message_start", message=_AnthMessageStart()),
        _AnthEvent("content_block_delta", delta=_AnthDelta("partial ")),
        _AnthEvent("content_block_delta", delta=_AnthDelta("never seen")),
    ]

    async def _spy_create(self, **kwargs):
        return _FakeAsyncStream(events)

    async def _run():
        acc, token = _make_acc()
        try:
            with mock.patch.object(_AnthAsyncMessages, "create", _spy_create):
                with PatchSet():
                    client = anthropic.AsyncAnthropic(api_key="test-key")
                    result = await client.messages.create(
                        model="claude-haiku-4-5",
                        messages=[{"role": "user", "content": "hi"}],
                        max_tokens=10,
                        stream=True,
                    )
                    # Simulate a consumer abandoning iteration mid-stream: consume
                    # message_start + the first text delta, then explicitly
                    # aclose() the async generator (deterministic — relying on
                    # GC/finalizer timing for abandoned async generators is not
                    # deterministic across event loops).
                    agen = result.__aiter__()
                    await agen.__anext__()
                    await agen.__anext__()
                    await agen.aclose()
        finally:
            _accumulator_var.reset(token)
        return acc.trace

    trace = asyncio.run(_run())
    assert len(trace) == 1
    assert trace[0]["response"] == "partial "


def test_anthropic_async_patcher_sync_close_records_without_touching_inner():
    from agentsnap.adapters.anthropic import AsyncAnthropicRecordingStream

    inner = _FakeAsyncStream(
        [
            _AnthEvent("message_start", message=_AnthMessageStart()),
            _AnthEvent("content_block_delta", delta=_AnthDelta("hi")),
        ]
    )
    acc, token = _make_acc()
    try:
        stream = AsyncAnthropicRecordingStream(inner, [{"role": "user", "content": "hi"}], acc)
        stream.close()  # sync close: records only, never touches inner
    finally:
        _accumulator_var.reset(token)

    assert len(acc.trace) == 1
    assert inner.aclose_called is False


# ── Anthropic async: replay ────────────────────────────────────────────────────

def test_anthropic_async_patcher_replay_non_stream_never_calls_original():
    async def _exploding_original(self, *args, **kwargs):
        raise AssertionError("original should never be awaited during replay")

    event = {
        "response": "recorded reply",
        "tokens": 9,
        "raw_response": {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "recorded reply"}],
            "model": "claude-haiku-4-5",
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 4, "output_tokens": 5},
        },
        "step": 0,
    }

    async def _run():
        acc, token = _make_acc(replay=_FakeReplaySession([event]))
        try:
            with mock.patch.object(_AnthAsyncMessages, "create", _exploding_original):
                with PatchSet():
                    client = anthropic.AsyncAnthropic(api_key="test-key")
                    return await client.messages.create(
                        model="claude-haiku-4-5",
                        messages=[{"role": "user", "content": "hi"}],
                        max_tokens=10,
                    )
        finally:
            _accumulator_var.reset(token)

    result = asyncio.run(_run())
    assert result.content[0].text == "recorded reply"


def test_anthropic_async_patcher_replay_stream_yields_real_chunks():
    raw_chunks = [
        {"type": "message_start", "message": {
            "id": "msg_1", "type": "message", "role": "assistant", "content": [],
            "model": "claude-haiku-4-5", "stop_reason": None, "stop_sequence": None,
            "usage": {"input_tokens": 4, "output_tokens": 0},
        }},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "hi"}},
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None}, "usage": {"output_tokens": 1}},
        {"type": "message_stop"},
    ]
    event = {
        "response": "hi",
        "tokens": 5,
        "raw_response": {"__stream__": True, "chunks": raw_chunks},
        "step": 0,
    }

    async def _exploding_original(self, *args, **kwargs):
        raise AssertionError("original should never be awaited during replay")

    async def _run():
        acc, token = _make_acc(replay=_FakeReplaySession([event]))
        chunks = []
        try:
            with mock.patch.object(_AnthAsyncMessages, "create", _exploding_original):
                with PatchSet():
                    client = anthropic.AsyncAnthropic(api_key="test-key")
                    result = await client.messages.create(
                        model="claude-haiku-4-5",
                        messages=[{"role": "user", "content": "hi"}],
                        max_tokens=10,
                        stream=True,
                    )
                    async with result as stream:
                        async for chunk in stream:
                            chunks.append(chunk)
        finally:
            _accumulator_var.reset(token)
        return chunks

    chunks = asyncio.run(_run())
    assert len(chunks) == len(raw_chunks)
    assert chunks[0].type == "message_start"
    assert chunks[2].delta.text == "hi"


def test_anthropic_async_patcher_replay_shape_mismatch_raises():
    """Recording is non-stream but the caller asks for stream=True."""
    event = {
        "response": "recorded reply",
        "tokens": 9,
        "raw_response": {"content": [{"type": "text", "text": "recorded reply"}]},
        "step": 0,
    }

    async def _exploding_original(self, *args, **kwargs):
        raise AssertionError("original should never be awaited during replay")

    async def _run():
        acc, token = _make_acc(replay=_FakeReplaySession([event]))
        try:
            with mock.patch.object(_AnthAsyncMessages, "create", _exploding_original):
                with PatchSet():
                    client = anthropic.AsyncAnthropic(api_key="test-key")
                    await client.messages.create(
                        model="claude-haiku-4-5",
                        messages=[{"role": "user", "content": "hi"}],
                        max_tokens=10,
                        stream=True,
                    )
        finally:
            _accumulator_var.reset(token)

    with pytest.raises(ReplayError):
        asyncio.run(_run())


# ── OpenAI async: record, non-stream ──────────────────────────────────────────

def test_openai_async_patcher_captures_llm_call():
    async def _run():
        acc, token = _make_acc()
        try:
            with mock.patch.object(
                _OAIAsyncCompletions, "create", mock.AsyncMock(return_value=_OAIResp())
            ):
                with PatchSet():
                    client = openai.AsyncOpenAI(api_key="test-key-12345")
                    await client.chat.completions.create(
                        model="gpt-4o",
                        messages=[{"role": "user", "content": "hello"}],
                        max_tokens=10,
                    )
        finally:
            _accumulator_var.reset(token)
        return acc.trace

    events = asyncio.run(_run())
    assert len(events) == 1
    assert events[0]["type"] == "llm_call"
    assert events[0]["response"] == "openai async patched response"
    assert events[0]["tokens"] == 20
    assert events[0]["raw_response"] is not None
    assert events[0]["tool_requests"] == []


def test_openai_async_patcher_forces_stream_false_when_caller_did_not_pass_stream():
    captured_kwargs = {}

    async def _spy_create(self, **kwargs):
        captured_kwargs.update(kwargs)
        return _OAIResp()

    async def _run():
        acc, token = _make_acc()
        try:
            with mock.patch.object(_OAIAsyncCompletions, "create", _spy_create):
                with PatchSet():
                    client = openai.AsyncOpenAI(api_key="test-key")
                    await client.chat.completions.create(
                        model="gpt-4o",
                        messages=[{"role": "user", "content": "hi"}],
                        max_tokens=5,
                    )
        finally:
            _accumulator_var.reset(token)

    asyncio.run(_run())
    assert captured_kwargs.get("stream") is False


def test_openai_async_patcher_noop_without_accumulator():
    async def _run():
        assert TraceAccumulator.current() is None
        with mock.patch.object(
            _OAIAsyncCompletions, "create", mock.AsyncMock(return_value=_OAIResp())
        ):
            with PatchSet():
                client = openai.AsyncOpenAI(api_key="test-key")
                return await client.chat.completions.create(
                    model="gpt-4o", messages=[{"role": "user", "content": "hi"}], max_tokens=5
                )

    result = asyncio.run(_run())
    assert result is not None


# ── OpenAI async: record, stream ──────────────────────────────────────────────

def test_openai_async_patcher_stream_tees_in_order():
    from agentsnap.adapters.openai import AsyncOpenAIRecordingStream

    chunks = [_OAIChunk("streamed "), _OAIChunk("reply"), _OAIChunk(None, tokens=7)]
    captured_kwargs = {}

    async def _spy_create(self, **kwargs):
        captured_kwargs.update(kwargs)
        return _FakeAsyncStream(chunks)

    async def _run():
        acc, token = _make_acc()
        seen = []
        try:
            with mock.patch.object(_OAIAsyncCompletions, "create", _spy_create):
                with PatchSet():
                    client = openai.AsyncOpenAI(api_key="test-key")
                    result = await client.chat.completions.create(
                        model="gpt-4o",
                        messages=[{"role": "user", "content": "hi"}],
                        max_tokens=5,
                        stream=True,
                    )
                    assert isinstance(result, AsyncOpenAIRecordingStream)
                    async for chunk in result:
                        seen.append(chunk)
        finally:
            _accumulator_var.reset(token)
        return acc.trace, seen

    trace, seen = asyncio.run(_run())
    assert captured_kwargs.get("stream") is True
    assert seen == chunks
    assert len(trace) == 1
    assert trace[0]["response"] == "streamed reply"
    assert trace[0]["tokens"] == 7
    assert trace[0]["raw_response"]["__stream__"] is True


def test_openai_async_patcher_abandoned_iteration_still_records():
    chunks = [_OAIChunk("partial "), _OAIChunk("never seen")]

    async def _spy_create(self, **kwargs):
        return _FakeAsyncStream(chunks)

    async def _run():
        acc, token = _make_acc()
        try:
            with mock.patch.object(_OAIAsyncCompletions, "create", _spy_create):
                with PatchSet():
                    client = openai.AsyncOpenAI(api_key="test-key")
                    result = await client.chat.completions.create(
                        model="gpt-4o",
                        messages=[{"role": "user", "content": "hi"}],
                        max_tokens=5,
                        stream=True,
                    )
                    # Simulate a consumer abandoning iteration mid-stream (see
                    # the analogous Anthropic test for why this is explicit
                    # rather than GC-timing-dependent).
                    agen = result.__aiter__()
                    await agen.__anext__()
                    await agen.aclose()
        finally:
            _accumulator_var.reset(token)
        return acc.trace

    trace = asyncio.run(_run())
    assert len(trace) == 1
    assert trace[0]["response"] == "partial "


def test_openai_async_patcher_sync_close_records_without_touching_inner():
    from agentsnap.adapters.openai import AsyncOpenAIRecordingStream

    inner = _FakeAsyncStream([_OAIChunk("hi")])
    acc, token = _make_acc()
    try:
        stream = AsyncOpenAIRecordingStream(inner, [{"role": "user", "content": "hi"}], acc)
        stream.close()
    finally:
        _accumulator_var.reset(token)

    assert len(acc.trace) == 1
    assert inner.aclose_called is False


# ── OpenAI async: replay ───────────────────────────────────────────────────────

def test_openai_async_patcher_replay_non_stream_never_calls_original():
    async def _exploding_original(self, *args, **kwargs):
        raise AssertionError("original should never be awaited during replay")

    event = {
        "response": "recorded reply",
        "tokens": 12,
        "raw_response": {
            "id": "chatcmpl-1",
            "created": 0,
            "model": "gpt-4o",
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "recorded reply"},
                }
            ],
        },
        "step": 0,
    }

    async def _run():
        acc, token = _make_acc(replay=_FakeReplaySession([event]))
        try:
            with mock.patch.object(_OAIAsyncCompletions, "create", _exploding_original):
                with PatchSet():
                    client = openai.AsyncOpenAI(api_key="test-key")
                    return await client.chat.completions.create(
                        model="gpt-4o",
                        messages=[{"role": "user", "content": "hi"}],
                        max_tokens=10,
                    )
        finally:
            _accumulator_var.reset(token)

    result = asyncio.run(_run())
    assert result.choices[0].message.content == "recorded reply"


def test_openai_async_patcher_replay_stream_yields_real_chunks():
    raw_chunks = [
        {"id": "c1", "choices": [{"index": 0, "delta": {"content": "hi"}, "finish_reason": None}],
         "created": 0, "model": "gpt-4o", "object": "chat.completion.chunk"},
        {"id": "c2", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
         "created": 0, "model": "gpt-4o", "object": "chat.completion.chunk"},
    ]
    event = {
        "response": "hi",
        "tokens": 3,
        "raw_response": {"__stream__": True, "chunks": raw_chunks},
        "step": 0,
    }

    async def _exploding_original(self, *args, **kwargs):
        raise AssertionError("original should never be awaited during replay")

    async def _run():
        acc, token = _make_acc(replay=_FakeReplaySession([event]))
        collected = []
        try:
            with mock.patch.object(_OAIAsyncCompletions, "create", _exploding_original):
                with PatchSet():
                    client = openai.AsyncOpenAI(api_key="test-key")
                    result = await client.chat.completions.create(
                        model="gpt-4o",
                        messages=[{"role": "user", "content": "hi"}],
                        max_tokens=10,
                        stream=True,
                    )
                    async with result as stream:
                        async for chunk in stream:
                            collected.append(chunk)
        finally:
            _accumulator_var.reset(token)
        return collected

    chunks = asyncio.run(_run())
    assert len(chunks) == len(raw_chunks)
    assert chunks[0].choices[0].delta.content == "hi"
    assert chunks[1].choices[0].finish_reason == "stop"


def test_openai_async_patcher_replay_shape_mismatch_raises():
    event = {
        "response": "recorded reply",
        "tokens": 12,
        "raw_response": {"choices": [{"message": {"content": "recorded reply"}}]},
        "step": 0,
    }

    async def _exploding_original(self, *args, **kwargs):
        raise AssertionError("original should never be awaited during replay")

    async def _run():
        acc, token = _make_acc(replay=_FakeReplaySession([event]))
        try:
            with mock.patch.object(_OAIAsyncCompletions, "create", _exploding_original):
                with PatchSet():
                    client = openai.AsyncOpenAI(api_key="test-key")
                    await client.chat.completions.create(
                        model="gpt-4o",
                        messages=[{"role": "user", "content": "hi"}],
                        max_tokens=10,
                        stream=True,
                    )
        finally:
            _accumulator_var.reset(token)

    with pytest.raises(ReplayError):
        asyncio.run(_run())
