from __future__ import annotations

import asyncio
import json
import unittest.mock as mock

import openai
import pytest
from openai.resources.responses.responses import Responses as _Responses
from openai.resources.responses.responses import AsyncResponses as _AsyncResponses
from openai.types.responses import Response

from agentsnap.adapters.openai import (
    extract_responses_text,
    extract_responses_tool_requests,
    normalize_responses_input,
    reconstruct_response_event,
)
from agentsnap.core.recorder import AgentRecorder, TraceAccumulator, _accumulator_var
from agentsnap.core.snapshot import read_snapshot
from agentsnap.exceptions import ReplayError
from agentsnap.patches import PatchSet


# ── Canned Response dict (bootstrap: must validate against the real SDK type) ──

def _canned_response_dict(text: str = "Hello world", tokens: int = 15) -> dict:
    return {
        "id": "resp_123",
        "created_at": 1700000000,
        "model": "gpt-4o-mini",
        "object": "response",
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "output": [
            {
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [
                    {"type": "output_text", "text": text, "annotations": []},
                ],
            },
            {
                "id": "fc_1",
                "type": "function_call",
                "call_id": "call_1",
                "name": "get_weather",
                "arguments": json.dumps({"city": "SF"}),
            },
        ],
        "usage": {
            "input_tokens": 10,
            "output_tokens": 5,
            "total_tokens": tokens,
            "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 0},
        },
    }


def test_canned_response_dict_validates_against_real_sdk_type():
    r = Response.model_validate(_canned_response_dict())
    assert r.output_text == "Hello world"
    assert r.usage.total_tokens == 15


def _make_response(text: str = "Hello world", tokens: int = 15) -> Response:
    return Response.model_validate(_canned_response_dict(text=text, tokens=tokens))


# ── extract_responses_text ─────────────────────────────────────────────────────

def test_extract_responses_text():
    resp = _make_response("hi there")
    assert extract_responses_text(resp) == "hi there"


def test_extract_responses_text_missing_attr_returns_empty():
    class Bare:
        pass
    assert extract_responses_text(Bare()) == ""


# ── extract_responses_tool_requests ────────────────────────────────────────────

def test_extract_responses_tool_requests():
    resp = _make_response()
    requests = extract_responses_tool_requests(resp)
    assert requests == [{"name": "get_weather", "args": {"city": "SF"}}]


def test_extract_responses_tool_requests_unparseable_args_fallback():
    d = _canned_response_dict()
    d["output"][1]["arguments"] = "not json{"
    resp = Response.model_validate(d)
    requests = extract_responses_tool_requests(resp)
    assert requests == [{"name": "get_weather", "args": "not json{"}]


def test_extract_responses_tool_requests_empty_output():
    class Bare:
        output = []
    assert extract_responses_tool_requests(Bare()) == []


def test_extract_responses_tool_requests_missing_output_attr():
    class Bare:
        pass
    assert extract_responses_tool_requests(Bare()) == []


# ── normalize_responses_input ──────────────────────────────────────────────────

def test_normalize_responses_input_str():
    messages = normalize_responses_input({"input": "hello"})
    assert messages == [{"role": "user", "content": "hello"}]


def test_normalize_responses_input_list():
    original = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]
    messages = normalize_responses_input({"input": original})
    assert messages == original


def test_normalize_responses_input_with_instructions():
    messages = normalize_responses_input({"input": "hello", "instructions": "be nice"})
    assert messages == [
        {"role": "system", "content": "be nice"},
        {"role": "user", "content": "hello"},
    ]


def test_normalize_responses_input_no_input_key():
    assert normalize_responses_input({}) == []


# ── Helpers for patcher tests ──────────────────────────────────────────────────

def _make_acc(replay=None):
    acc = TraceAccumulator(replay=replay)
    token = _accumulator_var.set(acc)
    return acc, token


class _FakeReplaySession:
    def __init__(self, events):
        self._events = list(events)

    def next_llm_event(self):
        return self._events.pop(0)


# ── Sync record end-to-end via AgentRecorder ───────────────────────────────────

def test_sync_responses_record_end_to_end(tmp_path):
    canned = _make_response("done thinking")

    with mock.patch.object(_Responses, "create", mock.Mock(return_value=canned)):
        with PatchSet():
            client = openai.OpenAI(api_key="test-key")
            with AgentRecorder("responses_rec", snapshot_dir=str(tmp_path)) as rec:
                client.responses.create(
                    model="gpt-4o-mini",
                    input="what's the weather in SF?",
                    instructions="be terse",
                )
                rec.output = "done"

    snap = read_snapshot("responses_rec", str(tmp_path))
    llm = [e for e in snap["trace"] if e["type"] == "llm_call"][0]
    assert llm["response"] == "done thinking"
    assert llm["tokens"] == 15
    assert llm["messages"] == [
        {"role": "system", "content": "be terse"},
        {"role": "user", "content": "what's the weather in SF?"},
    ]
    assert llm["tool_requests"] == [{"name": "get_weather", "args": {"city": "SF"}}]
    assert llm["raw_response"] is not None


def test_sync_responses_patcher_noop_without_accumulator():
    canned = _make_response()
    with mock.patch.object(_Responses, "create", mock.Mock(return_value=canned)):
        with PatchSet():
            client = openai.OpenAI(api_key="test-key")
            assert TraceAccumulator.current() is None
            result = client.responses.create(model="gpt-4o-mini", input="hi")
    assert result is canned


# ── Async record ────────────────────────────────────────────────────────────────

def test_async_responses_patcher_captures_llm_call():
    canned = _make_response("async reply")

    async def _run():
        acc, token = _make_acc()
        try:
            with mock.patch.object(
                _AsyncResponses, "create", mock.AsyncMock(return_value=canned)
            ):
                with PatchSet():
                    client = openai.AsyncOpenAI(api_key="test-key")
                    await client.responses.create(model="gpt-4o-mini", input="hello")
        finally:
            _accumulator_var.reset(token)
        return acc.trace

    events = asyncio.run(_run())
    assert len(events) == 1
    assert events[0]["type"] == "llm_call"
    assert events[0]["response"] == "async reply"
    assert events[0]["tokens"] == 15
    assert events[0]["messages"] == [{"role": "user", "content": "hello"}]
    assert events[0]["raw_response"] is not None
    assert events[0]["tool_requests"] == [{"name": "get_weather", "args": {"city": "SF"}}]


def test_async_responses_patcher_noop_without_accumulator():
    canned = _make_response()

    async def _run():
        assert TraceAccumulator.current() is None
        with mock.patch.object(
            _AsyncResponses, "create", mock.AsyncMock(return_value=canned)
        ):
            with PatchSet():
                client = openai.AsyncOpenAI(api_key="test-key")
                return await client.responses.create(model="gpt-4o-mini", input="hi")

    result = asyncio.run(_run())
    assert result is canned


# ── stream=True passes through unrecorded ──────────────────────────────────────

def test_sync_responses_stream_passes_through_unrecorded():
    sentinel = object()
    with mock.patch.object(_Responses, "create", mock.Mock(return_value=sentinel)):
        with PatchSet():
            acc, token = _make_acc()
            try:
                client = openai.OpenAI(api_key="test-key")
                result = client.responses.create(
                    model="gpt-4o-mini", input="hi", stream=True
                )
            finally:
                _accumulator_var.reset(token)
    assert result is sentinel
    assert acc.trace == []


def test_async_responses_stream_passes_through_unrecorded():
    sentinel = object()

    async def _run():
        acc, token = _make_acc()
        try:
            with mock.patch.object(
                _AsyncResponses, "create", mock.AsyncMock(return_value=sentinel)
            ):
                with PatchSet():
                    client = openai.AsyncOpenAI(api_key="test-key")
                    result = await client.responses.create(
                        model="gpt-4o-mini", input="hi", stream=True
                    )
        finally:
            _accumulator_var.reset(token)
        return result, acc.trace

    result, trace = asyncio.run(_run())
    assert result is sentinel
    assert trace == []


# ── Replay round trip ──────────────────────────────────────────────────────────

def test_sync_responses_replay_never_calls_original():
    def _exploding_original(self, *args, **kwargs):
        raise AssertionError("original should never be called during replay")

    event = {
        "response": "recorded reply",
        "tokens": 12,
        "raw_response": _canned_response_dict("recorded reply", tokens=12),
        "step": 0,
    }

    acc, token = _make_acc(replay=_FakeReplaySession([event]))
    try:
        with mock.patch.object(_Responses, "create", _exploding_original):
            with PatchSet():
                client = openai.OpenAI(api_key="test-key")
                result = client.responses.create(model="gpt-4o-mini", input="hi")
    finally:
        _accumulator_var.reset(token)

    assert result.output_text == "recorded reply"
    assert acc.trace[0]["response"] == "recorded reply"


def test_async_responses_replay_never_awaits_original():
    async def _exploding_original(self, *args, **kwargs):
        raise AssertionError("original should never be awaited during replay")

    event = {
        "response": "recorded reply",
        "tokens": 12,
        "raw_response": _canned_response_dict("recorded reply", tokens=12),
        "step": 0,
    }

    async def _run():
        acc, token = _make_acc(replay=_FakeReplaySession([event]))
        try:
            with mock.patch.object(_AsyncResponses, "create", _exploding_original):
                with PatchSet():
                    client = openai.AsyncOpenAI(api_key="test-key")
                    return await client.responses.create(model="gpt-4o-mini", input="hi")
        finally:
            _accumulator_var.reset(token)

    result = asyncio.run(_run())
    assert result.output_text == "recorded reply"


def test_sync_responses_replay_shape_mismatch_raises():
    """Recording is non-stream but the caller asks for stream=True."""
    def _exploding_original(self, *args, **kwargs):
        raise AssertionError("original should never be called during replay")

    event = {
        "response": "recorded reply",
        "tokens": 12,
        "raw_response": _canned_response_dict("recorded reply", tokens=12),
        "step": 0,
    }

    acc, token = _make_acc(replay=_FakeReplaySession([event]))
    try:
        with mock.patch.object(_Responses, "create", _exploding_original):
            with PatchSet():
                client = openai.OpenAI(api_key="test-key")
                with pytest.raises(ReplayError):
                    client.responses.create(model="gpt-4o-mini", input="hi", stream=True)
    finally:
        _accumulator_var.reset(token)


# ── Corrupt raw_response -> ReplayError with re-record hint ────────────────────

def test_reconstruct_response_event_corrupt_raises_replay_error_with_hint():
    event = {"raw_response": {"not": "a valid response"}, "step": 3}
    with pytest.raises(ReplayError) as exc_info:
        reconstruct_response_event(event)
    message = str(exc_info.value)
    assert "3" in message
    assert "pytest --agentsnap-record" in message


def test_reconstruct_response_event_names_cross_api_mismatch():
    """A chat.completion payload fed to the Responses reconstructor names the mismatch."""
    event = {
        "raw_response": {"object": "chat.completion", "id": "chatcmpl-1"},
        "step": 4,
    }
    with pytest.raises(ReplayError) as exc_info:
        reconstruct_response_event(event)
    message = str(exc_info.value)
    assert "chat.completion" in message
    assert "call order" in message


def test_reconstruct_event_names_cross_api_mismatch():
    """A Responses API payload fed to the Chat Completions reconstructor names the mismatch."""
    from agentsnap.adapters.openai import reconstruct_event

    event = {
        "raw_response": {"object": "response", "id": "resp-1"},
        "step": 5,
    }
    with pytest.raises(ReplayError) as exc_info:
        reconstruct_event(event)
    message = str(exc_info.value)
    assert "response" in message
    assert "call order" in message


def test_reconstruct_response_lenient_on_schema_drift():
    """A dump that fails strict validation (SDK schema evolution) still reconstructs."""
    from agentsnap.adapters.openai import reconstruct_response

    drifted = {
        "id": "r1", "object": "response", "created_at": 1.0, "model": "m",
        "parallel_tool_calls": True, "tool_choice": "auto", "tools": [],
        "output": [{"type": "message", "id": "m1", "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": "hi", "annotations": []}]}],
        # input_tokens_details lacks cache_write_tokens, required by newer schemas
        "usage": {"input_tokens": 1, "output_tokens": 2, "total_tokens": 3,
                  "input_tokens_details": {"cached_tokens": 0},
                  "output_tokens_details": {"reasoning_tokens": 0}},
    }
    resp = reconstruct_response(drifted)
    assert resp.output_text == "hi"


def test_reconstruct_response_still_rejects_garbage():
    """The lenient fallback must not swallow genuinely corrupt payloads."""
    import pytest as _pytest

    from agentsnap.adapters.openai import reconstruct_response_event
    from agentsnap.exceptions import ReplayError

    with _pytest.raises(ReplayError, match="agentsnap-record"):
        reconstruct_response_event({"step": 0, "raw_response": {"nonsense": True}})
