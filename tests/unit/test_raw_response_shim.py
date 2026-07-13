"""Direct unit coverage for the raw-response wrapper helpers, now shared in
agentsnap/adapters/_raw_response.py (re-exported from openai.py and
anthropic.py): wants_raw_response, unwrap_legacy_response,
ReplayLegacyResponse, RawResponseStreamShim, and the Fix-1 record path for
the Responses API (langchain-openai's use_responses_api=True route calls
responses.with_raw_response.create(), which must be unwrapped the same way
the chat route is), plus the anthropic-side parity wiring in
agentsnap/patches.py (_apply_anthropic).

No frameworks needed — these exercise the adapter helpers directly with
lightweight fakes.
"""
from __future__ import annotations

import warnings

import pytest

from agentsnap.adapters.openai import (
    RawResponseStreamShim,
    ReplayLegacyResponse,
    unwrap_legacy_response,
    wants_raw_response,
)


# ── wants_raw_response ──────────────────────────────────────────────────────

def test_wants_raw_response_true_when_header_present():
    assert wants_raw_response({"extra_headers": {"X-Stainless-Raw-Response": "true"}})


def test_wants_raw_response_false_when_header_absent():
    assert wants_raw_response({}) is False
    assert wants_raw_response({"extra_headers": {}}) is False


def test_wants_raw_response_false_for_other_header_values():
    assert wants_raw_response({"extra_headers": {"X-Stainless-Raw-Response": "stream"}}) is False


# ── unwrap_legacy_response ──────────────────────────────────────────────────

class _PlainChatResponse:
    choices = []


class _PlainResponsesResponse:
    output = []


def test_unwrap_legacy_response_identity_for_plain_chat_response():
    resp = _PlainChatResponse()
    assert unwrap_legacy_response(resp) is resp


def test_unwrap_legacy_response_identity_for_plain_responses_response():
    resp = _PlainResponsesResponse()
    assert unwrap_legacy_response(resp) is resp


class _LegacyWrapper:
    def __init__(self, parsed) -> None:
        self._parsed = parsed

    def parse(self):
        return self._parsed


def test_unwrap_legacy_response_unwraps_object_exposing_parse():
    parsed = object()
    wrapper = _LegacyWrapper(parsed)
    assert unwrap_legacy_response(wrapper) is parsed


class _ExplodingLegacyWrapper:
    def parse(self):
        raise RuntimeError("boom")


def test_unwrap_legacy_response_returns_original_when_parse_raises():
    wrapper = _ExplodingLegacyWrapper()
    with pytest.warns(UserWarning, match="failed to unwrap"):
        result = unwrap_legacy_response(wrapper)
    assert result is wrapper


def test_unwrap_legacy_response_no_warning_on_success():
    parsed = object()
    wrapper = _LegacyWrapper(parsed)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert unwrap_legacy_response(wrapper) is parsed


# ── ReplayLegacyResponse ─────────────────────────────────────────────────────

class _Parsed:
    def __init__(self) -> None:
        self.output_text = "hello"


def test_replay_legacy_response_parse_returns_inner():
    parsed = _Parsed()
    legacy = ReplayLegacyResponse(parsed)
    assert legacy.parse() is parsed


def test_replay_legacy_response_forwards_other_attrs():
    parsed = _Parsed()
    legacy = ReplayLegacyResponse(parsed)
    assert legacy.output_text == "hello"


# ── RawResponseStreamShim ────────────────────────────────────────────────────

class _Legacy:
    headers = {"x-request-id": "abc"}


def test_raw_response_stream_shim_parse_returns_tee():
    tee = object()
    legacy = _Legacy()
    shim = RawResponseStreamShim(tee, legacy)
    assert shim.parse() is tee


def test_raw_response_stream_shim_forwards_other_attrs_to_legacy():
    tee = object()
    legacy = _Legacy()
    shim = RawResponseStreamShim(tee, legacy)
    assert shim.headers == {"x-request-id": "abc"}


# ── Fix-1 record path: Responses API unwrap in _apply_openai_responses ─────

def test_apply_openai_responses_record_path_unwraps_raw_response(tmp_path):
    """A fake with_raw_response wrapper around a Responses API response must
    be unwrapped so the recorded event has real text/raw_response, not the
    empty defaults a naive extraction from the wrapper would produce."""
    import unittest.mock as mock

    import openai
    from openai.resources.responses.responses import Responses as _Responses
    from openai.types.responses import Response

    from agentsnap.core.recorder import AgentRecorder
    from agentsnap.core.snapshot import read_snapshot
    from agentsnap.patches import PatchSet

    canned_dict = {
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
                    {"type": "output_text", "text": "wrapped response", "annotations": []},
                ],
            },
        ],
        "usage": {
            "input_tokens": 10,
            "output_tokens": 5,
            "total_tokens": 15,
            "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 0},
        },
    }
    parsed = Response.model_validate(canned_dict)

    class _FakeLegacyResponse:
        """Mimics with_raw_response.create()'s LegacyAPIResponse: no
        .choices/.output of its own, just a .parse() that yields the real
        parsed Response."""

        def parse(self):
            return parsed

    fake_wrapped = _FakeLegacyResponse()

    with mock.patch.object(_Responses, "create", mock.Mock(return_value=fake_wrapped)):
        with PatchSet():
            client = openai.OpenAI(api_key="test-key")
            with AgentRecorder("responses_raw_unwrap", snapshot_dir=str(tmp_path)) as rec:
                client.responses.create(
                    model="gpt-4o-mini",
                    input="hi",
                    extra_headers={"X-Stainless-Raw-Response": "true"},
                )
                rec.output = "done"

    snap = read_snapshot("responses_raw_unwrap", str(tmp_path))
    llm = [e for e in snap["trace"] if e["type"] == "llm_call"][0]
    assert llm["response"] == "wrapped response"
    assert llm["tokens"] == 15
    assert llm["raw_response"] is not None
    assert llm["raw_response"]["output"][0]["content"][0]["text"] == "wrapped response"


# ── Anthropic-side parity: same helpers, shared module, wired in patches.py ──
# The anthropic SDK has the same with_raw_response/X-Stainless-Raw-Response
# mechanism as openai, used by e.g. langchain-anthropic. These tests mirror
# the openai record-path/replay coverage above but exercise _apply_anthropic
# (agentsnap/patches.py) directly with fake wrapped responses.


def test_apply_anthropic_record_path_unwraps_raw_response(tmp_path):
    """A fake with_raw_response wrapper around an Anthropic Message response
    must be unwrapped so the recorded event has real text/raw_response, not
    the empty defaults a naive extraction from the wrapper would produce."""
    import unittest.mock as mock

    import anthropic
    from anthropic.resources.messages.messages import Messages as _Messages
    from anthropic.types import Message

    from agentsnap.core.recorder import AgentRecorder
    from agentsnap.core.snapshot import read_snapshot
    from agentsnap.patches import PatchSet

    canned_dict = {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "claude-3-5-sonnet-20241022",
        "content": [{"type": "text", "text": "wrapped anthropic response"}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }
    parsed = Message.model_validate(canned_dict)

    class _FakeLegacyResponse:
        """Mimics with_raw_response.create()'s LegacyAPIResponse: no
        .content of its own, just a .parse() that yields the real parsed
        Message."""

        def parse(self):
            return parsed

    fake_wrapped = _FakeLegacyResponse()

    with mock.patch.object(_Messages, "create", mock.Mock(return_value=fake_wrapped)):
        with PatchSet():
            client = anthropic.Anthropic(api_key="test-key")
            with AgentRecorder("anthropic_raw_unwrap", snapshot_dir=str(tmp_path)) as rec:
                client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=100,
                    messages=[{"role": "user", "content": "hi"}],
                    extra_headers={"X-Stainless-Raw-Response": "true"},
                )
                rec.output = "done"

    snap = read_snapshot("anthropic_raw_unwrap", str(tmp_path))
    llm = [e for e in snap["trace"] if e["type"] == "llm_call"][0]
    assert llm["response"] == "wrapped anthropic response"
    assert llm["tokens"] == 15
    assert llm["raw_response"] is not None
    assert llm["raw_response"]["content"][0]["text"] == "wrapped anthropic response"


def test_apply_anthropic_replay_returns_legacy_response_when_raw_wanted(tmp_path):
    """During replay, a caller requesting with_raw_response (via the
    X-Stainless-Raw-Response header) must get back a .parse()-capable
    ReplayLegacyResponse instead of the bare reconstructed Message — there's
    no real LegacyAPIResponse in replay since no HTTP call is made."""
    import unittest.mock as mock

    import anthropic
    from anthropic.resources.messages.messages import Messages as _Messages
    from anthropic.types import Message

    from agentsnap.core.asserter import AgentAsserter
    from agentsnap.core.recorder import AgentRecorder
    from agentsnap.patches import PatchSet

    canned_dict = {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "claude-3-5-sonnet-20241022",
        "content": [{"type": "text", "text": "wrapped anthropic response"}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }
    parsed = Message.model_validate(canned_dict)

    class _FakeLegacyResponse:
        def parse(self):
            return parsed

    def _identical_embed(texts):
        return [[1.0, 0.0] for _ in texts]

    with mock.patch.object(_Messages, "create", mock.Mock(return_value=_FakeLegacyResponse())):
        with PatchSet():
            client = anthropic.Anthropic(api_key="test-key")
            with AgentRecorder("anthropic_replay_raw", snapshot_dir=str(tmp_path)) as rec:
                client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=100,
                    messages=[{"role": "user", "content": "hi"}],
                    extra_headers={"X-Stainless-Raw-Response": "true"},
                )
                rec.output = "done"

    def _raising_create(*_args, **_kwargs):
        raise AssertionError("wire touched during replay — should have short-circuited")

    with mock.patch.object(_Messages, "create", mock.Mock(side_effect=_raising_create)):
        with PatchSet():
            client = anthropic.Anthropic(api_key="test-key")
            with AgentAsserter(
                "anthropic_replay_raw",
                snapshot_dir=str(tmp_path),
                mode="replay",
                embed_fn=_identical_embed,
            ) as a:
                result = client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=100,
                    messages=[{"role": "user", "content": "hi"}],
                    extra_headers={"X-Stainless-Raw-Response": "true"},
                )
                assert hasattr(result, "parse")
                replayed = result.parse()
                assert replayed.content[0].text == "wrapped anthropic response"
                a.output = "done"


def test_apply_anthropic_stream_record_unwraps_raw_response_and_returns_shim(tmp_path):
    """A raw-response-wrapped streaming call must be unwrapped before teeing
    so the tee iterates the real stream, and the caller must get back a
    RawResponseStreamShim whose .parse() yields the (recording) tee."""
    import unittest.mock as mock

    import anthropic
    from anthropic.resources.messages.messages import Messages as _Messages

    from agentsnap.core.recorder import AgentRecorder
    from agentsnap.core.snapshot import read_snapshot
    from agentsnap.patches import PatchSet

    class _Delta:
        def __init__(self, text) -> None:
            self.text = text

    class _Usage:
        def __init__(self, input_tokens=0, output_tokens=0) -> None:
            self.input_tokens = input_tokens
            self.output_tokens = output_tokens

    class _Message:
        def __init__(self, usage) -> None:
            self.usage = usage

    class _FakeEvent:
        def __init__(self, etype, **kw) -> None:
            self.type = etype
            for k, v in kw.items():
                setattr(self, k, v)

        def model_dump(self, mode="json"):
            return {"type": self.type}

    events = [
        _FakeEvent("message_start", message=_Message(_Usage(input_tokens=10))),
        _FakeEvent("content_block_delta", delta=_Delta("wrapped stream text")),
        _FakeEvent("message_delta", usage=_Usage(output_tokens=5)),
    ]

    class _FakeStream:
        def __init__(self, items) -> None:
            self._items = items

        def __iter__(self):
            return iter(self._items)

        def close(self) -> None:
            pass

    real_stream = _FakeStream(events)

    class _FakeLegacyStreamResponse:
        headers = {"x-request-id": "abc"}

        def parse(self):
            return real_stream

    fake_wrapped = _FakeLegacyStreamResponse()

    with mock.patch.object(_Messages, "create", mock.Mock(return_value=fake_wrapped)):
        with PatchSet():
            client = anthropic.Anthropic(api_key="test-key")
            with AgentRecorder("anthropic_stream_raw_unwrap", snapshot_dir=str(tmp_path)) as rec:
                result = client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=100,
                    messages=[{"role": "user", "content": "hi"}],
                    stream=True,
                    extra_headers={"X-Stainless-Raw-Response": "true"},
                )
                assert hasattr(result, "parse")
                assert result.headers == {"x-request-id": "abc"}
                tee = result.parse()
                list(tee)  # exhaust the tee to trigger recording
                rec.output = "done"

    snap = read_snapshot("anthropic_stream_raw_unwrap", str(tmp_path))
    llm = [e for e in snap["trace"] if e["type"] == "llm_call"][0]
    assert llm["response"] == "wrapped stream text"
    assert llm["tokens"] == 15
