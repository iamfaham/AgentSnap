"""Direct unit coverage for the raw-response wrapper helpers in
agentsnap/adapters/openai.py: wants_raw_response, unwrap_legacy_response,
ReplayLegacyResponse, RawResponseStreamShim, and the Fix-1 record path for
the Responses API (langchain-openai's use_responses_api=True route calls
responses.with_raw_response.create(), which must be unwrapped the same way
the chat route is).

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
