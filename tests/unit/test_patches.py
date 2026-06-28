from __future__ import annotations

import unittest.mock as mock

import anthropic
import openai
import pytest

from agentsnap.core.recorder import TraceAccumulator, _accumulator_var
from agentsnap.patches import PatchSet
from anthropic.resources.messages.messages import Messages as _AnthMessages
from openai.resources.chat.completions.completions import Completions as _OAICompletions


# ── Fake response shapes ──────────────────────────────────────────────────────

class _AnthContent:
    text = "anthropic patched response"

class _AnthResp:
    content = [_AnthContent()]
    class usage:
        input_tokens = 5
        output_tokens = 10

class _OAIMessage:
    content = "openai patched response"

class _OAIChoice:
    message = _OAIMessage()

class _OAIResp:
    choices = [_OAIChoice()]
    class usage:
        total_tokens = 20


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_acc():
    acc = TraceAccumulator()
    token = _accumulator_var.set(acc)
    return acc, token


# ── PatchSet lifecycle ────────────────────────────────────────────────────────

def test_patchset_restores_anthropic_on_exit():
    original = _AnthMessages.create
    with PatchSet():
        assert _AnthMessages.create is not original
    assert _AnthMessages.create is original


def test_patchset_restores_openai_on_exit():
    original = _OAICompletions.create
    with PatchSet():
        assert _OAICompletions.create is not original
    assert _OAICompletions.create is original


def test_patchset_restores_on_exception():
    """Originals are restored even when an exception is raised inside the block."""
    original_anth = _AnthMessages.create
    original_oai = _OAICompletions.create
    try:
        with PatchSet():
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert _AnthMessages.create is original_anth
    assert _OAICompletions.create is original_oai


def test_patchset_can_be_entered_twice_sequentially():
    """Two sequential PatchSet uses don't leave the class permanently patched."""
    original = _AnthMessages.create
    with PatchSet():
        pass
    with PatchSet():
        pass
    assert _AnthMessages.create is original


# ── Anthropic patcher ─────────────────────────────────────────────────────────

def test_anthropic_patcher_captures_llm_call():
    acc, token = _make_acc()
    try:
        with mock.patch.object(_AnthMessages, "create", return_value=_AnthResp()):
            with PatchSet():
                client = anthropic.Anthropic(api_key="test-key-12345")
                client.messages.create(
                    model="claude-haiku-4-5",
                    messages=[{"role": "user", "content": "hello"}],
                    max_tokens=10,
                )
    finally:
        _accumulator_var.reset(token)

    events = acc.trace
    assert len(events) == 1
    assert events[0]["type"] == "llm_call"
    assert events[0]["response"] == "anthropic patched response"
    assert events[0]["tokens"] == 15
    assert events[0]["messages"] == [{"role": "user", "content": "hello"}]


def test_anthropic_patcher_noop_without_accumulator():
    """Patched method must be transparent when no TraceAccumulator is active."""
    assert TraceAccumulator.current() is None
    with mock.patch.object(_AnthMessages, "create", return_value=_AnthResp()):
        with PatchSet():
            client = anthropic.Anthropic(api_key="test-key")
            result = client.messages.create(
                model="test", messages=[{"role": "user", "content": "hi"}], max_tokens=5
            )
    assert result is not None  # forwarded the fake response unchanged


# ── OpenAI patcher ────────────────────────────────────────────────────────────

def test_openai_patcher_captures_llm_call():
    acc, token = _make_acc()
    try:
        with mock.patch.object(_OAICompletions, "create", return_value=_OAIResp()):
            with PatchSet():
                client = openai.OpenAI(api_key="test-key-12345")
                client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": "hello"}],
                    max_tokens=10,
                )
    finally:
        _accumulator_var.reset(token)

    events = acc.trace
    assert len(events) == 1
    assert events[0]["type"] == "llm_call"
    assert events[0]["response"] == "openai patched response"
    assert events[0]["tokens"] == 20


def test_openai_patcher_forces_stream_false():
    """OpenAI patcher must set stream=False to get a complete ChatCompletion."""
    captured_kwargs = {}

    def _spy_create(self, **kwargs):
        captured_kwargs.update(kwargs)
        return _OAIResp()

    acc, token = _make_acc()
    try:
        with mock.patch.object(_OAICompletions, "create", _spy_create):
            with PatchSet():
                client = openai.OpenAI(api_key="test-key")
                client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": "hi"}],
                    max_tokens=5,
                    stream=True,  # user asked for streaming
                )
    finally:
        _accumulator_var.reset(token)

    assert captured_kwargs.get("stream") is False


# ── Graceful skip for uninstalled SDKs ───────────────────────────────────────

def test_patchset_skips_uninstalled_sdk(monkeypatch):
    """If an SDK is not installed, its patcher is silently skipped."""
    import agentsnap.patches as _patches

    def _always_fails():
        raise ImportError("no such module")

    monkeypatch.setattr(_patches, "_apply_gemini", _always_fails)
    monkeypatch.setattr(_patches, "_apply_cohere", _always_fails)
    monkeypatch.setattr(_patches, "_apply_mistral", _always_fails)

    # Should not raise even though three patchers fail
    original_anth = _AnthMessages.create
    with PatchSet():
        assert _AnthMessages.create is not original_anth  # Anthropic still patched
    assert _AnthMessages.create is original_anth


def test_patchset_importable_from_top_level():
    from agentsnap import PatchSet as _PS
    assert _PS is PatchSet
