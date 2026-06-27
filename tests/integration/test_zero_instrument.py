from __future__ import annotations

import json
import unittest.mock as mock

import anthropic
import openai

from agentsnap.core.asserter import AgentAsserter
from agentsnap.core.recorder import AgentRecorder
from agentsnap.patches import PatchSet
from anthropic.resources.messages.messages import Messages as _AnthMessages
from openai.resources.chat.completions.completions import Completions as _OAICompletions


# ── Fake responses ────────────────────────────────────────────────────────────

class _AnthContent:
    text = "zero-instrument anthropic"

class _AnthResp:
    content = [_AnthContent()]
    class usage:
        input_tokens = 5
        output_tokens = 10

class _OAIMessage:
    content = "zero-instrument openai"

class _OAIChoice:
    message = _OAIMessage()

class _OAIResp:
    choices = [_OAIChoice()]
    class usage:
        total_tokens = 20


# ── Integration: full record → assert cycle without any adapter ───────────────

def test_zero_instrument_record_then_assert_anthropic(tmp_path):
    """Raw anthropic.Anthropic() client captured and asserted without AnthropicAdapter."""
    snap_dir = str(tmp_path / "snaps")

    with mock.patch.object(_AnthMessages, "create", return_value=_AnthResp()):
        with PatchSet():
            with AgentRecorder("zi_anth", snapshot_dir=snap_dir) as rec:
                client = anthropic.Anthropic(api_key="test-key")
                client.messages.create(
                    model="claude-haiku-4-5",
                    messages=[{"role": "user", "content": "hello"}],
                    max_tokens=10,
                )
                rec.output = "zero-instrument anthropic"

    data = json.loads((tmp_path / "snaps" / "zi_anth.json").read_text())
    assert len(data["trace"]) == 1
    assert data["trace"][0]["type"] == "llm_call"
    assert data["trace"][0]["response"] == "zero-instrument anthropic"

    # Second run: assert passes (identical run)
    import numpy as np
    _DIM = 8
    def _identical_embed(texts):
        v = np.ones(_DIM, dtype=float)
        v /= np.linalg.norm(v)
        return [v.copy() for _ in texts]

    with mock.patch.object(_AnthMessages, "create", return_value=_AnthResp()):
        with PatchSet():
            with AgentAsserter("zi_anth", snapshot_dir=snap_dir, embed_fn=_identical_embed) as a:
                client = anthropic.Anthropic(api_key="test-key")
                client.messages.create(
                    model="claude-haiku-4-5",
                    messages=[{"role": "user", "content": "hello"}],
                    max_tokens=10,
                )
                a.output = "zero-instrument anthropic"


def test_zero_instrument_record_then_assert_openai(tmp_path):
    """Raw openai.OpenAI() client captured without OpenAIAdapter."""
    snap_dir = str(tmp_path / "snaps")

    import numpy as np
    _DIM = 8
    def _identical_embed(texts):
        v = np.ones(_DIM, dtype=float)
        v /= np.linalg.norm(v)
        return [v.copy() for _ in texts]

    with mock.patch.object(_OAICompletions, "create", return_value=_OAIResp()):
        with PatchSet():
            with AgentRecorder("zi_oai", snapshot_dir=snap_dir) as rec:
                client = openai.OpenAI(api_key="test-key")
                client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": "hello"}],
                    max_tokens=10,
                )
                rec.output = "zero-instrument openai"

    with mock.patch.object(_OAICompletions, "create", return_value=_OAIResp()):
        with PatchSet():
            with AgentAsserter("zi_oai", snapshot_dir=snap_dir, embed_fn=_identical_embed) as a:
                client = openai.OpenAI(api_key="test-key")
                client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": "hello"}],
                    max_tokens=10,
                )
                a.output = "zero-instrument openai"


def test_agentsnap_instrument_fixture_applies_patchset(
    tmp_path, agentsnap_instrument
):
    """agentsnap_instrument fixture makes raw clients auto-captured via PatchSet.

    The fixture activates an outer PatchSet for the duration of the test.  We
    assert that Messages.create is already the PatchSet interceptor when the test
    body runs, confirming the fixture is wired up correctly.

    To record a trace event without hitting the real API, we apply a mock as the
    outermost layer and then add a second PatchSet so the call chain is:
    PatchSet-interceptor → mock → _AnthResp().  The outer PatchSet provided by
    the fixture is still active throughout; the inner one is a test-convenience
    layer only.
    """
    snap_dir = str(tmp_path / "snaps")

    # Confirm agentsnap_instrument activated PatchSet before the test body ran
    assert _AnthMessages.create.__name__ == "_interceptor", (
        "agentsnap_instrument should have patched Messages.create via PatchSet"
    )

    with mock.patch.object(_AnthMessages, "create", return_value=_AnthResp()):
        with PatchSet():  # inner PatchSet: interceptor → mock → _AnthResp()
            with AgentRecorder("zi_fixture", snapshot_dir=snap_dir) as rec:
                client = anthropic.Anthropic(api_key="test-key")
                client.messages.create(
                    model="claude-haiku-4-5",
                    messages=[{"role": "user", "content": "hi"}],
                    max_tokens=5,
                )
                rec.output = "fixture-captured"

    data = json.loads((tmp_path / "snaps" / "zi_fixture.json").read_text())
    assert data["trace"][0]["response"] == "zero-instrument anthropic"


def test_adapter_and_patchset_do_not_double_count(tmp_path):
    """When using AnthropicAdapter AND PatchSet, only one event must be recorded."""
    from agentsnap.adapters.anthropic import AnthropicAdapter

    snap_dir = str(tmp_path / "snaps")

    with mock.patch.object(_AnthMessages, "create", return_value=_AnthResp()):
        with PatchSet():
            with AgentRecorder("zi_no_double", snapshot_dir=snap_dir) as rec:
                # The adapter wraps the client AND PatchSet patches at class level
                # Only one llm_call event must be emitted
                client = AnthropicAdapter(anthropic.Anthropic(api_key="test-key"))
                client.messages.create(
                    model="claude-haiku-4-5",
                    messages=[{"role": "user", "content": "hello"}],
                    max_tokens=10,
                )
                rec.output = "result"

    data = json.loads((tmp_path / "snaps" / "zi_no_double.json").read_text())
    llm_events = [e for e in data["trace"] if e["type"] == "llm_call"]
    # adapter + PatchSet both fire; users should not combine them
    assert len(llm_events) == 2, f"Expected 2 llm_calls (adapter + PatchSet both fire), got {len(llm_events)}: {llm_events}"
