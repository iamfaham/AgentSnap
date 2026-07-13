"""Real-framework verification: LangChain's ChatAnthropic drives its genuine
code path (sync AND async Anthropic messages clients) through agentsnap's
PatchSet interceptors against an offline httpx.MockTransport fake.

No network, no API keys. Skips entirely if langchain-anthropic isn't
installed (``pip install -e ".[frameworks]"``).

Note: unlike langchain-openai, the installed langchain-anthropic (1.4.8)
does not call ``with_raw_response``/``.parse()`` anywhere in its
``_create``/``_acreate`` methods (verified by grepping the installed
package) — it calls ``self._client.messages.create(**payload)`` directly.
So this suite doesn't exercise the with_raw_response unwrap path the way
test_langchain.py's Responses-route test does for openai; it verifies the
genuine record/replay round trip through the real ChatAnthropic code path,
which is the framework-compatibility contract this fix is meant to protect
going forward (agentsnap/adapters/anthropic.py + patches.py now unwrap
with_raw_response defensively, mirroring openai, in case a future
langchain-anthropic release adopts the same mechanism).

ChatAnthropic builds its ``anthropic.Client``/``anthropic.AsyncClient`` lazily
via cached_property (``_client``/``_async_client``), with no constructor
param for injecting an http client. Reading the installed source
(site-packages/langchain_anthropic/chat_models.py) confirms these are plain
``functools.cached_property`` (non-data descriptors), so setting
``chat._client`` / ``chat._async_client`` directly on the instance before
first access pre-empts the cached computation — verified interactively
against a live ChatAnthropic instance before writing these tests.
"""
from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("langchain_anthropic")

pytestmark = pytest.mark.frameworks

import anthropic
import httpx
from anthropic.types import Message
from langchain_anthropic import ChatAnthropic

from agentsnap.core.asserter import AgentAsserter
from agentsnap.core.recorder import AgentRecorder
from agentsnap.core.snapshot import read_snapshot
from agentsnap.patches import PatchSet


# ── Canned Message payload, built from the real SDK type ──────────────────────

def _canned_payload(text: str) -> dict:
    d = {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "claude-3-5-sonnet-20241022",
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }
    return Message.model_validate(d).model_dump(mode="json")


def _make_chat_with_handler(handler) -> ChatAnthropic:
    chat = ChatAnthropic(api_key="test", model="claude-3-5-sonnet-20241022")
    sync_client = httpx.Client(transport=httpx.MockTransport(handler))
    chat._client = anthropic.Anthropic(api_key="test", http_client=sync_client)
    return chat


def _make_async_chat_with_handler(handler) -> ChatAnthropic:
    chat = ChatAnthropic(api_key="test", model="claude-3-5-sonnet-20241022")
    async_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    chat._async_client = anthropic.AsyncAnthropic(api_key="test", http_client=async_client)
    return chat


def _make_chat(text: str) -> ChatAnthropic:
    payload = _canned_payload(text)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    return _make_chat_with_handler(handler)


def _make_async_chat(text: str) -> ChatAnthropic:
    payload = _canned_payload(text)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    return _make_async_chat_with_handler(handler)


def _raising_handler(request: httpx.Request) -> httpx.Response:
    raise AssertionError("live wire call made during replay — PatchSet should have short-circuited")


def _identical_embed(texts):  # noqa: ARG001 - stub signature must match embed_fn contract
    """Deterministic offline stand-in for the real embedder: byte-identical
    outputs already short-circuit semantic scoring, but this guarantees the
    replay test never loads sentence-transformers even if that changes."""
    return [[1.0, 0.0] for _ in texts]


# ── Sync path: .invoke() ────────────────────────────────────────────────────

def test_langchain_anthropic_sync_invoke_records_llm_call(tmp_path):
    chat = _make_chat("hi from langchain-anthropic sync")

    with PatchSet():
        with AgentRecorder("lc_anthropic_sync", snapshot_dir=str(tmp_path)) as rec:
            result = chat.invoke("hi")
            rec.output = result.content

    snap = read_snapshot("lc_anthropic_sync", str(tmp_path))
    llm_calls = [e for e in snap["trace"] if e["type"] == "llm_call"]
    assert len(llm_calls) >= 1
    assert llm_calls[0]["response"] == "hi from langchain-anthropic sync"
    assert llm_calls[0]["raw_response"] is not None


# ── Async path: .ainvoke() ──────────────────────────────────────────────────

def test_langchain_anthropic_async_ainvoke_records_llm_call(tmp_path):
    chat = _make_async_chat("hi from langchain-anthropic async")

    async def _run():
        with PatchSet():
            with AgentRecorder("lc_anthropic_async", snapshot_dir=str(tmp_path)) as rec:
                result = await chat.ainvoke("hi")
                rec.output = result.content

    asyncio.run(_run())

    snap = read_snapshot("lc_anthropic_async", str(tmp_path))
    llm_calls = [e for e in snap["trace"] if e["type"] == "llm_call"]
    assert len(llm_calls) >= 1
    assert llm_calls[0]["response"] == "hi from langchain-anthropic async"
    assert llm_calls[0]["raw_response"] is not None


# ── Replay round-trip: recorded raw_response must reconstruct offline ──────────

def test_langchain_anthropic_replay_never_touches_wire(tmp_path):
    chat = _make_chat("hi from langchain-anthropic replay")

    with PatchSet():
        with AgentRecorder("lc_anthropic_replay", snapshot_dir=str(tmp_path)) as rec:
            result = chat.invoke("hi")
            rec.output = result.content

    # Replay: the transport raises if the wire is ever touched. If the
    # recorded raw_response fails to reconstruct, agentsnap falls back to
    # making a real call and this test fails with the AssertionError below.
    replay_chat = _make_chat_with_handler(_raising_handler)
    with PatchSet():
        with AgentAsserter(
            "lc_anthropic_replay",
            snapshot_dir=str(tmp_path),
            mode="replay",
            embed_fn=_identical_embed,
        ) as a:
            result = replay_chat.invoke("hi")
            a.output = result.content


def test_langchain_anthropic_async_ainvoke_replay_never_touches_wire(tmp_path):
    """Mirrors test_langchain_anthropic_replay_never_touches_wire for the
    async ainvoke() path."""
    chat = _make_async_chat("hi from langchain-anthropic async replay")

    async def _record():
        with PatchSet():
            with AgentRecorder("lc_anthropic_async_replay", snapshot_dir=str(tmp_path)) as rec:
                result = await chat.ainvoke("hi")
                rec.output = result.content

    asyncio.run(_record())

    replay_chat = _make_async_chat_with_handler(_raising_handler)

    async def _replay():
        with PatchSet():
            with AgentAsserter(
                "lc_anthropic_async_replay",
                snapshot_dir=str(tmp_path),
                mode="replay",
                embed_fn=_identical_embed,
            ) as a:
                result = await replay_chat.ainvoke("hi")
                a.output = result.content

    asyncio.run(_replay())
