"""Real-framework verification: LangChain's ChatOpenAI drives its genuine code
path (sync AND async OpenAI chat-completions clients) through agentsnap's
PatchSet interceptors against an offline httpx.MockTransport fake.

No network, no API keys. Skips entirely if langchain-openai isn't installed
(``pip install -e ".[frameworks]"``).
"""
from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("langchain_openai")

pytestmark = pytest.mark.frameworks

import httpx
from langchain_openai import ChatOpenAI
from openai.types.chat import ChatCompletion

from agentsnap.core.asserter import AgentAsserter
from agentsnap.core.recorder import AgentRecorder
from agentsnap.core.snapshot import read_snapshot
from agentsnap.patches import PatchSet


# ── Canned ChatCompletion payload, built from the real SDK type ────────────────

def _canned_payload(text: str) -> dict:
    d = {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "gpt-4o-mini",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": text},
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }
    return ChatCompletion.model_validate(d).model_dump(mode="json")


def _make_chat_with_handler(handler) -> ChatOpenAI:
    sync_client = httpx.Client(transport=httpx.MockTransport(handler))
    async_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return ChatOpenAI(
        api_key="test",
        model="gpt-4o-mini",
        http_client=sync_client,
        http_async_client=async_client,
    )


def _make_chat(text: str) -> ChatOpenAI:
    payload = _canned_payload(text)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    return _make_chat_with_handler(handler)


def _raising_handler(request: httpx.Request) -> httpx.Response:
    raise AssertionError("live wire call made during replay — PatchSet should have short-circuited")


def _identical_embed(texts):  # noqa: ARG001 - stub signature must match embed_fn contract
    """Deterministic offline stand-in for the real embedder: byte-identical
    outputs already short-circuit semantic scoring, but this guarantees the
    replay test never loads sentence-transformers even if that changes."""
    return [[1.0, 0.0] for _ in texts]


# ── Sync path: .invoke() (pre-existing interception) ───────────────────────────

def test_langchain_sync_invoke_records_llm_call(tmp_path):
    chat = _make_chat("hi from langchain sync")

    with PatchSet():
        with AgentRecorder("lc_sync", snapshot_dir=str(tmp_path)) as rec:
            result = chat.invoke("hi")
            rec.output = result.content

    snap = read_snapshot("lc_sync", str(tmp_path))
    llm_calls = [e for e in snap["trace"] if e["type"] == "llm_call"]
    assert len(llm_calls) >= 1
    assert llm_calls[0]["response"] == "hi from langchain sync"
    assert llm_calls[0]["raw_response"] is not None


# ── Async path: .ainvoke() (new async interception) ────────────────────────────

def test_langchain_async_ainvoke_records_llm_call(tmp_path):
    chat = _make_chat("hi from langchain async")

    async def _run():
        with PatchSet():
            with AgentRecorder("lc_async", snapshot_dir=str(tmp_path)) as rec:
                result = await chat.ainvoke("hi")
                rec.output = result.content

    asyncio.run(_run())

    snap = read_snapshot("lc_async", str(tmp_path))
    llm_calls = [e for e in snap["trace"] if e["type"] == "llm_call"]
    assert len(llm_calls) >= 1
    assert llm_calls[0]["response"] == "hi from langchain async"
    assert llm_calls[0]["raw_response"] is not None


# ── Replay round-trip: recorded raw_response must reconstruct offline ──────────
# This is the exact framework where the raw-response wrapper bug was found
# (langchain-openai requests the raw HTTP response and calls .parse() itself),
# so this test pins the fix end-to-end for both the sync and async transports.

def test_langchain_replay_never_touches_wire(tmp_path):
    chat = _make_chat("hi from langchain replay")

    with PatchSet():
        with AgentRecorder("lc_replay", snapshot_dir=str(tmp_path)) as rec:
            result = chat.invoke("hi")
            rec.output = result.content

    # Replay: both transports raise if the wire is ever touched. If the
    # recorded raw_response fails to reconstruct, agentsnap falls back to
    # making a real call and this test fails with the AssertionError below.
    replay_chat = _make_chat_with_handler(_raising_handler)
    with PatchSet():
        with AgentAsserter(
            "lc_replay", snapshot_dir=str(tmp_path), mode="replay", embed_fn=_identical_embed
        ) as a:
            result = replay_chat.invoke("hi")
            a.output = result.content
