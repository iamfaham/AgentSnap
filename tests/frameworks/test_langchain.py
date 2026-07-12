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


def test_langchain_async_ainvoke_replay_never_touches_wire(tmp_path):
    """Mirrors test_langchain_replay_never_touches_wire for the async
    ainvoke() path — same with_raw_response bug, async transport."""
    chat = _make_chat("hi from langchain async replay")

    async def _record():
        with PatchSet():
            with AgentRecorder("lc_async_replay", snapshot_dir=str(tmp_path)) as rec:
                result = await chat.ainvoke("hi")
                rec.output = result.content

    asyncio.run(_record())

    replay_chat = _make_chat_with_handler(_raising_handler)

    async def _replay():
        with PatchSet():
            with AgentAsserter(
                "lc_async_replay",
                snapshot_dir=str(tmp_path),
                mode="replay",
                embed_fn=_identical_embed,
            ) as a:
                result = await replay_chat.ainvoke("hi")
                a.output = result.content

    asyncio.run(_replay())


# ── Responses-route record (Fix 1): use_responses_api=True ─────────────────
# langchain-openai's Responses route calls
# responses.with_raw_response.create() directly, so the record-side unwrap
# (_apply_openai_responses) must run the same way it does for chat. A full
# replay round-trip through this route additionally hits a pre-existing
# reconstruct_response() gap (the real SDK's Responses object dumps
# usage.input_tokens_details.cache_write_tokens as null, which
# Response.model_validate then rejects as a required int) — unrelated to
# this fix, so only the record side is covered here.

def test_langchain_responses_route_record_unwraps_raw_response(tmp_path):
    payload = {
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
                    {"type": "output_text", "text": "hi from responses route", "annotations": []},
                ],
            }
        ],
        "usage": {
            "input_tokens": 10,
            "output_tokens": 5,
            "total_tokens": 15,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 0},
        },
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    sync_client = httpx.Client(transport=httpx.MockTransport(handler))
    chat = ChatOpenAI(
        api_key="test",
        model="gpt-4o-mini",
        use_responses_api=True,
        http_client=sync_client,
    )

    with PatchSet():
        with AgentRecorder("lc_responses_route", snapshot_dir=str(tmp_path)) as rec:
            result = chat.invoke("hi")
            rec.output = result.content

    snap = read_snapshot("lc_responses_route", str(tmp_path))
    llm_calls = [e for e in snap["trace"] if e["type"] == "llm_call"]
    assert len(llm_calls) >= 1
    assert llm_calls[0]["response"] == "hi from responses route"
    assert llm_calls[0]["tokens"] == 15
    assert llm_calls[0]["raw_response"] is not None
