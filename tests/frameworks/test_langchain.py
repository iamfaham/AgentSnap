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


def _make_chat(text: str) -> ChatOpenAI:
    payload = _canned_payload(text)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    sync_client = httpx.Client(transport=httpx.MockTransport(handler))
    async_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return ChatOpenAI(
        api_key="test",
        model="gpt-4o-mini",
        http_client=sync_client,
        http_async_client=async_client,
    )


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
