"""Real-framework verification: Pydantic AI drives its genuine code path
(OpenAIChatModel -> AsyncOpenAI -> AsyncCompletions.create) through agentsnap's
async PatchSet interceptor against an offline httpx.MockTransport fake.

No network, no API keys. Skips entirely if pydantic-ai isn't installed
(``pip install -e ".[frameworks]"``).
"""
from __future__ import annotations

import pytest

pytest.importorskip("pydantic_ai")

pytestmark = pytest.mark.frameworks

import httpx
import openai
from openai.types.chat import ChatCompletion
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from agentsnap.core.asserter import AgentAsserter
from agentsnap.core.recorder import AgentRecorder
from agentsnap.core.snapshot import read_snapshot
from agentsnap.patches import PatchSet


# ── Canned ChatCompletion payload, built from the real SDK type ────────────────

def _canned_chat_completion_dict(text: str = "hello from pydantic ai") -> dict:
    return {
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


def _canned_payload(text: str = "hello from pydantic ai") -> dict:
    return ChatCompletion.model_validate(_canned_chat_completion_dict(text)).model_dump(
        mode="json"
    )


def _make_agent(handler) -> Agent:
    """Build a pydantic_ai Agent whose OpenAIChatModel is wired to a custom
    AsyncOpenAI client backed by an offline httpx.MockTransport."""
    transport = httpx.MockTransport(handler)
    custom_client = openai.AsyncOpenAI(
        api_key="test", http_client=httpx.AsyncClient(transport=transport)
    )
    model = OpenAIChatModel("gpt-4o-mini", provider=OpenAIProvider(openai_client=custom_client))
    return Agent(model)


def _ok_handler(payload):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    return handler


def _raising_handler(request: httpx.Request) -> httpx.Response:
    raise AssertionError("live wire call made during replay — PatchSet should have short-circuited")


# ── Record: real Agent.run_sync drives the real OpenAI SDK through PatchSet ────

def test_pydantic_ai_records_llm_call(tmp_path):
    payload = _canned_payload("hello from pydantic ai")
    agent = _make_agent(_ok_handler(payload))

    with PatchSet():
        with AgentRecorder("pai", snapshot_dir=str(tmp_path)) as rec:
            result = agent.run_sync("say hi")
            rec.output = result.output

    snap = read_snapshot("pai", str(tmp_path))
    llm_calls = [e for e in snap["trace"] if e["type"] == "llm_call"]
    assert len(llm_calls) >= 1
    llm = llm_calls[0]
    assert llm["response"] == "hello from pydantic ai"
    assert llm["raw_response"] is not None


def test_pydantic_ai_replay_never_touches_wire(tmp_path):
    # Golden recording first.
    payload = _canned_payload("hello from pydantic ai")
    agent = _make_agent(_ok_handler(payload))
    with PatchSet():
        with AgentRecorder("pai_replay", snapshot_dir=str(tmp_path)) as rec:
            result = agent.run_sync("say hi")
            rec.output = result.output

    # Replay: transport handler raises if the wire is ever touched.
    replay_agent = _make_agent(_raising_handler)
    with PatchSet():
        with AgentAsserter("pai_replay", snapshot_dir=str(tmp_path), mode="replay") as a:
            result = replay_agent.run_sync("say hi")
            a.output = result.output
