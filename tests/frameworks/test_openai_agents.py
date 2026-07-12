"""Real-framework verification: the OpenAI Agents SDK drives its genuine code
path (Runner -> AsyncOpenAI -> Responses API) through agentsnap's Responses API
PatchSet interceptor against an offline httpx.MockTransport fake.

No network, no API keys. Skips entirely if openai-agents isn't installed
(``pip install -e ".[frameworks]"``).
"""
from __future__ import annotations

import pytest

pytest.importorskip("agents")

pytestmark = pytest.mark.frameworks

import httpx
import openai
from agents import Agent, Runner, set_default_openai_client, set_tracing_disabled

from agentsnap.core.asserter import AgentAsserter
from agentsnap.core.recorder import AgentRecorder
from agentsnap.core.snapshot import read_snapshot
from agentsnap.patches import PatchSet

set_tracing_disabled(True)


# ── Canned Responses-API payload ────────────────────────────────────────────────

def _canned_payload(text: str = "hi from agents sdk") -> dict:
    return {
        "id": "resp_1",
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
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        ],
        "usage": {
            "input_tokens": 10,
            "output_tokens": 5,
            "total_tokens": 15,
            "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 0},
        },
    }


def _ok_handler(payload):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    return handler


def _raising_handler(request: httpx.Request) -> httpx.Response:
    raise AssertionError("live wire call made during replay — PatchSet should have short-circuited")


def _wire_up_default_client(handler) -> None:
    """Install a custom AsyncOpenAI client (backed by an offline MockTransport)
    as the SDK's default client, and make sure tracing never phones home."""
    transport = httpx.MockTransport(handler)
    client = openai.AsyncOpenAI(
        api_key="test", http_client=httpx.AsyncClient(transport=transport)
    )
    set_default_openai_client(client, use_for_tracing=False)


def _make_agent() -> Agent:
    return Agent(name="tester", instructions="be terse")


# ── Record: real Runner.run_sync drives the real SDK through PatchSet ──────────

def test_openai_agents_records_responses_llm_call(tmp_path):
    _wire_up_default_client(_ok_handler(_canned_payload("hi from agents sdk")))
    agent = _make_agent()

    with PatchSet():
        with AgentRecorder("oa_agents", snapshot_dir=str(tmp_path)) as rec:
            result = Runner.run_sync(agent, "say hi")
            rec.output = result.final_output

    snap = read_snapshot("oa_agents", str(tmp_path))
    llm_calls = [e for e in snap["trace"] if e["type"] == "llm_call"]
    assert len(llm_calls) >= 1
    llm = llm_calls[0]
    assert llm["response"] == "hi from agents sdk"
    assert llm["raw_response"] is not None


def test_openai_agents_replay_never_touches_wire(tmp_path):
    _wire_up_default_client(_ok_handler(_canned_payload("hi from agents sdk")))
    agent = _make_agent()
    with PatchSet():
        with AgentRecorder("oa_agents_replay", snapshot_dir=str(tmp_path)) as rec:
            result = Runner.run_sync(agent, "say hi")
            rec.output = result.final_output

    # Replay: swap the default client for one whose transport raises if touched.
    _wire_up_default_client(_raising_handler)
    replay_agent = _make_agent()
    with PatchSet():
        with AgentAsserter("oa_agents_replay", snapshot_dir=str(tmp_path), mode="replay") as a:
            result = Runner.run_sync(replay_agent, "say hi")
            a.output = result.final_output
