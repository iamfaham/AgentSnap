from __future__ import annotations
from agentsnap.core.normalize import normalize_event, normalize_trace, DEFAULT_VOLATILE_FIELDS


def test_strips_tokens():
    event = {"type": "llm_call", "response": "hello", "tokens": 42, "step": 0}
    assert "tokens" not in normalize_event(event, DEFAULT_VOLATILE_FIELDS)


def test_preserves_step():
    event = {"type": "llm_call", "response": "hi", "tokens": 5, "step": 3}
    assert normalize_event(event, DEFAULT_VOLATILE_FIELDS)["step"] == 3


def test_does_not_mutate_original():
    event = {"type": "llm_call", "tokens": 10, "response": "hi"}
    original = set(event.keys())
    normalize_event(event, DEFAULT_VOLATILE_FIELDS)
    assert set(event.keys()) == original


def test_normalize_trace_applies_to_all_events():
    trace = [
        {"type": "llm_call", "response": "a", "tokens": 5, "step": 0},
        {"type": "tool_call", "name": "x", "args": {}, "tokens": 0, "step": 1},
    ]
    result = normalize_trace(trace, DEFAULT_VOLATILE_FIELDS)
    assert all("tokens" not in e for e in result)
    assert len(result) == 2


def test_custom_ignore_fields():
    event = {"type": "tool_call", "name": "search", "request_id": "abc123", "step": 0}
    result = normalize_event(event, {"request_id"})
    assert "request_id" not in result
    assert result["name"] == "search"


def test_empty_ignore_set_is_identity():
    event = {"type": "llm_call", "response": "hi", "tokens": 3}
    assert normalize_event(event, set()) == event
