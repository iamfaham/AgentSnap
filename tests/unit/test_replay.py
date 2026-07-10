import threading

import pytest

from agentsnap.core.replay import ReplaySession, validate_replayable
from agentsnap.exceptions import ReplayError, SnapshotFormatError

TRACE = [
    {"step": 0, "type": "llm_call", "messages": [{"role": "user", "content": "hi"}],
     "response": "hello", "tokens": 5, "raw_response": {"id": "1"}},
    {"step": 1, "type": "tool_call", "name": "search", "args": {"q": "x"}, "result": "found"},
    {"step": 2, "type": "llm_call", "messages": [{"role": "user", "content": "more"}],
     "response": "sure", "tokens": 5, "raw_response": {"id": "2"}},
]


def test_next_llm_event_in_order():
    s = ReplaySession(TRACE)
    assert s.next_llm_event()["raw_response"] == {"id": "1"}
    assert s.next_llm_event()["raw_response"] == {"id": "2"}


def test_next_llm_event_exhausted_raises():
    s = ReplaySession(TRACE)
    s.next_llm_event(); s.next_llm_event()
    with pytest.raises(ReplayError, match="more LLM calls"):
        s.next_llm_event()


def test_remaining_llm_calls():
    s = ReplaySession(TRACE)
    assert s.remaining_llm_calls == 2
    s.next_llm_event()
    assert s.remaining_llm_calls == 1


def test_next_tool_event_matches_name():
    s = ReplaySession(TRACE)
    assert s.next_tool_event("search")["result"] == "found"


def test_next_tool_event_wrong_name_raises():
    s = ReplaySession(TRACE)
    with pytest.raises(ReplayError, match="expected 'search'"):
        s.next_tool_event("fetch")
    # The mismatched event must not be consumed: retrying with the correct
    # name still returns it.
    assert s.next_tool_event("search")["result"] == "found"


def test_next_tool_event_exhausted_raises():
    s = ReplaySession(TRACE)
    s.next_tool_event("search")
    with pytest.raises(ReplayError, match="more tool calls"):
        s.next_tool_event("search")


def test_validate_replayable_passes_on_v11():
    validate_replayable({"version": "1.1", "trace": TRACE}, "t")


def test_validate_replayable_rejects_missing_raw_response():
    old = {"version": "1.0", "trace": [
        {"step": 0, "type": "llm_call", "messages": [], "response": "x", "tokens": 1},
    ]}
    with pytest.raises(SnapshotFormatError, match="re-record|--agentsnap-record"):
        validate_replayable(old, "t")


def test_validate_replayable_ignores_tool_calls():
    validate_replayable({"version": "1.1", "trace": [TRACE[1]]}, "t")


def test_next_llm_event_thread_safe_under_concurrency():
    trace = [
        {"step": i, "type": "llm_call", "messages": [], "response": str(i),
         "tokens": 1, "raw_response": {"id": str(i)}}
        for i in range(100)
    ]
    s = ReplaySession(trace)

    results: list[dict] = []
    results_lock = threading.Lock()
    errors: list[BaseException] = []

    def worker():
        # Exceptions in non-main threads die silently — capture them so the
        # main thread's `assert not errors` actually fails the test.
        try:
            while True:
                try:
                    event = s.next_llm_event()
                except ReplayError:
                    # Session exhausted: one extra pull must keep raising.
                    with pytest.raises(ReplayError):
                        s.next_llm_event()
                    return
                with results_lock:
                    results.append(event)
        except BaseException as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    assert len(results) == 100
    seen_ids = sorted(int(e["raw_response"]["id"]) for e in results)
    assert seen_ids == list(range(100))
