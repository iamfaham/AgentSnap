from __future__ import annotations

from agentsnap.exceptions import ReplayError, SnapshotFormatError


class ReplaySession:
    """Cursor over a golden trace used to replay recorded calls.

    Attached to a TraceAccumulator by AgentAsserter(mode="replay"); adapters
    consult it instead of forwarding calls to the real SDK.
    """

    def __init__(self, golden_trace: list[dict], replay_tools: bool = False) -> None:
        self._llm_events = [e for e in golden_trace if e.get("type") == "llm_call"]
        self._tool_events = [e for e in golden_trace if e.get("type") == "tool_call"]
        self._llm_cursor = 0
        self._tool_cursor = 0
        self.replay_tools = replay_tools

    def next_llm_event(self) -> dict:
        if self._llm_cursor >= len(self._llm_events):
            raise ReplayError(
                f"Agent made more LLM calls than the snapshot contains "
                f"({len(self._llm_events)} recorded). Extra call #{self._llm_cursor + 1} "
                "has no recorded response to replay. If this change is intentional, "
                "re-record the golden: pytest --agentsnap-record"
            )
        event = self._llm_events[self._llm_cursor]
        self._llm_cursor += 1
        return event

    def next_tool_event(self, name: str) -> dict:
        if self._tool_cursor >= len(self._tool_events):
            raise ReplayError(
                f"Agent made more tool calls than the snapshot contains "
                f"({len(self._tool_events)} recorded). Extra call to '{name}' "
                "has no recorded result to replay. If this change is intentional, "
                "re-record the golden: pytest --agentsnap-record"
            )
        event = self._tool_events[self._tool_cursor]
        self._tool_cursor += 1
        recorded_name = event.get("name")
        if recorded_name != name:
            raise ReplayError(
                f"Tool call order changed under replay: expected '{recorded_name}' "
                f"at position {self._tool_cursor - 1}, got '{name}'."
            )
        return event

    @property
    def remaining_llm_calls(self) -> int:
        return len(self._llm_events) - self._llm_cursor


def validate_replayable(snapshot: dict, test_name: str) -> None:
    """Raise SnapshotFormatError if the snapshot lacks raw responses for replay."""
    llm_events = [e for e in snapshot.get("trace", []) if e.get("type") == "llm_call"]
    missing = [e.get("step") for e in llm_events if not e.get("raw_response")]
    if missing:
        version = snapshot.get("version", "1.0")
        raise SnapshotFormatError(
            f"Snapshot for '{test_name}' (version {version}) has no raw_response for "
            f"llm_call step(s) {missing} — it was recorded before replay support. "
            "Re-record it: pytest --agentsnap-record (or delete the snapshot file "
            "and run the test once)."
        )
