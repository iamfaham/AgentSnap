from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agentsnap.core.diff import DiffReport


def _excerpt(text: str, max_len: int = 200) -> str:
    return text if len(text) <= max_len else text[:max_len] + "..."


class AgentRegressionError(Exception):
    """Raised by `AgentAsserter` when the new trace drifts beyond threshold.

    `str(exc)` is a human-readable multi-section report (structural, argument,
    model-tools, and semantic diffs). Programmatic callers should inspect
    `exc.diff_report`, a `DiffReport` with `structural_diff`, `argument_diffs`,
    `semantic_scores`, `semantic_reasons`, and `failed_checks`.
    """

    def __init__(
        self,
        test_name: str,
        diff_report: DiffReport,
        old_snapshot: dict,
        new_trace: list,
        new_output: str,
    ) -> None:
        self.test_name = test_name
        self.diff_report = diff_report
        self._old_snapshot = old_snapshot
        self._new_trace = new_trace
        self._new_output = new_output
        super().__init__(self._render())

    def _render(self) -> str:
        r = self.diff_report
        header = f"Agent regression in '{self.test_name}'"
        lines = [header, "=" * len(header), ""]

        if r.structural_diff:
            m = re.search(r"\[([^\]]*)\] -> \[([^\]]*)\]", r.structural_diff)
            if m:
                old_tools = [t.strip().strip("'") for t in m.group(1).split(",") if t.strip()]
                new_tools = [t.strip().strip("'") for t in m.group(2).split(",") if t.strip()]
                matched = sum(1 for t in new_tools if t in old_tools)
                total = max(len(old_tools), len(new_tools), 1)
                pct = int(100 * matched / total)
                lines.append(f"[STRUCTURAL] {pct}% tool match  ({r.structural_diff})")
            else:
                lines.append(f"[STRUCTURAL] {r.structural_diff}")
            lines.append("")

        if r.structural_score is not None:
            pct = int(r.structural_score * 100)
            verdict = "FAIL" if "structural" in r.failed_checks else "PASS"
            lines.append(f"  LLM judge: {pct}% equivalent [{verdict}]")
            if r.structural_reason:
                lines.append(f'  "{r.structural_reason}"')
            lines.append("")

        if r.model_tools_diff:
            lines.append(f"[MODEL TOOLS] {r.model_tools_diff}")
            lines.append("")

        for name, diff in (r.argument_diffs or {}).items():
            lines.append(f"[ARGS] {name}:")
            if isinstance(diff, dict):
                for field, (old_val, new_val) in diff.get("changed", {}).items():
                    lines.append(f"  {field}: {old_val!r} -> {new_val!r}")
                for field, val in diff.get("added", {}).items():
                    lines.append(f"  + {field}: {val!r}")
                for field, val in diff.get("removed", {}).items():
                    lines.append(f"  - {field}: {val!r}")
                for key, value in diff.items():
                    if key not in ("changed", "added", "removed", "old", "new"):
                        if isinstance(value, dict) and value and all(
                            isinstance(v, dict) and "old_value" in v and "new_value" in v
                            for v in value.values()
                        ):
                            lines.append(f"  {key}:")
                            for path, pair in value.items():
                                lines.append(
                                    f"    {path}: {_excerpt(repr(pair['old_value']))} "
                                    f"-> {_excerpt(repr(pair['new_value']))}"
                                )
                        else:
                            lines.append(f"  {key}: {_excerpt(repr(value))}")
            else:
                lines.append(f"  {diff}")
            lines.append("")

        old_output = self._old_snapshot.get("output", "")
        old_llm_calls = [s for s in self._old_snapshot.get("trace", []) if s.get("type") == "llm_call"]
        new_llm_calls = [s for s in self._new_trace if s.get("type") == "llm_call"]

        for step, score in (r.semantic_scores or {}).items():
            pct = int(score * 100)
            failed = f"semantic:{step}" in r.failed_checks
            verdict = "FAIL" if failed else "PASS"
            reason = (r.semantic_reasons or {}).get(step, "")
            reason_str = f'  "{reason}"' if reason else ""
            lines.append(f"[SEMANTIC] {step}: {pct}% {verdict}{reason_str}")
            if failed:
                if step == "output":
                    lines.append(f"  was: {_excerpt(old_output)!r}")
                    lines.append(f"  now: {_excerpt(self._new_output)!r}")
                else:
                    m = re.match(r"llm_call\[(\d+)\]", step)
                    if m:
                        idx = int(m.group(1))
                        old_resp = old_llm_calls[idx].get("response", "") if idx < len(old_llm_calls) else "<missing>"
                        new_resp = new_llm_calls[idx].get("response", "") if idx < len(new_llm_calls) else "<missing>"
                        if old_resp != "<missing>" or new_resp != "<missing>":
                            lines.append(f"  was: {_excerpt(old_resp)!r}")
                            lines.append(f"  now: {_excerpt(new_resp)!r}")

        lines.append("")
        lines.append(f"Failed checks: {r.failed_checks}")
        return "\n".join(lines)


class SnapshotNotFoundError(Exception):
    """Raised when no snapshot file exists for a test (direct SDK use only).

    `AgentAsserter` catches this internally and auto-records a golden run
    instead of propagating it; it only reaches calling code when a snapshot
    is looked up directly (e.g. `read_snapshot()`).
    """

    def __init__(self, test_name: str) -> None:
        super().__init__(
            f"Snapshot not found for '{test_name}'. Run 'agentsnap record' first."
        )
        self.test_name = test_name


class AdapterNotWrappedError(Exception):
    """Raised when an unwrapped client is used inside a recording context."""


class SnapshotFormatError(Exception):
    """Snapshot file cannot be used for the requested operation.

    Raised when replay mode is requested on a snapshot recorded before
    raw responses were captured (version 1.0 files)."""


class ReplayError(Exception):
    """Replay diverged from the recording.

    Raised when the agent makes more LLM calls than the snapshot contains,
    tool call order changes under replay_tools=True, or the provider does
    not support replay yet."""
