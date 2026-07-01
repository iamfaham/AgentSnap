from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agentsnap.core.diff import DiffReport


class AgentRegressionError(Exception):
    def __init__(
        self,
        test_name: str,
        diff_report: "DiffReport",
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
            import re
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

        for name, diff in (r.argument_diffs or {}).items():
            lines.append(f"[ARGS] {name}:")
            if isinstance(diff, dict):
                for field, (old_val, new_val) in diff.get("changed", {}).items():
                    lines.append(f"  {field}: {old_val!r} → {new_val!r}")
                for field, val in diff.get("added", {}).items():
                    lines.append(f"  + {field}: {val!r}")
                for field, val in diff.get("removed", {}).items():
                    lines.append(f"  - {field}: {val!r}")
            else:
                lines.append(f"  {diff}")
            lines.append("")

        old_output = self._old_snapshot.get("output", "")
        for step, score in (r.semantic_scores or {}).items():
            pct = int(score * 100)
            failed = f"semantic:{step}" in r.failed_checks
            verdict = "FAIL" if failed else "PASS"
            reason = (r.semantic_reasons or {}).get(step, "")
            reason_str = f'  "{reason}"' if reason else ""
            lines.append(f"[SEMANTIC] {step}: {pct}% {verdict}{reason_str}")
            if step == "output" and failed:
                lines.append(f"  was: {old_output!r}")
                lines.append(f"  now: {self._new_output!r}")

        lines.append("")
        lines.append(f"Failed checks: {r.failed_checks}")
        return "\n".join(lines)


class SnapshotNotFoundError(Exception):
    def __init__(self, test_name: str) -> None:
        super().__init__(
            f"Snapshot not found for '{test_name}'. Run 'agentsnap record' first."
        )
        self.test_name = test_name


class AdapterNotWrappedError(Exception):
    """Raised when an unwrapped client is used inside a recording context."""
