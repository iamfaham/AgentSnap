from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agentsnap.core.diff import DiffReport


class AgentRegressionError(Exception):
    def __init__(self, message: str, diff_report: DiffReport) -> None:
        super().__init__(message)
        self.diff_report = diff_report

    def __str__(self) -> str:
        r = self.diff_report
        lines = [super().__str__(), "", "-- Diff Report ------------------------------------------"]

        if r.structural_diff:
            # Extract tool lists to compute match % alongside the existing message
            import re
            m = re.search(r"\[([^\]]*)\] -> \[([^\]]*)\]", r.structural_diff)
            if m:
                old_tools = [t.strip().strip("'") for t in m.group(1).split(",") if t.strip()]
                new_tools = [t.strip().strip("'") for t in m.group(2).split(",") if t.strip()]
                matched = sum(1 for t in new_tools if t in old_tools)
                total = max(len(old_tools), len(new_tools), 1)
                pct = int(100 * matched / total)
                lines.append(f"  [STRUCTURAL] {pct}% tool match  ({r.structural_diff})")
            else:
                lines.append(f"  [STRUCTURAL] {r.structural_diff}")

        for name, diff in (r.argument_diffs or {}).items():
            added = diff.get("added", {})
            removed = diff.get("removed", {})
            changed = diff.get("changed", {})
            parts = []
            if added:
                parts.append(f"added={added}")
            if removed:
                parts.append(f"removed={removed}")
            if changed:
                parts.append(f"changed={changed}")
            lines.append(f"  [ARGS] {name}: {', '.join(parts)}")

        for step, score in (r.semantic_scores or {}).items():
            pct = int(score * 100)
            verdict = "PASS" if step not in r.failed_checks and f"semantic:{step}" not in r.failed_checks else "FAIL"
            reason = (r.semantic_reasons or {}).get(step, "")
            reason_str = f"  \"{reason}\"" if reason else ""
            lines.append(f"  [SEMANTIC] {step}: {pct}% ({verdict}){reason_str}")

        lines.append(f"  Failed checks: {r.failed_checks}")
        lines.append("---------------------------------------------------------")
        return "\n".join(lines)


class SnapshotNotFoundError(Exception):
    def __init__(self, test_name: str) -> None:
        super().__init__(
            f"Snapshot not found for '{test_name}'. Run 'agentsnap record' first."
        )
        self.test_name = test_name


class AdapterNotWrappedError(Exception):
    """Raised when an unwrapped client is used inside a recording context."""
