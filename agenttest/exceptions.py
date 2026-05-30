from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agenttest.core.diff import DiffReport


class AgentRegressionError(Exception):
    def __init__(self, message: str, diff_report: DiffReport) -> None:
        super().__init__(message)
        self.diff_report = diff_report

    def __str__(self) -> str:
        r = self.diff_report
        lines = [super().__str__(), "", "── Diff Report ──────────────────────────"]
        if r.structural_diff:
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
            lines.append(f"  [SEMANTIC] {step}: {score:.4f}")
        lines.append(f"  Failed checks: {r.failed_checks}")
        lines.append("─────────────────────────────────────────")
        return "\n".join(lines)


class SnapshotNotFoundError(Exception):
    def __init__(self, test_name: str) -> None:
        super().__init__(
            f"Snapshot not found for '{test_name}'. Run 'agenttest record' first."
        )
        self.test_name = test_name


class AdapterNotWrappedError(Exception):
    """Raised when an unwrapped client is used inside a recording context."""
