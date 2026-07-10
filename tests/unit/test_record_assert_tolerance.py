from __future__ import annotations

import json
from unittest.mock import patch

import numpy as np

from agentsnap.core.asserter import AgentAsserter
from agentsnap.core.diff import DiffReport
from agentsnap.pytest_plugin import SnapshotFixture

_DIM = 8


def _identical_embed(texts):
    v = np.ones(_DIM, dtype=float)
    v /= np.linalg.norm(v)
    return [v.copy() for _ in texts]


def _write_golden(path, output="ok"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "output": output, "trace": [], "model": "m", "input": None,
        "version": "1.0", "recorded_at": "2026-01-01T00:00:00+00:00",
    }), encoding="utf-8")


def test_asserter_accepts_structural_tolerance_param(tmp_path):
    """structural_tolerance is accepted by AgentAsserter and forwarded to DiffConfig."""
    _write_golden(tmp_path / "tol_test.json")
    captured = {}

    def fake_compute_diff(old_snapshot, new_trace, new_output, config=None, **kw):
        captured["structural_tolerance"] = config.structural_tolerance if config else None
        return DiffReport(passed=True, semantic_scores={"output": 1.0})

    with patch("agentsnap.core.asserter.compute_diff", fake_compute_diff):
        with AgentAsserter("tol_test", snapshot_dir=str(tmp_path),
                           structural_tolerance=2, embed_fn=_identical_embed) as a:
            a.output = "ok"

    assert captured["structural_tolerance"] == 2


def test_asserter_default_structural_tolerance_is_zero(tmp_path):
    """Default structural_tolerance stays 0 — no behavioral change for existing users."""
    _write_golden(tmp_path / "zero_test.json")
    captured = {}

    def fake_compute_diff(old_snapshot, new_trace, new_output, config=None, **kw):
        captured["structural_tolerance"] = config.structural_tolerance if config else -1
        return DiffReport(passed=True, semantic_scores={"output": 1.0})

    with patch("agentsnap.core.asserter.compute_diff", fake_compute_diff):
        with AgentAsserter("zero_test", snapshot_dir=str(tmp_path),
                           embed_fn=_identical_embed) as a:
            a.output = "ok"

    assert captured["structural_tolerance"] == 0


def test_snapshot_fixture_passes_structural_tolerance(tmp_path):
    """SnapshotFixture(structural_tolerance=3) threads through to the asserter."""
    _write_golden(tmp_path / "fix_test.json")
    captured = {}

    def fake_compute_diff(old_snapshot, new_trace, new_output, config=None, **kw):
        captured["structural_tolerance"] = config.structural_tolerance if config else -1
        return DiffReport(passed=True, semantic_scores={"output": 1.0})

    fixture = SnapshotFixture(
        snapshot_dir=str(tmp_path),
        semantic_threshold=0.92,
        llm_threshold=0.75,
        judge=None,
        structural_tolerance=3,
    )

    with patch("agentsnap.core.asserter.compute_diff", fake_compute_diff):
        with fixture.assert_agent("fix_test", embed_fn=_identical_embed) as a:
            a.output = "ok"

    assert captured["structural_tolerance"] == 3


def test_fixture_structural_tolerance_default_is_zero_when_not_set(tmp_path):
    """SnapshotFixture with no structural_tolerance arg defaults to 0."""
    _write_golden(tmp_path / "default_test.json")
    captured = {}

    def fake_compute_diff(old_snapshot, new_trace, new_output, config=None, **kw):
        captured["structural_tolerance"] = config.structural_tolerance if config else -1
        return DiffReport(passed=True, semantic_scores={"output": 1.0})

    fixture = SnapshotFixture(
        snapshot_dir=str(tmp_path),
        semantic_threshold=0.92,
        llm_threshold=0.75,
        judge=None,
        # structural_tolerance not passed — should default to 0
    )

    with patch("agentsnap.core.asserter.compute_diff", fake_compute_diff):
        with fixture.assert_agent("default_test", embed_fn=_identical_embed) as a:
            a.output = "ok"

    assert captured["structural_tolerance"] == 0


def test_per_test_structural_tolerance_overrides_fixture_default(tmp_path):
    """assert_agent(structural_tolerance=5) overrides the fixture-level default of 3."""
    _write_golden(tmp_path / "override_test.json")
    captured = {}

    def fake_compute_diff(old_snapshot, new_trace, new_output, config=None, **kw):
        captured["structural_tolerance"] = config.structural_tolerance if config else -1
        return DiffReport(passed=True, semantic_scores={"output": 1.0})

    fixture = SnapshotFixture(
        snapshot_dir=str(tmp_path),
        semantic_threshold=0.92,
        llm_threshold=0.75,
        judge=None,
        structural_tolerance=3,
    )

    with patch("agentsnap.core.asserter.compute_diff", fake_compute_diff):
        with fixture.assert_agent("override_test", structural_tolerance=5,
                                  embed_fn=_identical_embed) as a:
            a.output = "ok"

    assert captured["structural_tolerance"] == 5
