from __future__ import annotations

import numpy as np
import pytest

from agentsnap.core.diff import (
    DiffReport,
    _cosine_similarity,
    argument_diffs,
    compute_diff,
    semantic_scores,
    structural_diff,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────

_LLM_STEP = {
    "step": 0,
    "type": "llm_call",
    "messages": [{"role": "user", "content": "hello"}],
    "response": "The sky is blue and the grass is green.",
    "tokens": 30,
}
_TOOL_STEP = {
    "step": 1,
    "type": "tool_call",
    "name": "search",
    "args": {"query": "foo"},
    "result": "bar",
}
OLD_TRACE = [_LLM_STEP, _TOOL_STEP]

# Deterministic embedding stub: map each unique string to a fixed vector.
_EMBED_CACHE: dict[str, np.ndarray] = {}
_DIM = 8


def _stub_embed(texts: list[str]) -> list[np.ndarray]:
    result = []
    for t in texts:
        if t not in _EMBED_CACHE:
            rng = np.random.default_rng(abs(hash(t)) % (2**31))
            v = rng.standard_normal(_DIM).astype(float)
            v /= np.linalg.norm(v)
            _EMBED_CACHE[t] = v
        result.append(_EMBED_CACHE[t])
    return result


def _identical_embed(texts: list[str]) -> list[np.ndarray]:
    """Always returns identical unit vectors — cosine sim = 1.0."""
    v = np.ones(_DIM, dtype=float)
    v /= np.linalg.norm(v)
    return [v.copy() for _ in texts]


def _orthogonal_embed(texts: list[str]) -> list[np.ndarray]:
    """Returns orthogonal vectors in pairs — cosine sim = 0.0."""
    vecs = []
    for i, _ in enumerate(texts):
        v = np.zeros(_DIM, dtype=float)
        v[i % _DIM] = 1.0
        vecs.append(v)
    return vecs


# ── Cosine similarity ─────────────────────────────────────────────────────────

def test_cosine_identical():
    v = np.array([1.0, 0.0, 0.0])
    assert _cosine_similarity(v, v) == pytest.approx(1.0)


def test_cosine_orthogonal():
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert _cosine_similarity(a, b) == pytest.approx(0.0)


def test_cosine_zero_vector():
    z = np.zeros(4)
    v = np.array([1.0, 0.0, 0.0, 0.0])
    assert _cosine_similarity(z, v) == 0.0


# ── Structural diff ───────────────────────────────────────────────────────────

def test_structural_same():
    assert structural_diff(OLD_TRACE, OLD_TRACE) is None


def test_structural_no_tools():
    trace = [_LLM_STEP]
    assert structural_diff(trace, trace) is None


def test_structural_different_tool():
    new_trace = [
        _LLM_STEP,
        {**_TOOL_STEP, "name": "fetch"},
    ]
    result = structural_diff(OLD_TRACE, new_trace)
    assert result is not None
    assert "search" in result
    assert "fetch" in result
    assert "edit distance" in result


def test_structural_catches_reordering():
    trace_a = [
        {"step": 0, "type": "tool_call", "name": "fetch", "args": {}, "result": ""},
        {"step": 1, "type": "tool_call", "name": "summarize", "args": {}, "result": ""},
    ]
    trace_b = [
        {"step": 0, "type": "tool_call", "name": "summarize", "args": {}, "result": ""},
        {"step": 1, "type": "tool_call", "name": "fetch", "args": {}, "result": ""},
    ]
    result = structural_diff(trace_a, trace_b)
    assert result is not None


def test_structural_added_tool():
    new_trace = OLD_TRACE + [
        {"step": 2, "type": "tool_call", "name": "extra", "args": {}, "result": ""}
    ]
    assert structural_diff(OLD_TRACE, new_trace) is not None


# ── Argument diffs ────────────────────────────────────────────────────────────

def test_argument_no_change():
    assert argument_diffs(OLD_TRACE, OLD_TRACE) == {}


def test_argument_changed():
    new_trace = [_LLM_STEP, {**_TOOL_STEP, "args": {"query": "bar"}}]
    diffs = argument_diffs(OLD_TRACE, new_trace)
    assert "search[0]" in diffs
    assert diffs["search[0]"]["changed"]["query"] == ("foo", "bar")


def test_argument_added_key():
    new_trace = [_LLM_STEP, {**_TOOL_STEP, "args": {"query": "foo", "limit": 10}}]
    diffs = argument_diffs(OLD_TRACE, new_trace)
    assert "limit" in diffs["search[0]"]["added"]


def test_argument_removed_key():
    old_trace = [_LLM_STEP, {**_TOOL_STEP, "args": {"query": "foo", "limit": 5}}]
    diffs = argument_diffs(old_trace, OLD_TRACE)
    assert "limit" in diffs["search[0]"]["removed"]


def test_argument_ignored_fields():
    old_trace = [_LLM_STEP, {**_TOOL_STEP, "args": {"query": "foo", "ts": "old"}}]
    new_trace = [_LLM_STEP, {**_TOOL_STEP, "args": {"query": "foo", "ts": "new"}}]
    assert argument_diffs(old_trace, new_trace, ignored_fields=["ts"]) == {}


# ── Semantic scores ───────────────────────────────────────────────────────────

def test_semantic_identical(monkeypatch):
    scores, _ = semantic_scores(OLD_TRACE, OLD_TRACE, "same output", "same output", embed_fn=_identical_embed)
    for score in scores.values():
        assert score == pytest.approx(1.0)


def test_semantic_orthogonal(monkeypatch):
    scores, _ = semantic_scores(OLD_TRACE, OLD_TRACE, "output A", "output B", embed_fn=_orthogonal_embed)
    assert scores["output"] == pytest.approx(0.0, abs=0.01)


# ── compute_diff boundary cases ───────────────────────────────────────────────

def _make_snapshot(trace=None, output="hello world"):
    return {
        "version": "1.0",
        "model": "m",
        "input": {},
        "trace": trace or OLD_TRACE,
        "output": output,
        "recorded_at": "2026-01-01T00:00:00+00:00",
    }


def test_compute_diff_passes_at_threshold():
    snapshot = _make_snapshot(output="hello world")
    # identical embed → score = 1.0, threshold = 0.92 → passes
    report = compute_diff(snapshot, OLD_TRACE, "hello world", semantic_threshold=0.92, embed_fn=_identical_embed)
    assert report.passed


def test_compute_diff_fails_below_threshold():
    snapshot = _make_snapshot(output="hello world")
    # orthogonal embed → score = 0.0 < 0.92 → output fails; llm_threshold=0.0 so llm passes
    report = compute_diff(snapshot, OLD_TRACE, "completely different",
                          semantic_threshold=0.92, llm_threshold=0.0, embed_fn=_orthogonal_embed)
    assert not report.passed
    assert any("semantic" in f for f in report.failed_checks)


def test_compute_diff_llm_threshold_separate():
    snapshot = _make_snapshot(output="hello world")
    # orthogonal embed → llm score = 0.0, output score = 0.0
    # llm_threshold=0.0 → llm passes; semantic_threshold=0.92 → output fails
    report = compute_diff(snapshot, OLD_TRACE, "completely different",
                          semantic_threshold=0.92, llm_threshold=0.0, embed_fn=_orthogonal_embed)
    assert "semantic:output" in report.failed_checks
    assert not any(f.startswith("semantic:llm") for f in report.failed_checks)


def test_compute_diff_llm_threshold_catches_drift():
    snapshot = _make_snapshot(output="hello world")
    # identical output but orthogonal llm response → llm_threshold=0.9 catches it
    report = compute_diff(snapshot, OLD_TRACE, "hello world",
                          semantic_threshold=0.0, llm_threshold=0.9, embed_fn=_orthogonal_embed)
    assert not report.passed
    assert any(f.startswith("semantic:llm") for f in report.failed_checks)


def test_compute_diff_passes_above_threshold():
    snapshot = _make_snapshot(output="hello world")
    report = compute_diff(snapshot, OLD_TRACE, "hello world", semantic_threshold=0.50, embed_fn=_identical_embed)
    assert report.passed


def test_compute_diff_structural_failure_skips_arg_check():
    new_trace = [_LLM_STEP, {**_TOOL_STEP, "name": "different_tool"}]
    snapshot = _make_snapshot()
    report = compute_diff(snapshot, new_trace, "out", embed_fn=_identical_embed)
    assert not report.passed
    assert "structural" in report.failed_checks
    assert report.argument_diffs == {}


def test_diff_report_dataclass():
    r = DiffReport(passed=True)
    assert r.structural_diff is None
    assert r.argument_diffs == {}
    assert r.semantic_scores == {}
    assert r.failed_checks == []


# ── AgentRegressionError formatting ──────────────────────────────────────────

from agentsnap.exceptions import AgentRegressionError


def _make_error(
    test_name="my_test",
    struct=None,
    arg_diffs=None,
    scores=None,
    reasons=None,
    failed=None,
    old_output="old answer",
    new_output="new answer",
    old_trace=None,
    new_trace=None,
):
    report = DiffReport(
        passed=False,
        structural_diff=struct,
        argument_diffs=arg_diffs or {},
        semantic_scores=scores or {"output": 0.71},
        semantic_reasons=reasons or {},
        failed_checks=failed or ["semantic:output"],
    )
    old_snapshot = {"trace": old_trace or [], "output": old_output}
    return AgentRegressionError(test_name, report, old_snapshot, new_trace or [], new_output)


def test_error_str_contains_test_name():
    err = _make_error(test_name="billing_test")
    assert "billing_test" in str(err)


def test_error_str_shows_old_and_new_output_on_semantic_failure():
    err = _make_error(old_output="the old answer", new_output="the new answer")
    s = str(err)
    assert "the old answer" in s
    assert "the new answer" in s


def test_error_str_does_not_show_text_when_output_passes():
    report = DiffReport(
        passed=False,
        structural_diff="Tool sequence changed (edit distance 1): ['a'] -> ['b']",
        argument_diffs={},
        semantic_scores={"output": 0.98},
        semantic_reasons={},
        failed_checks=["structural"],
    )
    err = AgentRegressionError(
        "t", report, {"trace": [], "output": "old"}, [], "old"
    )
    s = str(err)
    # output text should not appear — it passed
    assert "old\n  now:" not in s


def test_error_str_shows_arg_changes():
    err = _make_error(
        arg_diffs={"search[0]": {"changed": {"query": ("old q", "new q")}}},
        failed=["arguments"],
        scores={"output": 1.0},
    )
    s = str(err)
    assert "search[0]" in s
    assert "old q" in s
    assert "new q" in s
