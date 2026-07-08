from __future__ import annotations

from agentsnap.core.diff import DiffReport
from agentsnap.exceptions import AgentRegressionError


def _make_error(failed_checks, semantic_scores, old_trace, new_trace,
                old_output="old out", new_output="new out"):
    old_snapshot = {"output": old_output, "trace": old_trace}
    report = DiffReport(
        passed=False,
        semantic_scores=semantic_scores,
        failed_checks=failed_checks,
    )
    return AgentRegressionError("my_test", report, old_snapshot, new_trace, new_output)


def test_llm_call_failure_shows_text_excerpts():
    old_trace = [{"type": "llm_call", "step": 0, "messages": [],
                  "response": "The sky is blue", "tokens": 10}]
    new_trace = [{"type": "llm_call", "step": 0, "messages": [],
                  "response": "The ocean is deep", "tokens": 10}]
    err = _make_error(
        failed_checks=["semantic:llm_call[0]"],
        semantic_scores={"llm_call[0]": 0.30, "output": 0.95},
        old_trace=old_trace,
        new_trace=new_trace,
    )
    msg = str(err)
    assert "The sky is blue" in msg
    assert "The ocean is deep" in msg


def test_output_failure_still_shows_text():
    err = _make_error(
        failed_checks=["semantic:output"],
        semantic_scores={"output": 0.30},
        old_trace=[],
        new_trace=[],
        old_output="Paris is in France",
        new_output="Paris is in Germany",
    )
    msg = str(err)
    assert "Paris is in France" in msg
    assert "Paris is in Germany" in msg


def test_passing_step_shows_no_text():
    """Text from passing steps must not appear in the error — only failing ones."""
    old_trace = [{"type": "llm_call", "step": 0, "messages": [],
                  "response": "secret internal step", "tokens": 10}]
    new_trace = [{"type": "llm_call", "step": 0, "messages": [],
                  "response": "secret internal step", "tokens": 10}]
    err = _make_error(
        failed_checks=["semantic:output"],
        semantic_scores={"llm_call[0]": 0.99, "output": 0.30},
        old_trace=old_trace,
        new_trace=new_trace,
    )
    msg = str(err)
    assert "secret internal step" not in msg


def test_long_response_is_truncated():
    long_text = "x" * 300
    old_trace = [{"type": "llm_call", "step": 0, "messages": [],
                  "response": long_text, "tokens": 10}]
    new_trace = [{"type": "llm_call", "step": 0, "messages": [],
                  "response": long_text, "tokens": 10}]
    err = _make_error(
        failed_checks=["semantic:llm_call[0]"],
        semantic_scores={"llm_call[0]": 0.10},
        old_trace=old_trace,
        new_trace=new_trace,
    )
    msg = str(err)
    assert "..." in msg
    assert long_text not in msg


def test_second_llm_call_shows_correct_text():
    """When llm_call[1] fails (not llm_call[0]), the right step's text is shown."""
    old_trace = [
        {"type": "llm_call", "step": 0, "messages": [], "response": "step zero text", "tokens": 10},
        {"type": "llm_call", "step": 1, "messages": [], "response": "step one old", "tokens": 10},
    ]
    new_trace = [
        {"type": "llm_call", "step": 0, "messages": [], "response": "step zero text", "tokens": 10},
        {"type": "llm_call", "step": 1, "messages": [], "response": "step one new", "tokens": 10},
    ]
    err = _make_error(
        failed_checks=["semantic:llm_call[1]"],
        semantic_scores={"llm_call[0]": 0.99, "llm_call[1]": 0.20, "output": 0.95},
        old_trace=old_trace,
        new_trace=new_trace,
    )
    msg = str(err)
    assert "step one old" in msg
    assert "step one new" in msg
    assert "step zero text" not in msg


def test_missing_new_trace_step_shows_missing_placeholder():
    """If new trace has fewer steps, the missing side shows <missing> not blank."""
    old_trace = [
        {"type": "llm_call", "step": 0, "messages": [], "response": "golden response", "tokens": 10},
        {"type": "llm_call", "step": 1, "messages": [], "response": "golden step 2", "tokens": 10},
    ]
    new_trace = [
        {"type": "llm_call", "step": 0, "messages": [], "response": "golden response", "tokens": 10},
        # step 1 is missing — agent stopped early
    ]
    err = _make_error(
        failed_checks=["semantic:llm_call[1]"],
        semantic_scores={"llm_call[0]": 0.99, "llm_call[1]": 0.0, "output": 0.50},
        old_trace=old_trace,
        new_trace=new_trace,
    )
    msg = str(err)
    assert "golden step 2" in msg
    assert "<missing>" in msg


def test_snapshot_format_error_is_exception():
    from agentsnap.exceptions import SnapshotFormatError
    err = SnapshotFormatError("old format")
    assert isinstance(err, Exception)
    assert "old format" in str(err)


def test_replay_error_is_exception():
    from agentsnap.exceptions import ReplayError
    err = ReplayError("extra call")
    assert isinstance(err, Exception)
    assert "extra call" in str(err)
