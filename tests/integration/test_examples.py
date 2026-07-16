from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

EXAMPLES_DIR = Path(__file__).parent.parent.parent / "examples"


def test_demo_mock_exits_zero(tmp_path):
    """demo_mock.py must complete successfully with no API keys."""
    result = subprocess.run(
        [sys.executable, str(EXAMPLES_DIR / "demo_mock.py"), f"--snapshot-dir={tmp_path / 'snaps'}"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, (
        f"demo_mock.py exited {result.returncode}\n"
        f"STDOUT:\n{result.stdout}\n"
        f"STDERR:\n{result.stderr}"
    )
    assert "All demos complete" in result.stdout
    assert "langgraph" in result.stdout.lower()


def test_demo_real_exits_zero_without_keys(tmp_path):
    """demo_real.py must skip all providers gracefully when no API keys are set."""
    # Strip all provider keys so every section hits its skip branch
    stripped_env = {
        k: v for k, v in os.environ.items()
        if not any(k.startswith(p) for p in [
            "OPENROUTER", "ANTHROPIC", "OPENAI", "GEMINI",
            "COHERE", "MISTRAL", "GROQ", "AGENTSNAP",
        ])
    }
    stripped_env["AGENTSNAP_SKIP_DOTENV"] = "1"
    result = subprocess.run(
        [sys.executable, str(EXAMPLES_DIR / "demo_real.py")],
        capture_output=True,
        text=True,
        timeout=30,
        env=stripped_env,
    )
    assert result.returncode == 0, (
        f"demo_real.py exited {result.returncode}\n"
        f"STDOUT:\n{result.stdout}\n"
        f"STDERR:\n{result.stderr}"
    )
    assert "skipped" in result.stdout.lower()


def test_replay_mock_runs_clean():
    """replay.py's mock_demo must record, replay with zero live calls, catch a prompt edit, and stub tools."""
    result = subprocess.run(
        [sys.executable, str(EXAMPLES_DIR / "replay.py")],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, (
        f"replay.py exited {result.returncode}\n"
        f"STDOUT:\n{result.stdout}\n"
        f"STDERR:\n{result.stderr}"
    )
    assert "PASSED deterministically" in result.stdout
    assert "Caught the prompt change" in result.stdout
    assert "the tool function never ran" in result.stdout


def test_streaming_mock_runs_clean():
    """streaming.py's mock_demo must record a streaming call, replay it with zero live calls, and finalize an abandoned stream."""
    result = subprocess.run(
        [sys.executable, str(EXAMPLES_DIR / "streaming.py")],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, (
        f"streaming.py exited {result.returncode}\n"
        f"STDOUT:\n{result.stdout}\n"
        f"STDERR:\n{result.stderr}"
    )
    assert "chunks arrived incrementally" in result.stdout
    assert "PASSED deterministically" in result.stdout
    assert "ZERO live API calls" in result.stdout
    assert "abandoned after 2 chunks" in result.stdout


def test_async_agents_mock_runs_clean():
    """async_agents.py's mock_demo must record an async client, replay with zero live calls, and catch a prompt edit."""
    result = subprocess.run(
        [sys.executable, str(EXAMPLES_DIR / "async_agents.py")],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, (
        f"async_agents.py exited {result.returncode}\n"
        f"STDOUT:\n{result.stdout}\n"
        f"STDERR:\n{result.stderr}"
    )
    assert "PASSED deterministically" in result.stdout
    assert "Caught the prompt change" in result.stdout


def test_model_tools_mock_runs_clean():
    """model_tools.py's mock_demo must record a model tool decision and catch it changing."""
    result = subprocess.run(
        [sys.executable, str(EXAMPLES_DIR / "model_tools.py")],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, (
        f"model_tools.py exited {result.returncode}\n"
        f"STDOUT:\n{result.stdout}\n"
        f"STDERR:\n{result.stderr}"
    )
    assert "[MODEL TOOLS]" in result.stdout
    assert "model_tools" in result.stdout


def test_quickstart_mock_runs_clean():
    """quickstart.py's mock_demo must record, pass, catch a regression, approve, and re-pass."""
    result = subprocess.run(
        [sys.executable, str(EXAMPLES_DIR / "quickstart.py")],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, (
        f"quickstart.py exited {result.returncode}\n"
        f"STDOUT:\n{result.stdout}\n"
        f"STDERR:\n{result.stderr}"
    )
    assert "no snapshot for 'quickstart' - recording golden run" in result.stdout
    assert "'quickstart' PASSED" in result.stdout
    assert "Agent regression in 'quickstart'" in result.stdout
    assert "Approved -- .last_run/quickstart.json promoted to golden." in result.stdout
    assert "Quickstart complete" in result.stdout


def test_scenarios_mock_runs_clean():
    """scenarios.py's mock_demo must write an explicit-scenario and an auto-hash golden, and warn once."""
    result = subprocess.run(
        [sys.executable, str(EXAMPLES_DIR / "scenarios.py")],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, (
        f"scenarios.py exited {result.returncode}\n"
        f"STDOUT:\n{result.stdout}\n"
        f"STDERR:\n{result.stderr}"
    )
    assert "weather__us_west.json" in result.stdout
    assert "weather__" in result.stdout and "sha8 of" in result.stdout
    assert "WARNING: input changed since snapshot was recorded" in result.stdout
    assert "Scenarios complete" in result.stdout


def test_tuning_mock_runs_clean():
    """tuning.py's mock_demo must show a threshold-driven pass/fail and a tolerance-absorbed tool swap."""
    result = subprocess.run(
        [sys.executable, str(EXAMPLES_DIR / "tuning.py")],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, (
        f"tuning.py exited {result.returncode}\n"
        f"STDOUT:\n{result.stdout}\n"
        f"STDERR:\n{result.stderr}"
    )
    assert "'tuning_output' PASSED" in result.stdout
    assert "Caught by the strict threshold" in result.stdout
    assert "model_tools: changed (absorbed by tolerance)" in result.stdout
    assert "Tuning complete" in result.stdout


def test_cli_workflow_mock_runs_clean():
    """cli_workflow.py's mock_demo must drive `agentsnap status`/`update` via subprocess through fail then pass."""
    result = subprocess.run(
        [sys.executable, str(EXAMPLES_DIR / "cli_workflow.py")],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, (
        f"cli_workflow.py exited {result.returncode}\n"
        f"STDOUT:\n{result.stdout}\n"
        f"STDERR:\n{result.stderr}"
    )
    assert "FAIL   semantic:output" in result.stdout
    assert "(exit code: 1)" in result.stdout
    assert "PASS   (live)" in result.stdout
    assert "CLI workflow complete" in result.stdout


def test_pytest_fixture_mock_runs_clean():
    """pytest_fixture.py's mock_demo must record then pass in a subprocess-run pytest, twice."""
    result = subprocess.run(
        [sys.executable, str(EXAMPLES_DIR / "pytest_fixture.py")],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, (
        f"pytest_fixture.py exited {result.returncode}\n"
        f"STDOUT:\n{result.stdout}\n"
        f"STDERR:\n{result.stderr}"
    )
    assert "RECORDED my_agent recorded golden run" in result.stdout
    assert "PASSED   my_agent (live)" in result.stdout
    assert "Pytest fixture complete" in result.stdout


def test_demo_mock_includes_zero_instrumentation(tmp_path):
    """demo_mock.py stdout must mention zero-instrumentation."""
    result = subprocess.run(
        [sys.executable, str(EXAMPLES_DIR / "demo_mock.py"), f"--snapshot-dir={tmp_path / 'snaps'}"],
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    assert "zero" in result.stdout.lower() or "instrument" in result.stdout.lower(), (
        "demo_mock.py should mention zero-instrumentation in its output"
    )
