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


def test_demo_replay_runs_clean():
    """demo_replay.py must record, replay with zero live calls, and catch a prompt edit."""
    result = subprocess.run(
        [sys.executable, str(EXAMPLES_DIR / "demo_replay.py")],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, (
        f"demo_replay.py exited {result.returncode}\n"
        f"STDOUT:\n{result.stdout}\n"
        f"STDERR:\n{result.stderr}"
    )
    assert "PASSED deterministically" in result.stdout
    assert "Caught the prompt change" in result.stdout


def test_demo_streaming_runs_clean():
    """demo_streaming.py must record a streaming call, then replay it with zero live calls."""
    result = subprocess.run(
        [sys.executable, str(EXAMPLES_DIR / "demo_streaming.py")],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, (
        f"demo_streaming.py exited {result.returncode}\n"
        f"STDOUT:\n{result.stdout}\n"
        f"STDERR:\n{result.stderr}"
    )
    assert "chunks arrived incrementally" in result.stdout
    assert "PASSED deterministically" in result.stdout
    assert "ZERO live API calls" in result.stdout


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
