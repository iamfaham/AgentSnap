from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from agentsnap.config import write_config


def test_write_config_creates_file_if_absent(tmp_path):
    path = tmp_path / "pyproject.toml"
    write_config(path, {"judge_model": "gpt-4o-mini", "semantic_threshold": 0.95})
    content = path.read_text()
    assert "judge_model" in content
    assert "gpt-4o-mini" in content
    assert "semantic_threshold" in content


def test_write_config_updates_existing_section(tmp_path):
    path = tmp_path / "pyproject.toml"
    path.write_text(textwrap.dedent("""\
        [project]
        name = "myproject"

        [tool.agentsnap]
        semantic_threshold = 0.92
        llm_threshold = 0.75
    """))
    write_config(path, {"semantic_threshold": 0.98})
    content = path.read_text()
    assert "0.98" in content
    assert "0.92" not in content   # old value replaced
    assert "llm_threshold" in content  # other key preserved
    assert 'name = "myproject"' in content  # other section preserved


def test_write_config_adds_section_when_missing(tmp_path):
    path = tmp_path / "pyproject.toml"
    path.write_text(textwrap.dedent("""\
        [project]
        name = "myproject"
    """))
    write_config(path, {"judge_model": "gpt-4o-mini"})
    content = path.read_text()
    assert "agentsnap" in content
    assert "judge_model" in content
    assert 'name = "myproject"' in content


def test_write_config_preserves_comments(tmp_path):
    path = tmp_path / "pyproject.toml"
    path.write_text(textwrap.dedent("""\
        [tool.agentsnap]
        # LLM judge settings
        judge_model = "openai/gpt-4o-mini"
    """))
    write_config(path, {"judge_model": "anthropic/claude-haiku"})
    content = path.read_text()
    assert "# LLM judge settings" in content


def test_write_config_sets_string_and_float(tmp_path):
    path = tmp_path / "pyproject.toml"
    write_config(path, {"judge_model": "gpt-4o", "semantic_threshold": 0.95})
    content = path.read_text()
    assert "0.95" in content          # float must not be quoted
    assert '"gpt-4o"' in content      # string must be quoted
