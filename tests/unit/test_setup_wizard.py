from __future__ import annotations

import textwrap
import unittest.mock as mock
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


from agentsnap.setup_wizard import (
    PROVIDERS,
    WizardResult,
    _write_env_key,
    apply_result,
    check_offline_model,
    test_judge_connection,
)


# ── WizardResult ───────────────────────────────────────────────────────────────

def test_wizard_result_defaults():
    r = WizardResult(backend="offline")
    assert r.judge_model is None
    assert r.judge_base_url is None
    assert r.api_key is None
    assert r.api_key_env_var is None
    assert r.save_key_to_env is False
    assert r.pre_download_model is False


def test_wizard_result_judge():
    r = WizardResult(
        backend="judge",
        judge_model="openai/gpt-4o-mini",
        judge_base_url="https://openrouter.ai/api/v1",
        api_key="sk-or-test",
        api_key_env_var="AGENTSNAP_JUDGE_API_KEY",
        save_key_to_env=True,
    )
    assert r.backend == "judge"
    assert r.judge_model == "openai/gpt-4o-mini"


# ── PROVIDERS preset ──────────────────────────────────────────────────────────

def test_providers_has_required_keys():
    for name, preset in PROVIDERS.items():
        assert "base_url" in preset, f"{name} missing base_url"
        assert "default_model" in preset, f"{name} missing default_model"
        assert "env_var" in preset, f"{name} missing env_var"
        assert "label" in preset, f"{name} missing label"


def test_providers_includes_openrouter_openai_custom():
    assert "openrouter" in PROVIDERS
    assert "openai" in PROVIDERS
    assert "custom" in PROVIDERS


# ── apply_result ──────────────────────────────────────────────────────────────

def test_apply_result_offline_writes_pyproject(tmp_path):
    result = WizardResult(backend="offline")
    apply_result(result, project_dir=tmp_path)
    content = (tmp_path / "pyproject.toml").read_text()
    assert "agentsnap" in content


def test_apply_result_judge_writes_model_and_base_url(tmp_path):
    result = WizardResult(
        backend="judge",
        judge_model="openai/gpt-4o-mini",
        judge_base_url="https://openrouter.ai/api/v1",
        api_key="sk-or-test",
        api_key_env_var="AGENTSNAP_JUDGE_API_KEY",
        save_key_to_env=False,
    )
    apply_result(result, project_dir=tmp_path)
    content = (tmp_path / "pyproject.toml").read_text()
    assert "judge_model" in content
    assert "openai/gpt-4o-mini" in content
    assert "judge_base_url" in content
    # API key must NOT be in pyproject.toml
    assert "sk-or-test" not in content


def test_apply_result_saves_key_to_env(tmp_path):
    result = WizardResult(
        backend="judge",
        judge_model="openai/gpt-4o-mini",
        judge_base_url="https://openrouter.ai/api/v1",
        api_key="sk-or-test",
        api_key_env_var="AGENTSNAP_JUDGE_API_KEY",
        save_key_to_env=True,
    )
    apply_result(result, project_dir=tmp_path)
    env_content = (tmp_path / ".env").read_text()
    assert "AGENTSNAP_JUDGE_API_KEY=sk-or-test" in env_content


def test_apply_result_env_additive(tmp_path):
    """Writing to .env must not erase existing entries."""
    env_file = tmp_path / ".env"
    env_file.write_text("EXISTING_KEY=existing_value\n")
    result = WizardResult(
        backend="judge",
        judge_model="openai/gpt-4o-mini",
        judge_base_url="https://openrouter.ai/api/v1",
        api_key="sk-or-test",
        api_key_env_var="AGENTSNAP_JUDGE_API_KEY",
        save_key_to_env=True,
    )
    apply_result(result, project_dir=tmp_path)
    content = env_file.read_text()
    assert "EXISTING_KEY=existing_value" in content
    assert "AGENTSNAP_JUDGE_API_KEY=sk-or-test" in content


def test_apply_result_env_updates_existing_key(tmp_path):
    """If key already in .env, update its value rather than appending a duplicate."""
    env_file = tmp_path / ".env"
    env_file.write_text("AGENTSNAP_JUDGE_API_KEY=old-key\n")
    result = WizardResult(
        backend="judge",
        judge_model="openai/gpt-4o-mini",
        judge_base_url="https://openrouter.ai/api/v1",
        api_key="new-key",
        api_key_env_var="AGENTSNAP_JUDGE_API_KEY",
        save_key_to_env=True,
    )
    apply_result(result, project_dir=tmp_path)
    content = env_file.read_text()
    assert "AGENTSNAP_JUDGE_API_KEY=new-key" in content
    assert "old-key" not in content
    assert content.count("AGENTSNAP_JUDGE_API_KEY") == 1


def test_write_env_key_appends_when_no_trailing_newline(tmp_path):
    """Appending to .env without trailing newline must not mangle the file."""
    env = tmp_path / ".env"
    env.write_text("EXISTING=val", encoding="utf-8")  # no trailing newline
    _write_env_key(env, "NEW_KEY", "newval")
    lines = env.read_text(encoding="utf-8").splitlines()
    assert lines == ["EXISTING=val", "NEW_KEY=newval"]


# ── test_judge_connection ─────────────────────────────────────────────────────

def test_judge_connection_returns_latency_on_success():
    fake_response = mock.MagicMock()
    fake_response.choices[0].message.content = "1.0"

    with mock.patch("openai.OpenAI") as MockOpenAI:
        instance = MockOpenAI.return_value
        instance.chat.completions.create.return_value = fake_response
        latency = test_judge_connection(
            base_url="https://openrouter.ai/api/v1",
            model="openai/gpt-4o-mini",
            api_key="sk-test",
        )
    assert isinstance(latency, float)
    assert latency >= 0


def test_judge_connection_raises_on_api_error():
    with mock.patch("openai.OpenAI") as MockOpenAI:
        instance = MockOpenAI.return_value
        instance.chat.completions.create.side_effect = Exception("auth failed")
        with pytest.raises(RuntimeError, match="Connection failed"):
            test_judge_connection(
                base_url="https://openrouter.ai/api/v1",
                model="openai/gpt-4o-mini",
                api_key="sk-bad",
            )


# ── check_offline_model ───────────────────────────────────────────────────────

def test_check_offline_model_returns_none_when_not_cached(monkeypatch):
    monkeypatch.setattr(
        "agentsnap.setup_wizard._hf_cache_dir",
        lambda: Path("/nonexistent/cache/path"),
    )
    result = check_offline_model()
    assert result is None


def test_check_offline_model_returns_path_when_cached(monkeypatch, tmp_path):
    model_dir = tmp_path / "models--sentence-transformers--all-MiniLM-L6-v2"
    model_dir.mkdir()
    monkeypatch.setattr(
        "agentsnap.setup_wizard._hf_cache_dir",
        lambda: tmp_path,
    )
    result = check_offline_model()
    assert result is not None
    assert "MiniLM" in result


def test_embed_raises_when_model_not_cached(tmp_path, monkeypatch):
    # Force the HF cache to a temp dir that has no model
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    # Reset the cached model instance
    import agentsnap.core.diff as diff_mod
    original = diff_mod._embedding_model
    diff_mod._embedding_model = None
    try:
        import pytest
        with pytest.raises(RuntimeError, match="agentsnap init"):
            diff_mod._get_embedding_model()
    finally:
        diff_mod._embedding_model = original
