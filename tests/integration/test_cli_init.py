from __future__ import annotations

import unittest.mock as mock
from pathlib import Path

from click.testing import CliRunner

from agentsnap.cli import cli


# ── agentsnap init — offline path ─────────────────────────────────────────────

def test_init_offline_no_predownload(tmp_path):
    """User picks [2] offline, declines pre-download."""
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(
            cli,
            ["init"],
            input="2\nn\n",   # [2] offline, [n] no pre-download
            catch_exceptions=False,
        )
    assert result.exit_code == 0, result.output
    assert "offline" in result.output.lower() or "embedding" in result.output.lower()


def test_init_offline_with_predownload(tmp_path):
    """User picks [2] offline and accepts pre-download; download is mocked."""
    runner = CliRunner()
    with mock.patch("agentsnap.setup_wizard._download_model") as mock_dl:
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(
                cli,
                ["init"],
                input="2\ny\n",
                catch_exceptions=False,
            )
    assert result.exit_code == 0, result.output
    mock_dl.assert_called_once()


def test_init_writes_pyproject_toml(tmp_path):
    """init must create/update pyproject.toml in the working directory."""
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        runner.invoke(cli, ["init"], input="2\nn\n", catch_exceptions=False)
        assert Path("pyproject.toml").exists()


def test_init_menu_shows_coming_soon(tmp_path):
    """The wizard output must mention coming soon for local LLM option."""
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(
            cli,
            ["init"],
            input="2\nn\n",
            catch_exceptions=False,
        )
    assert "coming soon" in result.output.lower()


# ── agentsnap init — judge path ───────────────────────────────────────────────

def test_init_judge_openrouter(tmp_path):
    """User picks [1] judge with OpenRouter; connectivity test is mocked."""
    runner = CliRunner()
    with mock.patch("agentsnap.setup_wizard.test_judge_connection", return_value=0.5):
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(
                cli,
                ["init"],
                # [1] judge, [1] openrouter, accept default model, api key, [y] save to .env
                input="1\n1\n\nsk-or-test\ny\n",
                catch_exceptions=False,
            )
    assert result.exit_code == 0, result.output
    assert "judge" in result.output.lower() or "connection" in result.output.lower()


def test_init_judge_saves_key_to_env_not_pyproject(tmp_path):
    """API key must be written to .env, never to pyproject.toml."""
    runner = CliRunner()
    with mock.patch("agentsnap.setup_wizard.test_judge_connection", return_value=0.3):
        with runner.isolated_filesystem(temp_dir=tmp_path):
            runner.invoke(
                cli,
                ["init"],
                input="1\n1\n\nsk-or-testkey\ny\n",
                catch_exceptions=False,
            )
            assert Path(".env").exists()
            assert "sk-or-testkey" in Path(".env").read_text()
            assert "sk-or-testkey" not in Path("pyproject.toml").read_text()


def test_init_judge_connection_failure_shows_warning(tmp_path):
    """If connectivity test fails, init still completes and shows a warning."""
    runner = CliRunner()
    with mock.patch(
        "agentsnap.setup_wizard.test_judge_connection",
        side_effect=RuntimeError("auth failed"),
    ):
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(
                cli,
                ["init"],
                input="1\n1\n\nsk-bad\ny\n",
                catch_exceptions=False,
            )
    assert result.exit_code == 0, result.output
    assert "warning" in result.output.lower() or "failed" in result.output.lower()


# ── agentsnap check ───────────────────────────────────────────────────────────

def test_check_offline_cached(tmp_path, monkeypatch):
    """check exits 0 and reports model cached when offline model is present."""
    monkeypatch.setattr(
        "agentsnap.setup_wizard.check_offline_model",
        lambda: "/fake/cache/models--sentence-transformers--all-MiniLM-L6-v2",
    )
    monkeypatch.delenv("AGENTSNAP_JUDGE_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(cli, ["check"], catch_exceptions=False)
    assert result.exit_code == 0, result.output
    assert "offline" in result.output.lower() or "embedding" in result.output.lower()


def test_check_offline_not_cached(tmp_path, monkeypatch):
    """check exits 0 but notes model will download on first test run."""
    monkeypatch.setattr(
        "agentsnap.setup_wizard.check_offline_model",
        lambda: None,
    )
    monkeypatch.delenv("AGENTSNAP_JUDGE_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(cli, ["check"], catch_exceptions=False)
    assert result.exit_code == 0
    assert (
        "download" in result.output.lower()
        or "not cached" in result.output.lower()
        or "first" in result.output.lower()
    )


def test_check_judge_connected(tmp_path, monkeypatch):
    """check exits 0 and reports latency when judge connectivity passes."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")

    with mock.patch("agentsnap.setup_wizard.test_judge_connection", return_value=0.4):
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(cli, ["check"], catch_exceptions=False)
    assert result.exit_code == 0, result.output
    assert (
        "ok" in result.output.lower()
        or "connected" in result.output.lower()
        or "0." in result.output
    )


def test_check_judge_unreachable(tmp_path, monkeypatch):
    """check exits 1 when judge API call fails."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-bad")

    with mock.patch(
        "agentsnap.setup_wizard.test_judge_connection",
        side_effect=RuntimeError("auth error"),
    ):
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(cli, ["check"])
    assert result.exit_code == 1
    assert "error" in result.output.lower() or "failed" in result.output.lower()
