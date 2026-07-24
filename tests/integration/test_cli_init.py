from __future__ import annotations

import sys
import types
import unittest.mock as mock
from pathlib import Path

import pytest
from click.testing import CliRunner

from agentsnap.cli import cli
from agentsnap.setup_wizard import WizardResult


def _fake_sentence_transformers(monkeypatch):
    """Make `import sentence_transformers` succeed without the real package."""
    monkeypatch.setitem(
        sys.modules, "sentence_transformers", types.ModuleType("sentence_transformers")
    )


@pytest.fixture(autouse=True)
def _clean_judge_env(monkeypatch):
    monkeypatch.delenv("AGENTSNAP_JUDGE_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)


@pytest.fixture(autouse=True)
def _offline_backend_installed(monkeypatch):
    # The wizard's install-offer prompt fires only when sentence-transformers
    # is NOT importable (true on CI, false on most dev machines), which would
    # shift every scripted stdin feed by one answer. Pin the probe so these
    # tests are environment-independent; tests exercising the prompt itself
    # re-patch it to False.
    monkeypatch.setattr("agentsnap.setup_wizard._offline_installed", lambda: True)


# ── agentsnap init — offline path ─────────────────────────────────────────────

def test_init_offline_no_predownload(tmp_path):
    """User picks [2] offline, declines pre-download."""
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(
            cli,
            ["init"],
            input="2\nn\nn\n",   # [2] offline, [n] no pre-download, [n] no example test
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
                input="2\ny\nn\n",
                catch_exceptions=False,
            )
    assert result.exit_code == 0, result.output
    mock_dl.assert_called_once()


def test_init_offline_installs_backend_when_requested(tmp_path):
    """When the wizard result carries install_offline=True, init runs pip install."""
    canned = WizardResult(backend="offline", install_offline=True, pre_download_model=False)
    runner = CliRunner()
    with mock.patch("agentsnap.setup_wizard.run_wizard", return_value=canned):
        with mock.patch("agentsnap.cli.subprocess.run") as mock_run:
            mock_run.return_value = mock.Mock(returncode=0)
            with runner.isolated_filesystem(temp_dir=tmp_path):
                result = runner.invoke(
                    cli,
                    ["init"],
                    input="n\n",  # [n] no example test (backend prompts skipped via mock)
                    catch_exceptions=False,
                )
    assert result.exit_code == 0, result.output
    mock_run.assert_called_once_with(
        [sys.executable, "-m", "pip", "install", "agentsnap[offline]"], check=False
    )
    assert "installed" in result.output.lower()


def test_init_offline_install_failure_shows_manual_hint(tmp_path):
    """If pip install fails, init prints the manual command but still exits 0."""
    canned = WizardResult(backend="offline", install_offline=True, pre_download_model=False)
    runner = CliRunner()
    with mock.patch("agentsnap.setup_wizard.run_wizard", return_value=canned):
        with mock.patch("agentsnap.cli.subprocess.run") as mock_run:
            mock_run.return_value = mock.Mock(returncode=1)
            with runner.isolated_filesystem(temp_dir=tmp_path):
                result = runner.invoke(
                    cli,
                    ["init"],
                    input="n\n",
                    catch_exceptions=False,
                )
    assert result.exit_code == 0, result.output
    assert "pip install agentsnap[offline]" in result.output


def test_init_writes_pyproject_toml(tmp_path):
    """init must create/update pyproject.toml in the working directory."""
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        runner.invoke(cli, ["init"], input="2\nn\nn\n", catch_exceptions=False)
        assert Path("pyproject.toml").exists()


def test_init_menu_shows_coming_soon(tmp_path):
    """The wizard output must mention coming soon for local LLM option."""
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(
            cli,
            ["init"],
            input="2\nn\nn\n",
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
                # [1] judge, [1] openrouter, api key, [y] save to .env, accept default model, [n] no example test
                input="1\n1\nsk-or-test\ny\n\nn\n",
                catch_exceptions=False,
            )
    assert result.exit_code == 0, result.output
    assert "judge" in result.output.lower() or "ok" in result.output.lower()


def test_init_judge_saves_key_to_env_not_pyproject(tmp_path):
    """API key must be written to .env, never to pyproject.toml."""
    runner = CliRunner()
    with mock.patch("agentsnap.setup_wizard.test_judge_connection", return_value=0.3):
        with runner.isolated_filesystem(temp_dir=tmp_path):
            runner.invoke(
                cli,
                ["init"],
                input="1\n1\nsk-or-testkey\ny\n\nn\n",
                catch_exceptions=False,
            )
            assert Path(".env").exists()
            assert "sk-or-testkey" in Path(".env").read_text()
            assert "sk-or-testkey" not in Path("pyproject.toml").read_text()


def test_init_judge_connection_failure_shows_warning(tmp_path):
    """If connectivity test fails and user declines retry, init still completes."""
    runner = CliRunner()
    with mock.patch(
        "agentsnap.setup_wizard.test_judge_connection",
        side_effect=RuntimeError("auth failed"),
    ):
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(
                cli,
                ["init"],
                # judge, openrouter, key, save, model(default), decline retry, [n] no example test
                input="1\n1\nsk-bad\ny\n\nn\nn\n",
                catch_exceptions=False,
            )
    assert result.exit_code == 0, result.output
    assert "failed" in result.output.lower()


def test_init_judge_uses_existing_env_key(tmp_path):
    """If AGENTSNAP_JUDGE_API_KEY is set, wizard skips key prompt and does not save to .env."""
    runner = CliRunner()
    with mock.patch("agentsnap.setup_wizard.test_judge_connection", return_value=0.3):
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(
                cli,
                ["init"],
                # judge -> openrouter -> no key prompt -> accept default model -> [n] no example test
                input="1\n1\n\nn\n",
                env={"AGENTSNAP_JUDGE_API_KEY": "sk-existing-test"},
                catch_exceptions=False,
            )
    assert result.exit_code == 0, result.output
    assert "AGENTSNAP_JUDGE_API_KEY" in result.output
    assert not (tmp_path / ".env").exists() or "sk-existing-test" not in (tmp_path / ".env").read_text()


# ── agentsnap init — scaffolding (.gitignore + example test) ─────────────────

def test_init_creates_gitignore_with_entry(tmp_path):
    """Fresh dir: init creates .gitignore containing the .last_run ignore line."""
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(cli, ["init"], input="2\nn\nn\n", catch_exceptions=False)
        assert result.exit_code == 0, result.output
        gitignore = Path(".gitignore").read_text(encoding="utf-8")
        assert "__agent_snapshots__/.last_run/" in gitignore.splitlines()
        assert "added to .gitignore" in result.output


def test_init_gitignore_idempotent_on_second_run(tmp_path):
    """Running init twice must not duplicate the .gitignore entry."""
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        runner.invoke(cli, ["init"], input="2\nn\nn\n", catch_exceptions=False)
        result = runner.invoke(cli, ["init"], input="2\nn\nn\n", catch_exceptions=False)
        assert result.exit_code == 0, result.output
        gitignore = Path(".gitignore").read_text(encoding="utf-8")
        assert gitignore.splitlines().count("__agent_snapshots__/.last_run/") == 1
        assert "already in .gitignore" in result.output


def test_init_example_test_created_on_confirm_yes(tmp_path):
    """Confirming the example-test prompt writes the template with the skip marker."""
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(cli, ["init"], input="2\nn\ny\n", catch_exceptions=False)
        assert result.exit_code == 0, result.output
        example_path = Path("tests") / "test_agentsnap_example.py"
        assert example_path.exists()
        content = example_path.read_text(encoding="utf-8")
        assert "pytest.mark.skip" in content
        assert "def my_agent" in content


def test_init_example_test_not_created_on_confirm_no(tmp_path):
    """Declining the example-test prompt leaves tests/ untouched."""
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(cli, ["init"], input="2\nn\nn\n", catch_exceptions=False)
        assert result.exit_code == 0, result.output
        assert not (Path("tests") / "test_agentsnap_example.py").exists()


def test_init_example_test_preexisting_untouched(tmp_path):
    """If tests/test_agentsnap_example.py already exists, init must not overwrite it."""
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        Path("tests").mkdir()
        example_path = Path("tests") / "test_agentsnap_example.py"
        custom_content = "# my custom test, do not touch\n"
        example_path.write_text(custom_content, encoding="utf-8")

        result = runner.invoke(cli, ["init"], input="2\nn\ny\n", catch_exceptions=False)
        assert result.exit_code == 0, result.output
        assert example_path.read_text(encoding="utf-8") == custom_content
        assert "already exists" in result.output.lower()


# ── agentsnap check ───────────────────────────────────────────────────────────

def test_check_exits_1_when_not_configured(tmp_path, monkeypatch):
    """check exits 1 and tells user to run init when no backend is configured."""
    monkeypatch.delenv("AGENTSNAP_JUDGE_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(cli, ["check"])
    assert result.exit_code == 1
    assert "agentsnap init" in result.output


def test_check_offline_cached(tmp_path, monkeypatch):
    """check exits 0 and reports model cached when offline model is present."""
    _fake_sentence_transformers(monkeypatch)
    monkeypatch.setattr(
        "agentsnap.setup_wizard.check_offline_model",
        lambda: "/fake/cache/models--sentence-transformers--all-MiniLM-L6-v2",
    )
    monkeypatch.delenv("AGENTSNAP_JUDGE_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        # Simulate having run agentsnap init with offline backend
        Path("pyproject.toml").write_text('[tool.agentsnap]\nbackend = "offline"\n')
        result = runner.invoke(cli, ["check"], catch_exceptions=False)
    assert result.exit_code == 0, result.output
    assert "offline" in result.output.lower() or "embedding" in result.output.lower()


def test_check_offline_not_cached(tmp_path, monkeypatch):
    """check exits 0 but notes model will download on first test run."""
    _fake_sentence_transformers(monkeypatch)
    monkeypatch.setattr(
        "agentsnap.setup_wizard.check_offline_model",
        lambda: None,
    )
    monkeypatch.delenv("AGENTSNAP_JUDGE_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        # Simulate having run agentsnap init with offline backend
        Path("pyproject.toml").write_text('[tool.agentsnap]\nbackend = "offline"\n')
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
