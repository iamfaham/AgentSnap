from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# -- Provider presets (API-based judge) ----------------------------------------

PROVIDERS: dict[str, dict[str, Any]] = {
    "openrouter": {
        "label":         "OpenRouter (recommended - many models, one key)",
        "base_url":      "https://openrouter.ai/api/v1",
        "default_model": "openai/gpt-4o-mini",
        "env_var":       "AGENTSNAP_JUDGE_API_KEY",
    },
    "openai": {
        "label":         "OpenAI",
        "base_url":      "https://api.openai.com/v1",
        "default_model": "gpt-4o-mini",
        "env_var":       "AGENTSNAP_JUDGE_API_KEY",
    },
    "custom": {
        "label":         "Custom (any OpenAI-compatible endpoint)",
        "base_url":      None,
        "default_model": None,
        "env_var":       "AGENTSNAP_JUDGE_API_KEY",
    },
}

_OFFLINE_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_HF_CACHE_SUBDIR    = "models--sentence-transformers--all-MiniLM-L6-v2"


# -- Data transfer object ------------------------------------------------------

@dataclass
class WizardResult:
    backend: str                          # "offline" | "judge"
    judge_model: str | None     = None
    judge_base_url: str | None  = None
    api_key: str | None         = None
    api_key_env_var: str | None = None
    save_key_to_env: bool       = False
    pre_download_model: bool    = False


# -- Core logic (testable, no I/O) --------------------------------------------

def apply_result(result: WizardResult, project_dir: Path) -> None:
    """Write wizard choices to pyproject.toml and optionally .env."""
    from agentsnap.config import write_config

    pyproject = project_dir / "pyproject.toml"
    updates: dict[str, Any] = {}

    updates["backend"] = result.backend  # marks wizard as having been run

    if result.backend == "judge":
        if result.judge_model:
            updates["judge_model"] = result.judge_model
        if result.judge_base_url:
            updates["judge_base_url"] = result.judge_base_url

    write_config(pyproject, updates)

    if result.save_key_to_env and result.api_key and result.api_key_env_var:
        _write_env_key(project_dir / ".env", result.api_key_env_var, result.api_key)


def _write_env_key(env_path: Path, key: str, value: str) -> None:
    """Add or update KEY=value in .env without touching other lines.

    Writes with mode 0o600 (owner-read-only) to protect API keys.
    On Windows the mode bits are ignored by the OS but the call is harmless.
    """
    if env_path.exists():
        lines = env_path.read_text(encoding="utf-8").splitlines(keepends=True)
        new_lines = []
        updated = False
        for line in lines:
            if line.startswith(f"{key}=") or line.startswith(f"{key} ="):
                new_lines.append(f"{key}={value}\n")
                updated = True
            else:
                new_lines.append(line)
        if not updated:
            if new_lines and not new_lines[-1].endswith("\n"):
                new_lines[-1] += "\n"
            new_lines.append(f"{key}={value}\n")
        data = "".join(new_lines)
    else:
        data = f"{key}={value}\n"

    fd = os.open(env_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(data)
    finally:
        try:
            os.chmod(env_path, 0o600)
        except OSError:
            pass


def test_judge_connection(base_url: str, model: str, api_key: str) -> float:
    """Make a minimal test call to the judge API. Returns latency in seconds.

    Raises RuntimeError on any failure (bad key, unreachable, etc.).
    """
    import openai

    client = openai.OpenAI(api_key=api_key, base_url=base_url)
    start = time.monotonic()
    try:
        client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Reply with the number 1."}],
            max_tokens=5,
        )
    except Exception as exc:
        raise RuntimeError(f"Connection failed: {exc}") from exc
    return time.monotonic() - start


# Tell pytest not to collect this function as a test (name starts with "test_").
test_judge_connection.__test__ = False  # type: ignore[attr-defined]


def _hf_cache_dir() -> Path:
    """Return the Hugging Face hub cache directory."""
    custom = os.getenv("HF_HOME") or os.getenv("HUGGINGFACE_HUB_CACHE")
    if custom:
        return Path(custom)
    return Path.home() / ".cache" / "huggingface" / "hub"


def check_offline_model() -> str | None:
    """Return the local cache path for all-MiniLM-L6-v2, or None if not cached."""
    cache = _hf_cache_dir()
    model_dir = cache / _HF_CACHE_SUBDIR
    if model_dir.exists():
        return str(model_dir)
    return None


def _download_model() -> None:
    """Pre-download all-MiniLM-L6-v2 into the Hugging Face cache."""
    from sentence_transformers import SentenceTransformer
    SentenceTransformer(_OFFLINE_MODEL_NAME)


# -- Interactive wizard (uses click I/O) ---------------------------------------

def run_wizard() -> WizardResult:
    """Prompt the user interactively and return a WizardResult.

    All I/O goes through click so tests can inject input via CliRunner.

    Menu:
      [1] LLM judge (API)       - recommended, default
      [2] Offline embeddings    - all-MiniLM-L6-v2, explicit opt-in
      [3] Local LLM judge       - coming soon, displayed but NOT selectable
    """
    import click

    click.echo("\nWelcome to agentsnap setup!\n")
    click.echo("How do you want to compare LLM responses between runs?\n")
    click.echo("  [1] LLM judge (API)     - most accurate, API key required  [recommended]")
    click.echo("  [2] Offline embeddings  - all-MiniLM-L6-v2, no API key, runs anywhere")
    click.echo("  [3] Local LLM judge     - run the judge on your own machine  [coming soon]\n")

    backend_choice = click.prompt(
        "Your choice",
        type=click.Choice(["1", "2"]),   # [3] is shown but not selectable
        default="1",
    )

    if backend_choice == "2":
        # Offline embeddings - explicit opt-in
        pre_download = click.confirm(
            "\nPre-download the embedding model now so tests don't pause on first run?",
            default=True,
        )
        return WizardResult(backend="offline", pre_download_model=pre_download)

    # -- LLM judge path (choice "1") -------------------------------------------
    click.echo("\nProvider:\n")
    provider_keys = list(PROVIDERS.keys())
    for i, key in enumerate(provider_keys, 1):
        click.echo(f"  [{i}] {PROVIDERS[key]['label']}")

    valid_choices = [str(i) for i in range(1, len(provider_keys) + 1)]
    provider_idx = click.prompt(
        "\nYour choice",
        type=click.Choice(valid_choices),
        default="1",
    )
    provider_key = provider_keys[int(provider_idx) - 1]
    preset = PROVIDERS[provider_key]

    base_url      = preset["base_url"]
    default_model = preset["default_model"]
    env_var       = preset["env_var"]

    if provider_key == "custom":
        base_url      = click.prompt("Base URL (e.g. https://api.openai.com/v1)")
        default_model = click.prompt("Model name")

    model = click.prompt("Model", default=default_model)

    existing_key = os.environ.get("AGENTSNAP_JUDGE_API_KEY")
    if existing_key:
        click.echo("  Using existing AGENTSNAP_JUDGE_API_KEY from environment.")
        api_key  = existing_key
        save_key = False
    else:
        api_key  = click.prompt("API key", hide_input=True)
        save_key = click.confirm(
            f"\nSave to .env as {env_var}? (recommended - keeps key out of code)",
            default=True,
        )

    return WizardResult(
        backend="judge",
        judge_model=model,
        judge_base_url=base_url,
        api_key=api_key,
        api_key_env_var=env_var,
        save_key_to_env=save_key,
    )
