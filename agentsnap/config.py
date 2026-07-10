"""
agentsnap configuration.

Priority (highest to lowest):
  1. Environment variables
  2. [tool.agentsnap] in the nearest pyproject.toml
  3. Built-in defaults

API key resolution
------------------
agentsnap does NOT require a separate key. It looks for a key in this order:

  1. AGENTSNAP_JUDGE_API_KEY  -- explicit override, always wins
  2. Provider-specific key derived from judge_base_url:
       openrouter.ai   -> OPENROUTER_API_KEY
       api.openai.com  -> OPENAI_API_KEY
       anthropic.com   -> ANTHROPIC_API_KEY
       api.groq.com    -> GROQ_API_KEY
       api.mistral.ai  -> MISTRAL_API_KEY
       api.cohere.com  -> COHERE_API_KEY

So if you already have OPENROUTER_API_KEY set (and judge_base_url points to
OpenRouter, which is the default), the judge works with zero additional config.

Other environment variables
----------------------------
AGENTSNAP_JUDGE_MODEL     -- optional model override
AGENTSNAP_JUDGE_BASE_URL  -- optional base URL override

pyproject.toml (under [tool.agentsnap])
----------------------------------------
[tool.agentsnap]
judge_model        = "openai/gpt-4o-mini"
judge_base_url     = "https://openrouter.ai/api/v1"
semantic_threshold = 0.92
llm_threshold      = 0.75
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

# -- Environment variable names ------------------------------------------------

ENV_API_KEY  = "AGENTSNAP_JUDGE_API_KEY"
ENV_MODEL    = "AGENTSNAP_JUDGE_MODEL"
ENV_BASE_URL = "AGENTSNAP_JUDGE_BASE_URL"

# -- Provider key lookup: substring of base_url -> env var name ---------------

_PROVIDER_KEY_MAP: list[tuple[str, str]] = [
    ("openrouter.ai",  "OPENROUTER_API_KEY"),
    ("api.openai.com", "OPENAI_API_KEY"),
    ("anthropic.com",  "ANTHROPIC_API_KEY"),
    ("api.groq.com",   "GROQ_API_KEY"),
    ("api.mistral.ai", "MISTRAL_API_KEY"),
    ("api.cohere.com", "COHERE_API_KEY"),
]

# -- Built-in defaults ---------------------------------------------------------

DEFAULTS: dict[str, Any] = {
    "judge_model":           "openai/gpt-4o-mini",
    "judge_base_url":        "https://openrouter.ai/api/v1",
    "semantic_threshold":    0.92,
    "llm_threshold":         0.75,
    "structural_tolerance":  0,
    "mode":                  "live",
}


def _find_pyproject(start: Path | None = None) -> Path | None:
    here = (start or Path.cwd()).resolve()
    for candidate in [here, *here.parents]:
        p = candidate / "pyproject.toml"
        if p.exists():
            return p
    return None


def _load_pyproject(path: Path) -> dict[str, Any]:
    try:
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib  # type: ignore[no-redef]
        data = tomllib.loads(path.read_text(encoding="utf-8"))
        return data.get("tool", {}).get("agentsnap", {})
    except Exception:
        return {}


def _resolve_api_key(base_url: str) -> str | None:
    """Find the right API key env var for a given base URL."""
    # Explicit override always wins
    if key := os.getenv(ENV_API_KEY):
        return key
    # Fall back to provider-specific key
    for url_fragment, env_var in _PROVIDER_KEY_MAP:
        if url_fragment in base_url:
            return os.getenv(env_var)
    return None


def load(start: Path | None = None) -> dict[str, Any]:
    """Return merged config: defaults < pyproject.toml < .env < env vars."""
    pyproject = _find_pyproject(start)

    try:
        from dotenv import load_dotenv
        # Load .env from the same directory as pyproject.toml (project root).
        # override=False so real env vars always win over .env values.
        env_file = pyproject.parent / ".env" if pyproject else Path.cwd() / ".env"
        load_dotenv(env_file, override=False)
    except ImportError:
        pass

    cfg = dict(DEFAULTS)

    if pyproject:
        cfg.update(_load_pyproject(pyproject))

    if model := os.getenv(ENV_MODEL):
        cfg["judge_model"] = model
    if base_url := os.getenv(ENV_BASE_URL):
        cfg["judge_base_url"] = base_url

    cfg["judge_api_key"] = _resolve_api_key(cfg["judge_base_url"])
    return cfg


def judge_from_env(start: Path | None = None):
    """Return a configured LLMJudge if an API key can be resolved, else None."""
    from agentsnap.core.diff import LLMJudge

    cfg = load(start)
    if not cfg.get("judge_api_key"):
        return None
    return LLMJudge(
        api_key=cfg["judge_api_key"],
        model=cfg["judge_model"],
        base_url=cfg["judge_base_url"],
    )


def write_config(path: Path, updates: dict[str, Any]) -> None:
    """Merge *updates* into [tool.agentsnap] in the TOML file at *path*.

    Creates the file (and the section) if either is absent.
    Preserves all existing keys, comments, and other sections.
    """
    import tomlkit

    if path.exists():
        doc = tomlkit.parse(path.read_text(encoding="utf-8"))
    else:
        doc = tomlkit.document()

    if "tool" not in doc:
        doc.add("tool", tomlkit.table())

    if "agentsnap" not in doc["tool"]:
        doc["tool"].add("agentsnap", tomlkit.table())

    for key, value in updates.items():
        doc["tool"]["agentsnap"][key] = value

    path.write_text(tomlkit.dumps(doc), encoding="utf-8")
