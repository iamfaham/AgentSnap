"""
agentsnap configuration.

Priority (highest to lowest):
  1. Environment variables
  2. [tool.agentsnap] in the nearest pyproject.toml
  3. Built-in defaults

Environment variables
---------------------
AGENTSNAP_JUDGE_API_KEY   -- required to enable LLM judge (keep out of files)
AGENTSNAP_JUDGE_MODEL     -- optional, overrides pyproject.toml
AGENTSNAP_JUDGE_BASE_URL  -- optional, overrides pyproject.toml

pyproject.toml (under [tool.agentsnap])
----------------------------------------
[tool.agentsnap]
judge_model          = "openai/gpt-4o-mini"
judge_base_url       = "https://openrouter.ai/api/v1"
semantic_threshold   = 0.92
llm_threshold        = 0.75
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

# -- Environment variable names ------------------------------------------------

ENV_API_KEY  = "AGENTSNAP_JUDGE_API_KEY"
ENV_MODEL    = "AGENTSNAP_JUDGE_MODEL"
ENV_BASE_URL = "AGENTSNAP_JUDGE_BASE_URL"

# -- Built-in defaults ---------------------------------------------------------

DEFAULTS: dict[str, Any] = {
    "judge_model":        "openai/gpt-4o-mini",
    "judge_base_url":     "https://openrouter.ai/api/v1",
    "semantic_threshold": 0.92,
    "llm_threshold":      0.75,
}


def _find_pyproject(start: Path | None = None) -> Path | None:
    """Walk up from start (or cwd) to find the nearest pyproject.toml."""
    here = (start or Path.cwd()).resolve()
    for candidate in [here, *here.parents]:
        p = candidate / "pyproject.toml"
        if p.exists():
            return p
    return None


def _load_pyproject(path: Path) -> dict[str, Any]:
    """Parse [tool.agentsnap] from a pyproject.toml."""
    try:
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib  # type: ignore[no-redef]
        data = tomllib.loads(path.read_text(encoding="utf-8"))
        return data.get("tool", {}).get("agentsnap", {})
    except Exception:
        return {}


def load(start: Path | None = None) -> dict[str, Any]:
    """Return merged config: defaults < pyproject.toml < env vars."""
    cfg = dict(DEFAULTS)

    pyproject = _find_pyproject(start)
    if pyproject:
        cfg.update(_load_pyproject(pyproject))

    # Env vars always win
    if model := os.getenv(ENV_MODEL):
        cfg["judge_model"] = model
    if base_url := os.getenv(ENV_BASE_URL):
        cfg["judge_base_url"] = base_url

    cfg["judge_api_key"] = os.getenv(ENV_API_KEY)
    return cfg


def judge_from_env(start: Path | None = None):
    """Return a configured LLMJudge if AGENTSNAP_JUDGE_API_KEY is set, else None."""
    from agentsnap.core.diff import LLMJudge

    cfg = load(start)
    if not cfg.get("judge_api_key"):
        return None
    return LLMJudge(
        api_key=cfg["judge_api_key"],
        model=cfg["judge_model"],
        base_url=cfg["judge_base_url"],
    )
