from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore[assignment]

_embedding_model = None


# ---------------------------------------------------------------------------
# Embedding-based semantic backend (default, offline)
# ---------------------------------------------------------------------------

def _get_embedding_model(model_name: str = "all-MiniLM-L6-v2"):
    global _embedding_model
    if _embedding_model is None:
        import os
        from pathlib import Path

        # Require explicit setup: either wizard wrote backend="offline" to
        # pyproject.toml, or a judge API key is available in the environment.
        api_key = os.environ.get("AGENTSNAP_JUDGE_API_KEY")
        if not api_key:
            try:
                from agentsnap.config import load
                cfg = load()
                backend_configured = bool(cfg.get("backend") or cfg.get("judge_api_key"))
            except Exception:
                backend_configured = False
        else:
            backend_configured = True

        if not backend_configured:
            raise RuntimeError(
                "No semantic backend configured.\n"
                "Run 'agentsnap init' to set up your comparison backend."
            )

        cache_str = os.getenv("HF_HOME") or os.getenv("HUGGINGFACE_HUB_CACHE")
        cache_root = Path(cache_str) if cache_str else Path.home() / ".cache" / "huggingface" / "hub"
        model_dir = cache_root / "models--sentence-transformers--all-MiniLM-L6-v2"
        if not model_dir.exists():
            raise RuntimeError(
                "Offline embedding model not downloaded.\n"
                "Run 'agentsnap init' and choose option [2] to download it."
            )

        from sentence_transformers import SentenceTransformer
        import transformers.utils.logging as _hf_log
        _was_enabled = _hf_log.is_progress_bar_enabled()
        _hf_log.disable_progress_bar()
        try:
            _embedding_model = SentenceTransformer(model_name)
        finally:
            if _was_enabled:
                _hf_log.enable_progress_bar()
    return _embedding_model


def _cosine_similarity(a: Any, b: Any) -> float:
    a, b = np.array(a, dtype=float), np.array(b, dtype=float)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < 1e-10:
        return 1.0 if np.allclose(a, b) else 0.0
    return float(np.dot(a, b) / denom)


def _embed(texts: list[str], model_name: str = "all-MiniLM-L6-v2") -> list[Any]:
    return _get_embedding_model(model_name).encode(texts)


# ---------------------------------------------------------------------------
# LLM-as-judge semantic backend (optional, requires API key)
# ---------------------------------------------------------------------------

_JUDGE_PROMPT = """\
You are a strict judge comparing two AI agent outputs for functional equivalence.

OLD OUTPUT:
{old}

NEW OUTPUT:
{new}

Respond with ONLY a JSON object on one line:
{{"score": <float 0.0-1.0>, "reason": "<one sentence>"}}

Scoring guide:
  1.0 = identical or trivially rephrased (same facts, same intent)
  0.8 = same core answer, minor stylistic differences
  0.5 = partially equivalent, some facts changed or omitted
  0.2 = significantly different content or conclusion
  0.0 = completely different or contradictory
"""


class LLMJudge:
    """Semantic scorer that uses an LLM to compare outputs instead of embeddings.

    More accurate than cosine similarity for factual content — understands that
    "Paris is in France" and "France contains Paris" are equivalent, and that
    "Python 3.9" vs "Python 3.12" is a meaningful factual change.

    Works with any OpenAI-compatible endpoint (OpenRouter, OpenAI, etc.).

    Usage:
        from agentsnap.core.diff import LLMJudge
        judge = LLMJudge(api_key="sk-or-...", base_url="https://openrouter.ai/api/v1")
        with AgentAsserter("my_test", judge=judge) as a:
            ...
    """

    def __init__(
        self,
        api_key: str,
        model: str = "openai/gpt-4o-mini",
        base_url: str = "https://openrouter.ai/api/v1",
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self._reasons: dict[str, str] = {}

    def score(self, old: str, new: str) -> float:
        """Return a 0.0–1.0 equivalence score and cache the reason."""
        import json
        import openai

        client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url)
        resp = client.chat.completions.create(
            model=self.model,
            max_tokens=100,
            temperature=0.0,
            messages=[{
                "role": "user",
                "content": _JUDGE_PROMPT.format(old=old, new=new),
            }],
        )
        raw = (resp.choices[0].message.content or "").strip()
        try:
            parsed = json.loads(raw)
            score = float(parsed.get("score", 0.0))
            reason = parsed.get("reason", "")
        except (json.JSONDecodeError, ValueError):
            score = 0.0
            reason = f"Judge returned unparseable response: {raw!r}"
        self._reasons[f"{old[:30]}..."] = reason
        return max(0.0, min(1.0, score))

    def last_reasons(self) -> dict[str, str]:
        return dict(self._reasons)

    @classmethod
    def from_env(cls) -> "LLMJudge | None":
        """Return a configured LLMJudge if AGENTSNAP_JUDGE_API_KEY is set, else None.

        Reads model and base_url from env vars or [tool.agentsnap] in pyproject.toml.
        """
        from agentsnap.config import judge_from_env
        return judge_from_env()


# ---------------------------------------------------------------------------
# Structural diff — edit distance on tool sequence
# ---------------------------------------------------------------------------

def _edit_distance(a: list, b: list) -> int:
    """Levenshtein distance between two lists."""
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[j] = prev[j - 1]
            else:
                dp[j] = 1 + min(prev[j], dp[j - 1], prev[j - 1])
    return dp[n]


@dataclass
class DiffReport:
    passed: bool
    structural_diff: str | None = None
    argument_diffs: dict[str, Any] = field(default_factory=dict)
    semantic_scores: dict[str, float] = field(default_factory=dict)
    semantic_reasons: dict[str, str] = field(default_factory=dict)
    failed_checks: list[str] = field(default_factory=list)


def _tool_calls(trace: list[dict]) -> list[dict]:
    return [s for s in trace if s.get("type") == "tool_call"]


def _llm_calls(trace: list[dict]) -> list[dict]:
    return [s for s in trace if s.get("type") == "llm_call"]


def structural_diff(old_trace: list[dict], new_trace: list[dict]) -> str | None:
    """Compare tool call sequences using edit distance.

    Returns None if identical. Returns a descriptive message including the
    edit distance if they differ — so a one-tool rename is distinguishable
    from a complete rewrite of the sequence.
    """
    old_tools = [s["name"] for s in _tool_calls(old_trace)]
    new_tools = [s["name"] for s in _tool_calls(new_trace)]
    if old_tools == new_tools:
        return None
    dist = _edit_distance(old_tools, new_tools)
    return (
        f"Tool sequence changed (edit distance {dist}): "
        f"{old_tools!r} -> {new_tools!r}"
    )


# ---------------------------------------------------------------------------
# Argument diff — deepdiff when available, fallback to plain dict diff
# ---------------------------------------------------------------------------

def _deepdiff_args(old_args: dict, new_args: dict, ignored_fields: set) -> dict | None:
    """Use deepdiff for rich path-based, type-aware argument comparison."""
    try:
        from deepdiff import DeepDiff
        filtered_old = {k: v for k, v in old_args.items() if k not in ignored_fields}
        filtered_new = {k: v for k, v in new_args.items() if k not in ignored_fields}
        dd = DeepDiff(filtered_old, filtered_new, ignore_order=True, verbose_level=2)
        if not dd:
            return None
        return dd.to_dict()
    except ImportError:
        return None


def _plain_diff_args(old_args: dict, new_args: dict, ignored_fields: set) -> dict | None:
    filtered_old = {k: v for k, v in old_args.items() if k not in ignored_fields}
    filtered_new = {k: v for k, v in new_args.items() if k not in ignored_fields}
    if filtered_old == filtered_new:
        return None
    return {
        "old": filtered_old,
        "new": filtered_new,
        "added":   {k: v for k, v in filtered_new.items() if k not in filtered_old},
        "removed": {k: v for k, v in filtered_old.items() if k not in filtered_new},
        "changed": {
            k: (filtered_old[k], filtered_new[k])
            for k in filtered_old
            if k in filtered_new and filtered_old[k] != filtered_new[k]
        },
    }


def argument_diffs(
    old_trace: list[dict],
    new_trace: list[dict],
    ignored_fields: list[str] | None = None,
) -> dict[str, Any]:
    """Diff tool call arguments. Uses deepdiff if installed, else plain dict diff."""
    ignored = set(ignored_fields or [])
    old_tools = _tool_calls(old_trace)
    new_tools = _tool_calls(new_trace)
    diffs: dict[str, Any] = {}
    for i, (old, new) in enumerate(zip(old_tools, new_tools)):
        key = f"{old['name']}[{i}]"
        old_args = old.get("args", {})
        new_args = new.get("args", {})
        diff = _deepdiff_args(old_args, new_args, ignored) or _plain_diff_args(old_args, new_args, ignored)
        if diff:
            diffs[key] = diff
    return diffs


# ---------------------------------------------------------------------------
# Semantic scoring — embedding cosine sim or LLM judge
# ---------------------------------------------------------------------------

def semantic_scores(
    old_trace: list[dict],
    new_trace: list[dict],
    old_output: str,
    new_output: str,
    embed_fn: Callable[[list[str]], list[Any]] | None = None,
    judge: LLMJudge | None = None,
) -> tuple[dict[str, float], dict[str, str]]:
    """Return (scores dict, reasons dict).

    If a judge is provided it is used instead of embedding cosine similarity.
    reasons is populated only when using the judge.
    """
    scores: dict[str, float] = {}
    reasons: dict[str, str] = {}

    if judge is not None:
        old_llm = _llm_calls(old_trace)
        new_llm = _llm_calls(new_trace)
        for i, (old, new) in enumerate(zip(old_llm, new_llm)):
            key = f"llm_call[{i}]"
            old_resp = str(old.get("response", ""))
            new_resp = str(new.get("response", ""))
            scores[key] = judge.score(old_resp, new_resp)
        scores["output"] = judge.score(old_output, new_output)
        reasons = judge.last_reasons()
    else:
        _embed_fn = embed_fn or _embed
        old_llm = _llm_calls(old_trace)
        new_llm = _llm_calls(new_trace)
        for i, (old, new) in enumerate(zip(old_llm, new_llm)):
            old_resp = str(old.get("response", ""))
            new_resp = str(new.get("response", ""))
            embs = _embed_fn([old_resp, new_resp])
            scores[f"llm_call[{i}]"] = _cosine_similarity(embs[0], embs[1])
        embs = _embed_fn([old_output, new_output])
        scores["output"] = _cosine_similarity(embs[0], embs[1])

    return scores, reasons


# ---------------------------------------------------------------------------
# Top-level diff
# ---------------------------------------------------------------------------

def compute_diff(
    old_snapshot: dict,
    new_trace: list[dict],
    new_output: str,
    semantic_threshold: float = 0.92,
    llm_threshold: float = 0.75,
    ignored_fields: list[str] | None = None,
    embed_fn: Callable[[list[str]], list[Any]] | None = None,
    judge: LLMJudge | None = None,
) -> DiffReport:
    """Compare a new trace against a golden snapshot.

    Semantic comparison uses embedding cosine similarity by default.
    Pass judge=LLMJudge(...) to use an LLM for more accurate comparison.

    Two thresholds:
    - semantic_threshold (0.92): final agent output — should be stable.
    - llm_threshold (0.75): intermediate LLM responses — tolerates natural variance.
    """
    old_trace = old_snapshot["trace"]
    old_output = old_snapshot["output"]
    failed: list[str] = []

    struct = structural_diff(old_trace, new_trace)
    if struct:
        failed.append("structural")

    arg_diffs: dict[str, Any] = {}
    if struct is None:
        arg_diffs = argument_diffs(old_trace, new_trace, ignored_fields)
        if arg_diffs:
            failed.append("arguments")

    sem, reasons = semantic_scores(
        old_trace, new_trace, old_output, new_output,
        embed_fn=embed_fn, judge=judge,
    )
    for key, score in sem.items():
        threshold = semantic_threshold if key == "output" else llm_threshold
        if score < threshold:
            failed.append(f"semantic:{key}")

    return DiffReport(
        passed=len(failed) == 0,
        structural_diff=struct,
        argument_diffs=arg_diffs,
        semantic_scores=sem,
        semantic_reasons=reasons,
        failed_checks=failed,
    )
