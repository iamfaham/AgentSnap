from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from agentsnap.core.normalize import DEFAULT_VOLATILE_FIELDS, normalize_trace

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore[assignment]

_DEFAULT_LLM_THRESHOLD_EMBED = 0.75
_DEFAULT_LLM_THRESHOLD_JUDGE = 0.40

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
    if np is None:
        # Pure-python fallback so custom embed_fn vectors work without numpy
        # (numpy ships with the offline extra, not the base install).
        av, bv = [float(x) for x in a], [float(x) for x in b]
        denom = sum(x * x for x in av) ** 0.5 * sum(y * y for y in bv) ** 0.5
        if denom < 1e-10:
            return 1.0 if av == bv else 0.0
        return sum(x * y for x, y in zip(av, bv)) / denom
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

_STRUCTURAL_JUDGE_PROMPT = """\
You are evaluating whether a change in an AI agent's tool call sequence is a meaningful behavioral regression.

Old tool sequence: {old_tools}
New tool sequence: {new_tools}

Respond with ONLY a JSON object on one line:
{{"score": <float 0.0-1.0>, "reason": "<one sentence>"}}

Scoring guide:
  1.0 = sequences are functionally equivalent (same tools, minor reorder or extra retry)
  0.5 = partially equivalent (some tools added/removed that could affect the result)
  0.0 = fundamentally different path (key tools dropped or replaced)
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
        self._call_count: int = 0
        import openai
        self._client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url)

    def _parse_judge_response(self, raw: str) -> tuple[float, str]:
        import json
        try:
            parsed = json.loads(raw)
            score = float(parsed.get("score", 0.0))
            reason = parsed.get("reason", "")
        except (json.JSONDecodeError, ValueError):
            score = 0.0
            reason = f"Judge returned unparseable response: {raw!r}"
        return max(0.0, min(1.0, score)), reason

    def score(self, old: str, new: str, key: str | None = None) -> float:
        """Return a 0.0-1.0 equivalence score for two text outputs.

        key: if given, reasons are stored under this key (e.g. "llm_call[0]", "output")
             so AgentRegressionError can look them up by step name.
             If None, falls back to a sequential counter key.
        """
        resp = self._client.chat.completions.create(
            model=self.model,
            max_tokens=100,
            temperature=0.0,
            messages=[{"role": "user", "content": _JUDGE_PROMPT.format(old=old, new=new)}],
        )
        raw = (resp.choices[0].message.content or "").strip()
        score, reason = self._parse_judge_response(raw)
        actual_key = key if key is not None else f"comparison[{self._call_count}]"
        self._call_count += 1
        self._reasons[actual_key] = reason
        return score

    def score_structural(self, old_tools: list[str], new_tools: list[str]) -> tuple[float, str]:
        """Score whether a tool sequence change is a meaningful behavioral regression.

        Returns (score, reason). Score 1.0 = equivalent, 0.0 = fundamentally different.
        Does not increment _call_count or populate _reasons (structural is a separate concern).
        """
        resp = self._client.chat.completions.create(
            model=self.model,
            max_tokens=100,
            temperature=0.0,
            messages=[{
                "role": "user",
                "content": _STRUCTURAL_JUDGE_PROMPT.format(
                    old_tools=old_tools, new_tools=new_tools
                ),
            }],
        )
        raw = (resp.choices[0].message.content or "").strip()
        return self._parse_judge_response(raw)

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
class DiffConfig:
    """Consolidates all comparison settings passed to compute_diff."""
    semantic_threshold: float = 0.92
    llm_threshold: float | None = None
    structural_tolerance: int = 0
    structural_threshold: float = 0.8  # judge threshold for structural check only
    ignored_fields: list[str] = field(default_factory=list)
    judge: "LLMJudge | None" = None
    compare_llm_requests: bool = False  # replay mode: diff request messages

    def _resolved_llm_threshold(self) -> float:
        if self.llm_threshold is not None:
            return self.llm_threshold
        return _DEFAULT_LLM_THRESHOLD_JUDGE if self.judge is not None else _DEFAULT_LLM_THRESHOLD_EMBED


@dataclass
class DiffReport:
    passed: bool
    structural_diff: str | None = None
    structural_score: float | None = None
    structural_reason: str | None = None
    argument_diffs: dict[str, Any] = field(default_factory=dict)
    semantic_scores: dict[str, float] = field(default_factory=dict)
    semantic_reasons: dict[str, str] = field(default_factory=dict)
    failed_checks: list[str] = field(default_factory=list)
    model_tools_diff: str | None = None


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


def model_tool_diffs(
    old_trace: list[dict],
    new_trace: list[dict],
    ignored_fields: list[str] | None = None,
) -> tuple[str | None, int, dict[str, Any]]:
    """Compare the tool calls the MODEL requested (llm_call "tool_requests").

    Gate: if any llm_call event on EITHER side lacks the "tool_requests" key
    (old-format snapshot, or a streamed event), the comparison is skipped
    entirely — returns (None, 0, {}) so old snapshots keep passing untouched.

    Returns (message, edit_distance, arg_diffs):
      - message: None if the flattened name sequences match, else a
        descriptive string with the edit distance.
      - edit_distance: 0 when sequences match; the caller decides whether to
        fail based on config.structural_tolerance (mirrors structural_diff).
      - arg_diffs: dict keyed f"model_tool:{name}[{i}]" for each pairwise
        request whose args differ (zipped across both sides).
    """
    old_llm = _llm_calls(old_trace)
    new_llm = _llm_calls(new_trace)
    if any("tool_requests" not in e for e in old_llm) or any("tool_requests" not in e for e in new_llm):
        return None, 0, {}

    old_reqs = [r for e in old_llm for r in e["tool_requests"]]
    new_reqs = [r for e in new_llm for r in e["tool_requests"]]
    old_names = [r["name"] for r in old_reqs]
    new_names = [r["name"] for r in new_reqs]

    ignored = set(ignored_fields or [])
    arg_diffs: dict[str, Any] = {}
    for i, (old_r, new_r) in enumerate(zip(old_reqs, new_reqs)):
        old_wrapped = {"args": old_r.get("args")}
        new_wrapped = {"args": new_r.get("args")}
        diff = _deepdiff_args(old_wrapped, new_wrapped, ignored) or _plain_diff_args(old_wrapped, new_wrapped, ignored)
        if diff:
            arg_diffs[f"model_tool:{old_r['name']}[{i}]"] = diff

    if old_names == new_names:
        return None, 0, arg_diffs

    dist = _edit_distance(old_names, new_names)
    message = (
        f"Model-requested tool sequence changed (edit distance {dist}): "
        f"{old_names!r} -> {new_names!r}"
    )
    return message, dist, arg_diffs


def llm_request_diffs(
    old_trace: list[dict],
    new_trace: list[dict],
    ignored_fields: list[str] | None = None,
) -> dict[str, Any]:
    """Diff the request side of llm_calls (messages sent). Used in replay mode."""
    ignored = set(ignored_fields or [])
    old_llm = _llm_calls(old_trace)
    new_llm = _llm_calls(new_trace)
    diffs: dict[str, Any] = {}
    if len(old_llm) != len(new_llm):
        diffs["llm_call_count"] = (
            f"snapshot has {len(old_llm)} llm_call(s), new run made {len(new_llm)}"
        )
    for i, (old, new) in enumerate(zip(old_llm, new_llm)):
        old_msgs = {"messages": old.get("messages", [])}
        new_msgs = {"messages": new.get("messages", [])}
        diff = _deepdiff_args(old_msgs, new_msgs, ignored) or _plain_diff_args(old_msgs, new_msgs, ignored)
        if diff:
            diffs[f"llm_call[{i}].messages"] = diff
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
            if old_resp == new_resp:
                scores[key] = 1.0
                continue
            scores[key] = judge.score(old_resp, new_resp, key=key)
        if old_output == new_output:
            scores["output"] = 1.0
        else:
            scores["output"] = judge.score(old_output, new_output, key="output")
        reasons = judge.last_reasons()
    else:
        _embed_fn = embed_fn or _embed
        old_llm = _llm_calls(old_trace)
        new_llm = _llm_calls(new_trace)
        for i, (old, new) in enumerate(zip(old_llm, new_llm)):
            old_resp = str(old.get("response", ""))
            new_resp = str(new.get("response", ""))
            if old_resp == new_resp:
                scores[f"llm_call[{i}]"] = 1.0
                continue
            embs = _embed_fn([old_resp, new_resp])
            scores[f"llm_call[{i}]"] = _cosine_similarity(embs[0], embs[1])
        if old_output == new_output:
            scores["output"] = 1.0
        else:
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
    config: DiffConfig | None = None,
    embed_fn: Callable[[list[str]], list[Any]] | None = None,
    normalize: bool = True,
) -> DiffReport:
    """Compare a new trace against a golden snapshot.

    normalize=True strips volatile fields before comparison.
    Pass config=DiffConfig(...) to control thresholds, tolerance, and judge.
    """
    if config is None:
        config = DiffConfig()

    old_trace = old_snapshot["trace"]
    old_output = old_snapshot["output"]
    failed: list[str] = []
    llm_threshold = config._resolved_llm_threshold()

    if normalize:
        old_trace = normalize_trace(old_trace, DEFAULT_VOLATILE_FIELDS)
        new_trace = normalize_trace(new_trace, DEFAULT_VOLATILE_FIELDS)

    old_tools = [s["name"] for s in _tool_calls(old_trace)]
    new_tools = [s["name"] for s in _tool_calls(new_trace)]
    struct_desc = structural_diff(old_trace, new_trace)
    struct_score: float | None = None
    struct_reason: str | None = None

    if struct_desc is not None:
        if config.judge is not None:
            struct_score, struct_reason = config.judge.score_structural(old_tools, new_tools)
            if struct_score < config.structural_threshold:
                failed.append("structural")
        else:
            dist = _edit_distance(old_tools, new_tools)
            if dist > config.structural_tolerance:
                failed.append("structural")

    arg_diffs: dict[str, Any] = {}
    if "structural" not in failed:
        arg_diffs = argument_diffs(old_trace, new_trace, config.ignored_fields)
        if arg_diffs:
            failed.append("arguments")

    model_tools_msg, model_tools_dist, model_tool_arg_diffs = model_tool_diffs(
        old_trace, new_trace, config.ignored_fields
    )
    if model_tools_msg is not None and model_tools_dist > config.structural_tolerance:
        failed.append("model_tools")
    if model_tool_arg_diffs:
        arg_diffs.update(model_tool_arg_diffs)
        failed.append("model_tool_args")

    if config.compare_llm_requests:
        req_diffs = llm_request_diffs(old_trace, new_trace, config.ignored_fields)
        if req_diffs:
            arg_diffs.update(req_diffs)
            failed.append("llm_requests")

    sem, reasons = semantic_scores(
        old_trace, new_trace, old_output, new_output,
        embed_fn=embed_fn, judge=config.judge,
    )
    for step, score in sem.items():
        threshold = config.semantic_threshold if step == "output" else llm_threshold
        if score < threshold:
            failed.append(f"semantic:{step}")

    return DiffReport(
        passed=len(failed) == 0,
        structural_diff=struct_desc,
        structural_score=struct_score,
        structural_reason=struct_reason,
        argument_diffs=arg_diffs,
        semantic_scores=sem,
        semantic_reasons=reasons,
        failed_checks=failed,
        model_tools_diff=model_tools_msg,
    )
