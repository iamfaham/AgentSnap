from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

_embedding_model = None


def _get_embedding_model(model_name: str = "all-MiniLM-L6-v2"):
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer

        _embedding_model = SentenceTransformer(model_name)
    return _embedding_model


def _cosine_similarity(a: Any, b: Any) -> float:
    a, b = np.array(a, dtype=float), np.array(b, dtype=float)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < 1e-10:
        return 1.0 if np.allclose(a, b) else 0.0
    return float(np.dot(a, b) / denom)


def _embed(texts: list[str], model_name: str = "all-MiniLM-L6-v2") -> list[Any]:
    model = _get_embedding_model(model_name)
    return model.encode(texts)


@dataclass
class DiffReport:
    passed: bool
    structural_diff: str | None = None
    argument_diffs: dict[str, Any] = field(default_factory=dict)
    semantic_scores: dict[str, float] = field(default_factory=dict)
    failed_checks: list[str] = field(default_factory=list)


def _tool_calls(trace: list[dict]) -> list[dict]:
    return [s for s in trace if s.get("type") == "tool_call"]


def _llm_calls(trace: list[dict]) -> list[dict]:
    return [s for s in trace if s.get("type") == "llm_call"]


def structural_diff(old_trace: list[dict], new_trace: list[dict]) -> str | None:
    old_tools = [s["name"] for s in _tool_calls(old_trace)]
    new_tools = [s["name"] for s in _tool_calls(new_trace)]
    if old_tools != new_tools:
        return f"Tool call sequence changed: {old_tools!r} → {new_tools!r}"
    return None


def argument_diffs(
    old_trace: list[dict],
    new_trace: list[dict],
    ignored_fields: list[str] | None = None,
) -> dict[str, Any]:
    ignored = set(ignored_fields or [])
    old_tools = _tool_calls(old_trace)
    new_tools = _tool_calls(new_trace)
    diffs: dict[str, Any] = {}
    for i, (old, new) in enumerate(zip(old_tools, new_tools)):
        key = f"{old['name']}[{i}]"
        old_args = {k: v for k, v in old.get("args", {}).items() if k not in ignored}
        new_args = {k: v for k, v in new.get("args", {}).items() if k not in ignored}
        if old_args == new_args:
            continue
        diffs[key] = {
            "old": old_args,
            "new": new_args,
            "added": {k: v for k, v in new_args.items() if k not in old_args},
            "removed": {k: v for k, v in old_args.items() if k not in new_args},
            "changed": {
                k: (old_args[k], new_args[k])
                for k in old_args
                if k in new_args and old_args[k] != new_args[k]
            },
        }
    return diffs


def semantic_scores(
    old_trace: list[dict],
    new_trace: list[dict],
    old_output: str,
    new_output: str,
    embed_fn: Callable[[list[str]], list[Any]] | None = None,
) -> dict[str, float]:
    _embed_fn = embed_fn or _embed
    scores: dict[str, float] = {}

    old_llm = _llm_calls(old_trace)
    new_llm = _llm_calls(new_trace)
    for i, (old, new) in enumerate(zip(old_llm, new_llm)):
        old_resp = str(old.get("response", ""))
        new_resp = str(new.get("response", ""))
        embs = _embed_fn([old_resp, new_resp])
        scores[f"llm_call[{i}]"] = _cosine_similarity(embs[0], embs[1])

    embs = _embed_fn([old_output, new_output])
    scores["output"] = _cosine_similarity(embs[0], embs[1])
    return scores


def compute_diff(
    old_snapshot: dict,
    new_trace: list[dict],
    new_output: str,
    semantic_threshold: float = 0.92,
    ignored_fields: list[str] | None = None,
    embed_fn: Callable[[list[str]], list[Any]] | None = None,
) -> DiffReport:
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

    sem = semantic_scores(old_trace, new_trace, old_output, new_output, embed_fn=embed_fn)
    for key, score in sem.items():
        if score < semantic_threshold:
            failed.append(f"semantic:{key}")

    return DiffReport(
        passed=len(failed) == 0,
        structural_diff=struct,
        argument_diffs=arg_diffs,
        semantic_scores=sem,
        failed_checks=failed,
    )
