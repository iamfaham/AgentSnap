"""
_common.py -- shared plumbing for agentsnap's examples/*.py scripts.

Not a public agentsnap API. Imported by sibling example scripts only:

    sys.path.insert(0, str(Path(__file__).parent))
    import _common as ex

Provides:
  - make_anthropic_message() / make_openai_chat_completion() -- mock LLM
    responses, schema-validated against the real SDK types (anthropic.types.Message,
    openai.types.chat.ChatCompletion) so anything an adapter reads off a real
    response (block.type, tool_use.input, usage.*) also works on the mock.
  - MockSequence -- a scripted client that returns responses from a list, one
    per call, in order (drives multi-call demos).
  - detect_real_client() -- picks whichever provider key is present in the
    environment and returns a ready-to-use client, or a skip hint if none is.
  - header()/subheader() -- section banners used by every example's story.
  - temp_snapshot_dir(keep) -- scratch snapshot dir, deleted unless --keep.
  - parse_args() -- the --real/--keep argparse boilerplate every example uses.

Mock mode never loads .env (no network, no keys needed). Real mode loads
.env explicitly via detect_real_client() -> agentsnap.config-style resolution
(plain os.environ read; set AGENTSNAP_SKIP_DOTENV=1 to skip .env entirely).
"""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
import zlib
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

SEPARATOR = "=" * 60
THIN = "-" * 60


# ---------------------------------------------------------------------------
# Banners
# ---------------------------------------------------------------------------

def header(title: str) -> None:
    print(f"\n{SEPARATOR}\n  {title}\n{SEPARATOR}")


def subheader(title: str) -> None:
    print(f"\n{THIN}\n  {title}\n{THIN}")


# ---------------------------------------------------------------------------
# Lightweight comparator (no sentence-transformers, no judge, no network)
# ---------------------------------------------------------------------------

def demo_embed(texts: list[str]) -> list[list[float]]:
    """Deterministic offline embedding stub: hashed bag-of-words.

    Every example uses this instead of agentsnap's default sentence-transformers
    backend, so the demos need no extra dependency, no downloaded model, and no
    API key beyond whichever provider key drives --real. Identical texts score
    1.0; texts with different words score low -- enough to demonstrate
    PASS/FAIL. Real projects should run `agentsnap init` to set up a proper
    comparison backend instead of using this.
    """
    vecs = []
    for text in texts:
        v = [0.0] * 256
        for word in text.lower().split():
            v[zlib.crc32(word.encode()) % 256] += 1.0
        vecs.append(v)
    return vecs


# ---------------------------------------------------------------------------
# Mock response builders (schema-validated against the real SDK types)
# ---------------------------------------------------------------------------

def make_anthropic_message(
    text: str,
    *,
    tool_uses: list[tuple[str, dict]] | None = None,
    model: str = "claude-haiku-4-5",
    input_tokens: int = 10,
    output_tokens: int = 20,
):
    """Build an anthropic.types.Message-shaped mock response.

    Validated via ``Message.model_validate`` so it round-trips through
    ``.model_dump()`` (needed for replay) and exposes the exact attributes
    (``.content[i].type``, ``.content[i].text``, tool_use ``.input``/``.name``,
    ``.usage.input_tokens``/``.usage.output_tokens``) the real adapters read.
    """
    from anthropic.types import Message

    content: list[dict[str, Any]] = [{"type": "text", "text": text}]
    for i, (name, tool_input) in enumerate(tool_uses or []):
        content.append(
            {"type": "tool_use", "id": f"toolu_mock{i}", "name": name, "input": tool_input}
        )

    return Message.model_validate(
        {
            "id": "msg_mock",
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": content,
            "stop_reason": "tool_use" if tool_uses else "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
        }
    )


def make_openai_chat_completion(
    text: str | None,
    *,
    tool_calls: list[tuple[str, str]] | None = None,
    model: str = "gpt-4o-mini",
    prompt_tokens: int = 10,
    completion_tokens: int = 20,
):
    """Build an openai.types.chat.ChatCompletion-shaped mock response.

    ``tool_calls`` is a list of ``(name, json_arguments_str)`` pairs. Validated
    via ``ChatCompletion.model_validate`` so it round-trips through
    ``.model_dump()`` for replay.
    """
    from openai.types.chat import ChatCompletion

    message: dict[str, Any] = {"role": "assistant", "content": text}
    if tool_calls:
        message["tool_calls"] = [
            {
                "id": f"call_mock{i}",
                "type": "function",
                "function": {"name": name, "arguments": args},
            }
            for i, (name, args) in enumerate(tool_calls)
        ]

    return ChatCompletion.model_validate(
        {
            "id": "chatcmpl-mock",
            "object": "chat.completion",
            "created": 0,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": "tool_calls" if tool_calls else "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }
    )


# ---------------------------------------------------------------------------
# Scripted multi-call mock client
# ---------------------------------------------------------------------------

class MockSequence:
    """A scripted client that returns responses from a list, one per call, in order.

    Drives multi-call demos (e.g. a two-step fetch-then-summarize agent) without
    needing a real API. Raises ``IndexError`` with a clear message if the agent
    makes more calls than were scripted -- that's a bug in the demo, not
    something to paper over silently.

    ``provider="anthropic"`` exposes ``.messages.create(**kwargs)``.
    ``provider="openai"`` exposes ``.chat.completions.create(**kwargs)``.
    """

    def __init__(self, responses: list[Any], provider: str = "anthropic") -> None:
        if provider not in ("anthropic", "openai"):
            raise ValueError(f"provider must be 'anthropic' or 'openai', got {provider!r}")
        self._responses = list(responses)
        self._index = 0
        self._provider = provider
        if provider == "anthropic":
            self.messages = _SequenceEndpoint(self)
        else:
            self.chat = _ChatNamespace(self)

    def _next(self) -> Any:
        if self._index >= len(self._responses):
            raise IndexError(
                f"MockSequence exhausted: got call #{self._index + 1}, "
                f"only {len(self._responses)} response(s) scripted."
            )
        resp = self._responses[self._index]
        self._index += 1
        return resp


class _SequenceEndpoint:
    def __init__(self, seq: MockSequence) -> None:
        self._seq = seq

    def create(self, **kwargs: Any) -> Any:
        return self._seq._next()


class _ChatNamespace:
    def __init__(self, seq: MockSequence) -> None:
        self.completions = _SequenceEndpoint(seq)


# ---------------------------------------------------------------------------
# Real-client detection
# ---------------------------------------------------------------------------

@dataclass
class DetectedClient:
    """Result of `detect_real_client()`.

    ``client`` is None (and ``hint`` explains why) when no usable key was found.
    """

    provider: str | None
    client: Any
    model: str | None
    hint: str | None = None


def maybe_load_dotenv() -> None:
    """Load .env into os.environ for the --real path only.

    Mock mode never calls this (no network, no keys needed). Real mode calls
    it before `detect_real_client()` so a key committed to a local .env is
    picked up the same way `agentsnap.config.load()` does. Set
    AGENTSNAP_SKIP_DOTENV=1 to opt out.
    """
    if os.getenv("AGENTSNAP_SKIP_DOTENV"):
        return
    try:
        from dotenv import load_dotenv

        load_dotenv(Path(__file__).parent.parent / ".env", override=False)
    except ImportError:
        pass


def detect_real_client() -> DetectedClient:
    """Pick whichever provider key is present and return a ready client.

    Priority: ANTHROPIC_API_KEY -> anthropic client, claude-haiku-4-5
              OPENAI_API_KEY    -> openai client, gpt-4o-mini
              OPENROUTER_API_KEY-> openai client (OpenRouter base_url), openai/gpt-4o-mini

    Does NOT load .env itself -- call `maybe_load_dotenv()` first if you want
    .env-committed keys picked up (every example's `real_demo()` does this).

    Returns a `DetectedClient` with `client=None` and a one-line `hint` string
    naming which env var to set when no key is found. Callers should treat
    that as a skip, not a failure.
    """
    if key := os.getenv("ANTHROPIC_API_KEY"):
        import anthropic

        return DetectedClient("anthropic", anthropic.Anthropic(api_key=key), "claude-haiku-4-5")

    if key := os.getenv("OPENAI_API_KEY"):
        import openai

        return DetectedClient("openai", openai.OpenAI(api_key=key), "gpt-4o-mini")

    if key := os.getenv("OPENROUTER_API_KEY"):
        import openai

        return DetectedClient(
            "openai",
            openai.OpenAI(api_key=key, base_url="https://openrouter.ai/api/v1"),
            "openai/gpt-4o-mini",
        )

    return DetectedClient(
        None,
        None,
        None,
        hint=(
            "no API key found -- set ANTHROPIC_API_KEY, OPENAI_API_KEY, "
            "or OPENROUTER_API_KEY to run --real"
        ),
    )


# Real calls: keep them cheap and deterministic-ish.
REAL_TEMPERATURE = 0
REAL_MAX_TOKENS = 150


# ---------------------------------------------------------------------------
# Temp snapshot dir
# ---------------------------------------------------------------------------

@contextmanager
def temp_snapshot_dir(keep: bool = False) -> Iterator[str]:
    """Scratch snapshot dir for a single example run.

    Deleted on exit unless ``keep`` is True, in which case the path is
    printed so the user can inspect the written snapshots.
    """
    d = tempfile.mkdtemp(prefix="agentsnap_example_")
    try:
        yield d
    finally:
        if keep:
            print(f"\n  (--keep) snapshot dir preserved: {d}")
        else:
            shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------------------
# Argparse boilerplate
# ---------------------------------------------------------------------------

def parse_args(description: str = "") -> argparse.Namespace:
    """--real (also run the real-LLM part) and --keep (keep the temp snapshot dir)."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--real",
        action="store_true",
        help="Also run the real-LLM part (skips with a hint if no key is set)",
    )
    parser.add_argument(
        "--keep",
        action="store_true",
        help="Keep the temp snapshot dir instead of deleting it, and print its path",
    )
    return parser.parse_args()
