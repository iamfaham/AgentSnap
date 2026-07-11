from __future__ import annotations

import warnings

from agentsnap.adapters.anthropic import (
    AnthropicRecordingStream,
    AsyncAnthropicRecordingStream,
    dump_raw as _anthropic_dump_raw,
    extract_tool_requests as _anthropic_extract_tool_requests,
    reconstruct_event as _anthropic_reconstruct_event,
    replay_stream as _anthropic_replay_stream,
    replay_stream_async as _anthropic_replay_stream_async,
)
from agentsnap.adapters.openai import (
    AsyncOpenAIRecordingStream,
    OpenAIRecordingStream,
    dump_raw as _openai_dump_raw,
    extract_tool_requests as _openai_extract_tool_requests,
    reconstruct_event as _openai_reconstruct_event,
    replay_stream as _openai_replay_stream,
    replay_stream_async as _openai_replay_stream_async,
)
from agentsnap.core.recorder import TraceAccumulator
from agentsnap.exceptions import ReplayError


# ── Safe apply helper ──────────────────────────────────────────────────────────

def _safe_apply(fn):
    """Call fn(); return [] on any ImportError or AttributeError."""
    try:
        return fn()
    except (ImportError, AttributeError):
        return []


# ── Anthropic ──────────────────────────────────────────────────────────────────

def _apply_anthropic() -> list[tuple]:
    from anthropic.resources.messages.messages import Messages

    original = Messages.create

    def _interceptor(self, *args, **kwargs):
        acc = TraceAccumulator.current()
        if acc is None:
            return original(self, *args, **kwargs)
        messages = kwargs.get("messages", [])

        if acc.replay is not None:
            event = acc.replay.next_llm_event()
            pushed = {
                "type": "llm_call",
                "messages": messages,
                "response": event.get("response", ""),
                "tokens": event.get("tokens", 0),
                "raw_response": event.get("raw_response"),
            }
            if "tool_requests" in event:
                pushed["tool_requests"] = event["tool_requests"]
            acc.push(pushed)
            raw = event.get("raw_response")
            is_stream_recording = isinstance(raw, dict) and raw.get("__stream__")
            wants_stream = bool(kwargs.get("stream"))
            if wants_stream != bool(is_stream_recording):
                raise ReplayError(
                    f"Replay shape mismatch at llm_call step {event.get('step', '?')}: "
                    f"the snapshot recorded a {'streaming' if is_stream_recording else 'non-streaming'} call "
                    f"but the agent requested {'streaming' if wants_stream else 'non-streaming'}. "
                    "Re-record the golden: pytest --agentsnap-record"
                )
            if is_stream_recording:
                return _anthropic_replay_stream(raw["chunks"])
            return _anthropic_reconstruct_event(event)

        if kwargs.get("stream"):
            response = original(self, *args, **kwargs)
            return AnthropicRecordingStream(response, messages, acc)

        response = original(self, *args, **kwargs)
        text = ""
        tokens = 0
        if hasattr(response, "content"):
            for block in response.content:
                if hasattr(block, "text"):
                    text += block.text
        if hasattr(response, "usage"):
            tokens = (
                getattr(response.usage, "input_tokens", 0)
                + getattr(response.usage, "output_tokens", 0)
            )
        acc.push(
            {
                "type": "llm_call",
                "messages": messages,
                "response": text,
                "tokens": tokens,
                "raw_response": _anthropic_dump_raw(response),
                "tool_requests": _anthropic_extract_tool_requests(response),
            }
        )
        return response

    Messages.create = _interceptor
    return [(Messages, "create", original)]


def _apply_anthropic_async() -> list[tuple]:
    from anthropic.resources.messages.messages import AsyncMessages

    original = AsyncMessages.create

    async def _interceptor(self, *args, **kwargs):
        acc = TraceAccumulator.current()
        if acc is None:
            return await original(self, *args, **kwargs)
        messages = kwargs.get("messages", [])

        if acc.replay is not None:
            event = acc.replay.next_llm_event()
            pushed = {
                "type": "llm_call",
                "messages": messages,
                "response": event.get("response", ""),
                "tokens": event.get("tokens", 0),
                "raw_response": event.get("raw_response"),
            }
            if "tool_requests" in event:
                pushed["tool_requests"] = event["tool_requests"]
            acc.push(pushed)
            raw = event.get("raw_response")
            is_stream_recording = isinstance(raw, dict) and raw.get("__stream__")
            wants_stream = bool(kwargs.get("stream"))
            if wants_stream != bool(is_stream_recording):
                raise ReplayError(
                    f"Replay shape mismatch at llm_call step {event.get('step', '?')}: "
                    f"the snapshot recorded a {'streaming' if is_stream_recording else 'non-streaming'} call "
                    f"but the agent requested {'streaming' if wants_stream else 'non-streaming'}. "
                    "Re-record the golden: pytest --agentsnap-record"
                )
            if is_stream_recording:
                return _anthropic_replay_stream_async(raw["chunks"])
            return _anthropic_reconstruct_event(event)

        if kwargs.get("stream"):
            response = await original(self, *args, **kwargs)
            return AsyncAnthropicRecordingStream(response, messages, acc)

        response = await original(self, *args, **kwargs)
        text = ""
        tokens = 0
        if hasattr(response, "content"):
            for block in response.content:
                if hasattr(block, "text"):
                    text += block.text
        if hasattr(response, "usage"):
            tokens = (
                getattr(response.usage, "input_tokens", 0)
                + getattr(response.usage, "output_tokens", 0)
            )
        acc.push(
            {
                "type": "llm_call",
                "messages": messages,
                "response": text,
                "tokens": tokens,
                "raw_response": _anthropic_dump_raw(response),
                "tool_requests": _anthropic_extract_tool_requests(response),
            }
        )
        return response

    AsyncMessages.create = _interceptor
    return [(AsyncMessages, "create", original)]


# ── OpenAI (also covers Groq and OpenRouter — same SDK interface) ──────────────

def _apply_openai() -> list[tuple]:
    from openai.resources.chat.completions.completions import Completions

    original = Completions.create

    def _interceptor(self, *args, **kwargs):
        acc = TraceAccumulator.current()
        if acc is None:
            return original(self, *args, **kwargs)
        messages = kwargs.get("messages", [])

        if acc.replay is not None:
            event = acc.replay.next_llm_event()
            pushed = {
                "type": "llm_call",
                "messages": messages,
                "response": event.get("response", ""),
                "tokens": event.get("tokens", 0),
                "raw_response": event.get("raw_response"),
            }
            if "tool_requests" in event:
                pushed["tool_requests"] = event["tool_requests"]
            acc.push(pushed)
            raw = event.get("raw_response")
            is_stream_recording = isinstance(raw, dict) and raw.get("__stream__")
            wants_stream = bool(kwargs.get("stream"))
            if wants_stream != bool(is_stream_recording):
                raise ReplayError(
                    f"Replay shape mismatch at llm_call step {event.get('step', '?')}: "
                    f"the snapshot recorded a {'streaming' if is_stream_recording else 'non-streaming'} call "
                    f"but the agent requested {'streaming' if wants_stream else 'non-streaming'}. "
                    "Re-record the golden: pytest --agentsnap-record"
                )
            if is_stream_recording:
                return _openai_replay_stream(raw["chunks"])
            return _openai_reconstruct_event(event)

        if kwargs.get("stream"):
            response = original(self, *args, **kwargs)
            return OpenAIRecordingStream(response, messages, acc)

        kwargs["stream"] = False
        response = original(self, *args, **kwargs)
        text = ""
        tokens = 0
        if hasattr(response, "choices") and response.choices:
            text = response.choices[0].message.content or ""
        if hasattr(response, "usage"):
            tokens = getattr(response.usage, "total_tokens", 0)
        acc.push(
            {
                "type": "llm_call",
                "messages": messages,
                "response": text,
                "tokens": tokens,
                "raw_response": _openai_dump_raw(response),
                "tool_requests": _openai_extract_tool_requests(response),
            }
        )
        return response

    Completions.create = _interceptor
    return [(Completions, "create", original)]


def _apply_openai_async() -> list[tuple]:
    from openai.resources.chat.completions.completions import AsyncCompletions

    original = AsyncCompletions.create

    async def _interceptor(self, *args, **kwargs):
        acc = TraceAccumulator.current()
        if acc is None:
            return await original(self, *args, **kwargs)
        messages = kwargs.get("messages", [])

        if acc.replay is not None:
            event = acc.replay.next_llm_event()
            pushed = {
                "type": "llm_call",
                "messages": messages,
                "response": event.get("response", ""),
                "tokens": event.get("tokens", 0),
                "raw_response": event.get("raw_response"),
            }
            if "tool_requests" in event:
                pushed["tool_requests"] = event["tool_requests"]
            acc.push(pushed)
            raw = event.get("raw_response")
            is_stream_recording = isinstance(raw, dict) and raw.get("__stream__")
            wants_stream = bool(kwargs.get("stream"))
            if wants_stream != bool(is_stream_recording):
                raise ReplayError(
                    f"Replay shape mismatch at llm_call step {event.get('step', '?')}: "
                    f"the snapshot recorded a {'streaming' if is_stream_recording else 'non-streaming'} call "
                    f"but the agent requested {'streaming' if wants_stream else 'non-streaming'}. "
                    "Re-record the golden: pytest --agentsnap-record"
                )
            if is_stream_recording:
                return _openai_replay_stream_async(raw["chunks"])
            return _openai_reconstruct_event(event)

        if kwargs.get("stream"):
            response = await original(self, *args, **kwargs)
            return AsyncOpenAIRecordingStream(response, messages, acc)

        kwargs["stream"] = False
        response = await original(self, *args, **kwargs)
        text = ""
        tokens = 0
        if hasattr(response, "choices") and response.choices:
            text = response.choices[0].message.content or ""
        if hasattr(response, "usage"):
            tokens = getattr(response.usage, "total_tokens", 0)
        acc.push(
            {
                "type": "llm_call",
                "messages": messages,
                "response": text,
                "tokens": tokens,
                "raw_response": _openai_dump_raw(response),
                "tool_requests": _openai_extract_tool_requests(response),
            }
        )
        return response

    AsyncCompletions.create = _interceptor
    return [(AsyncCompletions, "create", original)]


# ── Gemini (google-genai >= 1.0) ───────────────────────────────────────────────

def _apply_gemini() -> list[tuple]:
    from google.genai.models import Models  # type: ignore[import]

    original = Models.generate_content

    def _interceptor(self, *, model: str, contents, **kwargs):
        acc = TraceAccumulator.current()
        if acc is None:
            return original(self, model=model, contents=contents, **kwargs)
        if acc.replay is not None:
            raise ReplayError(
                "replay mode does not yet support Gemini; "
                "use mode='live' for this test."
            )
        if isinstance(contents, str):
            messages = [{"role": "user", "content": contents}]
        elif isinstance(contents, list):
            messages = contents
        else:
            messages = [{"role": "user", "content": str(contents)}]
        response = original(self, model=model, contents=contents, **kwargs)
        text = ""
        tokens = 0
        if hasattr(response, "text"):
            text = response.text or ""
        elif hasattr(response, "candidates") and response.candidates:
            for part in response.candidates[0].content.parts:
                if hasattr(part, "text"):
                    text += part.text
        if hasattr(response, "usage_metadata"):
            tokens = getattr(response.usage_metadata, "total_token_count", 0)
        acc.push({"type": "llm_call", "messages": messages, "response": text, "tokens": tokens})
        return response

    Models.generate_content = _interceptor
    return [(Models, "generate_content", original)]


# ── Cohere (cohere >= 5.0) ────────────────────────────────────────────────────

def _apply_cohere() -> list[tuple]:
    import cohere  # type: ignore[import]

    cls = cohere.ClientV2
    original = cls.chat

    def _interceptor(self, *args, **kwargs):
        acc = TraceAccumulator.current()
        if acc is None:
            return original(self, *args, **kwargs)
        if acc.replay is not None:
            raise ReplayError(
                "replay mode does not yet support Cohere; "
                "use mode='live' for this test."
            )
        messages = kwargs.get("messages", [])
        response = original(self, *args, **kwargs)
        text = ""
        tokens = 0
        if hasattr(response, "message") and hasattr(response.message, "content"):
            for block in response.message.content:
                if hasattr(block, "text"):
                    text += block.text
        if hasattr(response, "usage"):
            tokens = getattr(response.usage, "tokens", None)
            if tokens is None:
                inp = getattr(response.usage, "input_tokens", 0) or 0
                out = getattr(response.usage, "output_tokens", 0) or 0
                tokens = inp + out
        acc.push({"type": "llm_call", "messages": messages, "response": text, "tokens": tokens or 0})
        return response

    cls.chat = _interceptor
    return [(cls, "chat", original)]


# ── Mistral (mistralai >= 1.0) ────────────────────────────────────────────────

def _apply_mistral() -> list[tuple]:
    # Try known class paths; return [] if neither exists.
    cls = None
    try:
        from mistralai.resources.chat import Chat as _Chat  # type: ignore[import]
        cls = _Chat
    except (ImportError, AttributeError):
        pass
    if cls is None:
        try:
            from mistralai.sync_client import Chat as _Chat2  # type: ignore[import]
            cls = _Chat2
        except (ImportError, AttributeError):
            return []

    original = cls.complete

    def _interceptor(self, *args, **kwargs):
        acc = TraceAccumulator.current()
        if acc is None:
            return original(self, *args, **kwargs)
        if acc.replay is not None:
            raise ReplayError(
                "replay mode does not yet support Mistral; "
                "use mode='live' for this test."
            )
        messages = kwargs.get("messages", [])
        kwargs["stream"] = False
        response = original(self, *args, **kwargs)
        text = ""
        tokens = 0
        if hasattr(response, "choices") and response.choices:
            text = response.choices[0].message.content or ""
        if hasattr(response, "usage"):
            tokens = getattr(response.usage, "total_tokens", 0) or 0
        acc.push({"type": "llm_call", "messages": messages, "response": text, "tokens": tokens})
        return response

    cls.complete = _interceptor
    return [(cls, "complete", original)]


# ── PatchSet ──────────────────────────────────────────────────────────────────

_PATCHER_FNS = [
    _apply_anthropic,
    _apply_openai,
    _apply_anthropic_async,
    _apply_openai_async,
    _apply_gemini,
    _apply_cohere,
    _apply_mistral,
]


class PatchSet:
    """Context manager that monkey-patches all installed LLM SDKs.

    Intercepts LLM calls at the SDK class level so any client — wrapped or
    unwrapped — is captured by an active TraceAccumulator. Patchers for
    SDKs that are not installed are silently skipped.

    .. warning::
        Do **not** combine ``PatchSet`` with agentsnap adapters (e.g.
        ``AnthropicAdapter``) on the same client. Both the adapter and the
        PatchSet interceptor will fire, silently recording every LLM call
        twice. Use one or the other, not both.

    Usage::

        with PatchSet():
            client = anthropic.Anthropic()   # no AnthropicAdapter needed
            with AgentRecorder("my_test") as rec:
                rec.output = my_agent(client, "query")
    """

    def __init__(self) -> None:
        self._applied: list[tuple] = []

    def __enter__(self) -> PatchSet:
        for fn in _PATCHER_FNS:
            patches = _safe_apply(fn)
            for cls, attr, original in patches:
                if getattr(original, "__name__", None) == "_interceptor":
                    warnings.warn(
                        f"PatchSet is patching {cls.__name__}.{attr} which is already patched "
                        "(by an agentsnap adapter or a nested PatchSet). "
                        "LLM events will be recorded twice. "
                        "Use PatchSet OR adapters, not both.",
                        stacklevel=2,
                    )
            self._applied.extend(patches)
        return self

    def __exit__(self, *args) -> None:
        for cls, attr, original in self._applied:
            setattr(cls, attr, original)
        self._applied.clear()
