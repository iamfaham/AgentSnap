from __future__ import annotations

import warnings

from agentsnap.adapters.anthropic import (
    AnthropicRecordingStream,
    AsyncAnthropicRecordingStream,
)
from agentsnap.adapters.anthropic import (
    RawResponseStreamShim as _AnthropicRawResponseStreamShim,
)
from agentsnap.adapters.anthropic import (
    ReplayLegacyResponse as _AnthropicReplayLegacyResponse,
)
from agentsnap.adapters.anthropic import (
    dump_raw as _anthropic_dump_raw,
)
from agentsnap.adapters.anthropic import (
    extract_tool_requests as _anthropic_extract_tool_requests,
)
from agentsnap.adapters.anthropic import (
    reconstruct_event as _anthropic_reconstruct_event,
)
from agentsnap.adapters.anthropic import (
    replay_stream as _anthropic_replay_stream,
)
from agentsnap.adapters.anthropic import (
    replay_stream_async as _anthropic_replay_stream_async,
)
from agentsnap.adapters.anthropic import (
    unwrap_legacy_response as _anthropic_unwrap_legacy_response,
)
from agentsnap.adapters.anthropic import (
    wants_raw_response as _anthropic_wants_raw_response,
)
from agentsnap.adapters.openai import (
    AsyncOpenAIRecordingStream,
    OpenAIRecordingStream,
)
from agentsnap.adapters.openai import (
    RawResponseStreamShim as _OpenAIRawResponseStreamShim,
)
from agentsnap.adapters.openai import (
    ReplayLegacyResponse as _OpenAIReplayLegacyResponse,
)
from agentsnap.adapters.openai import (
    dump_raw as _openai_dump_raw,
)
from agentsnap.adapters.openai import (
    extract_responses_text as _openai_extract_responses_text,
)
from agentsnap.adapters.openai import (
    extract_responses_tool_requests as _openai_extract_responses_tool_requests,
)
from agentsnap.adapters.openai import (
    extract_tool_requests as _openai_extract_tool_requests,
)
from agentsnap.adapters.openai import (
    normalize_responses_input as _openai_normalize_responses_input,
)
from agentsnap.adapters.openai import (
    reconstruct_event as _openai_reconstruct_event,
)
from agentsnap.adapters.openai import (
    reconstruct_response_event as _openai_reconstruct_response_event,
)
from agentsnap.adapters.openai import (
    replay_stream as _openai_replay_stream,
)
from agentsnap.adapters.openai import (
    replay_stream_async as _openai_replay_stream_async,
)
from agentsnap.adapters.openai import (
    unwrap_legacy_response as _openai_unwrap_legacy_response,
)
from agentsnap.adapters.openai import (
    wants_raw_response as _openai_wants_raw_response,
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
            reconstructed = _anthropic_reconstruct_event(event)
            # Callers that used with_raw_response (e.g. langchain-anthropic)
            # expect a legacy-response-like object with .parse() back —
            # there's no real LegacyAPIResponse in replay since no HTTP call
            # was made.
            if _anthropic_wants_raw_response(kwargs):
                return _AnthropicReplayLegacyResponse(reconstructed)
            return reconstructed

        if kwargs.get("stream"):
            response = original(self, *args, **kwargs)
            # Some callers (e.g. langchain-anthropic) request the raw HTTP
            # response and call .parse() themselves; unwrap so the tee
            # iterates the real Stream object rather than the legacy
            # wrapper. Identity for normal (already-parsed) streams.
            stream = _anthropic_unwrap_legacy_response(response)
            tee = AnthropicRecordingStream(stream, messages, acc)
            # A with_raw_response caller expects to call .parse() on the
            # return value to get the stream; give it back a shim that
            # satisfies that surface while still handing out the (recording)
            # tee.
            if _anthropic_wants_raw_response(kwargs):
                return _AnthropicRawResponseStreamShim(tee, response)
            return tee

        response = original(self, *args, **kwargs)
        parsed = _anthropic_unwrap_legacy_response(response)
        text = ""
        tokens = 0
        if hasattr(parsed, "content"):
            for block in parsed.content:
                if hasattr(block, "text"):
                    text += block.text
        if hasattr(parsed, "usage"):
            tokens = (
                getattr(parsed.usage, "input_tokens", 0)
                + getattr(parsed.usage, "output_tokens", 0)
            )
        acc.push(
            {
                "type": "llm_call",
                "messages": messages,
                "response": text,
                "tokens": tokens,
                "raw_response": _anthropic_dump_raw(parsed),
                "tool_requests": _anthropic_extract_tool_requests(parsed),
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
            reconstructed = _anthropic_reconstruct_event(event)
            # Callers that used with_raw_response (e.g. langchain-anthropic)
            # expect a legacy-response-like object with .parse() back —
            # there's no real LegacyAPIResponse in replay since no HTTP call
            # was made.
            if _anthropic_wants_raw_response(kwargs):
                return _AnthropicReplayLegacyResponse(reconstructed)
            return reconstructed

        if kwargs.get("stream"):
            response = await original(self, *args, **kwargs)
            # Unwrap a legacy raw-response wrapper (e.g. from
            # langchain-anthropic) so the tee iterates the real async Stream
            # object; identity-safe for already-parsed streams.
            stream = _anthropic_unwrap_legacy_response(response)
            tee = AsyncAnthropicRecordingStream(stream, messages, acc)
            # A with_raw_response caller expects to call .parse() on the
            # return value to get the stream; give it back a shim that
            # satisfies that surface while still handing out the (recording)
            # tee.
            if _anthropic_wants_raw_response(kwargs):
                return _AnthropicRawResponseStreamShim(tee, response)
            return tee

        response = await original(self, *args, **kwargs)
        parsed = _anthropic_unwrap_legacy_response(response)
        text = ""
        tokens = 0
        if hasattr(parsed, "content"):
            for block in parsed.content:
                if hasattr(block, "text"):
                    text += block.text
        if hasattr(parsed, "usage"):
            tokens = (
                getattr(parsed.usage, "input_tokens", 0)
                + getattr(parsed.usage, "output_tokens", 0)
            )
        acc.push(
            {
                "type": "llm_call",
                "messages": messages,
                "response": text,
                "tokens": tokens,
                "raw_response": _anthropic_dump_raw(parsed),
                "tool_requests": _anthropic_extract_tool_requests(parsed),
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
            reconstructed = _openai_reconstruct_event(event)
            # Callers that used with_raw_response (e.g. langchain-openai) expect
            # a legacy-response-like object with .parse() back — there's no real
            # LegacyAPIResponse in replay since no HTTP call was made.
            if _openai_wants_raw_response(kwargs):
                return _OpenAIReplayLegacyResponse(reconstructed)
            return reconstructed

        if kwargs.get("stream"):
            response = original(self, *args, **kwargs)
            # Some callers (e.g. langchain-openai) request the raw HTTP response
            # and call .parse() themselves; unwrap so the tee iterates the real
            # Stream object rather than the legacy wrapper. Identity for normal
            # (already-parsed) streams.
            stream = _openai_unwrap_legacy_response(response)
            tee = OpenAIRecordingStream(stream, messages, acc)
            # A with_raw_response caller expects to call .parse() on the return
            # value to get the stream; give it back a shim that satisfies that
            # surface while still handing out the (recording) tee.
            if _openai_wants_raw_response(kwargs):
                return _OpenAIRawResponseStreamShim(tee, response)
            return tee

        kwargs["stream"] = False
        response = original(self, *args, **kwargs)
        parsed = _openai_unwrap_legacy_response(response)
        text = ""
        tokens = 0
        if hasattr(parsed, "choices") and parsed.choices:
            text = parsed.choices[0].message.content or ""
        if hasattr(parsed, "usage"):
            tokens = getattr(parsed.usage, "total_tokens", 0)
        acc.push(
            {
                "type": "llm_call",
                "messages": messages,
                "response": text,
                "tokens": tokens,
                "raw_response": _openai_dump_raw(parsed),
                "tool_requests": _openai_extract_tool_requests(parsed),
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
            reconstructed = _openai_reconstruct_event(event)
            # Callers that used with_raw_response (e.g. langchain-openai) expect
            # a legacy-response-like object with .parse() back — there's no real
            # LegacyAPIResponse in replay since no HTTP call was made.
            if _openai_wants_raw_response(kwargs):
                return _OpenAIReplayLegacyResponse(reconstructed)
            return reconstructed

        if kwargs.get("stream"):
            response = await original(self, *args, **kwargs)
            # Unwrap a legacy raw-response wrapper (e.g. from langchain-openai)
            # so the tee iterates the real async Stream object; identity-safe
            # for already-parsed streams.
            stream = _openai_unwrap_legacy_response(response)
            tee = AsyncOpenAIRecordingStream(stream, messages, acc)
            # A with_raw_response caller expects to call .parse() on the return
            # value to get the stream; give it back a shim that satisfies that
            # surface while still handing out the (recording) tee.
            if _openai_wants_raw_response(kwargs):
                return _OpenAIRawResponseStreamShim(tee, response)
            return tee

        kwargs["stream"] = False
        response = await original(self, *args, **kwargs)
        parsed = _openai_unwrap_legacy_response(response)
        text = ""
        tokens = 0
        if hasattr(parsed, "choices") and parsed.choices:
            text = parsed.choices[0].message.content or ""
        if hasattr(parsed, "usage"):
            tokens = getattr(parsed.usage, "total_tokens", 0)
        acc.push(
            {
                "type": "llm_call",
                "messages": messages,
                "response": text,
                "tokens": tokens,
                "raw_response": _openai_dump_raw(parsed),
                "tool_requests": _openai_extract_tool_requests(parsed),
            }
        )
        return response

    AsyncCompletions.create = _interceptor
    return [(AsyncCompletions, "create", original)]


def _apply_openai_responses() -> list[tuple]:
    from openai.resources.responses.responses import Responses

    original = Responses.create

    def _interceptor(self, *args, **kwargs):
        acc = TraceAccumulator.current()
        if acc is None:
            return original(self, *args, **kwargs)

        if acc.replay is not None:
            event = acc.replay.next_llm_event()
            pushed = {
                "type": "llm_call",
                "messages": _openai_normalize_responses_input(kwargs),
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
            reconstructed = _openai_reconstruct_response_event(event)
            # Callers that used with_raw_response (e.g. langchain-openai's
            # Responses route) expect a legacy-response-like object with
            # .parse() back — there's no real LegacyAPIResponse in replay
            # since no HTTP call was made.
            if _openai_wants_raw_response(kwargs):
                return _OpenAIReplayLegacyResponse(reconstructed)
            return reconstructed

        if kwargs.get("stream"):
            # Streamed Responses API runs pass through unrecorded this release —
            # a documented limitation (no stream recordings exist for responses).
            return original(self, *args, **kwargs)

        messages = _openai_normalize_responses_input(kwargs)
        response = original(self, *args, **kwargs)
        # Some callers (e.g. langchain-openai's Responses route) request the
        # raw HTTP response via with_raw_response; unwrap so we extract from
        # the real parsed Response object. Identity for normal responses.
        parsed = _openai_unwrap_legacy_response(response)
        usage = getattr(parsed, "usage", None)
        acc.push(
            {
                "type": "llm_call",
                "messages": messages,
                "response": _openai_extract_responses_text(parsed),
                "tokens": getattr(usage, "total_tokens", 0) or 0,
                "raw_response": _openai_dump_raw(parsed),
                "tool_requests": _openai_extract_responses_tool_requests(parsed),
            }
        )
        return response

    Responses.create = _interceptor
    return [(Responses, "create", original)]


def _apply_openai_responses_async() -> list[tuple]:
    from openai.resources.responses.responses import AsyncResponses

    original = AsyncResponses.create

    async def _interceptor(self, *args, **kwargs):
        acc = TraceAccumulator.current()
        if acc is None:
            return await original(self, *args, **kwargs)

        if acc.replay is not None:
            event = acc.replay.next_llm_event()
            pushed = {
                "type": "llm_call",
                "messages": _openai_normalize_responses_input(kwargs),
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
            reconstructed = _openai_reconstruct_response_event(event)
            # Callers that used with_raw_response (e.g. langchain-openai's
            # Responses route) expect a legacy-response-like object with
            # .parse() back — there's no real LegacyAPIResponse in replay
            # since no HTTP call was made.
            if _openai_wants_raw_response(kwargs):
                return _OpenAIReplayLegacyResponse(reconstructed)
            return reconstructed

        if kwargs.get("stream"):
            # Streamed Responses API runs pass through unrecorded this release —
            # a documented limitation (no stream recordings exist for responses).
            return await original(self, *args, **kwargs)

        messages = _openai_normalize_responses_input(kwargs)
        response = await original(self, *args, **kwargs)
        # Some callers (e.g. langchain-openai's Responses route) request the
        # raw HTTP response via with_raw_response; unwrap so we extract from
        # the real parsed Response object. Identity for normal responses.
        parsed = _openai_unwrap_legacy_response(response)
        usage = getattr(parsed, "usage", None)
        acc.push(
            {
                "type": "llm_call",
                "messages": messages,
                "response": _openai_extract_responses_text(parsed),
                "tokens": getattr(usage, "total_tokens", 0) or 0,
                "raw_response": _openai_dump_raw(parsed),
                "tool_requests": _openai_extract_responses_tool_requests(parsed),
            }
        )
        return response

    AsyncResponses.create = _interceptor
    return [(AsyncResponses, "create", original)]


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
    _apply_openai_responses,
    _apply_openai_responses_async,
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
        try:
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
        except Exception:
            # Roll back any patches already applied this __enter__ before
            # propagating — otherwise a mid-loop failure (e.g. warnings
            # escalated to errors) leaves the SDK partially patched.
            self.__exit__()
            raise
        return self

    def __exit__(self, *args) -> None:
        for cls, attr, original in self._applied:
            setattr(cls, attr, original)
        self._applied.clear()
