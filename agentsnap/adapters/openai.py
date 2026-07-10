from __future__ import annotations

from agentsnap.core.recorder import TraceAccumulator


def dump_raw(response) -> dict | None:
    """Serialize a provider response for replay. None if the object can't dump."""
    dump = getattr(response, "model_dump", None)
    if dump is None:
        return None
    try:
        return dump(mode="json")
    except TypeError:
        try:
            return dump()
        except Exception:
            return None
    except Exception:
        return None


def reconstruct(raw: dict):
    """Rebuild an openai ChatCompletion object from a recorded raw_response dict."""
    from openai.types.chat import ChatCompletion

    return ChatCompletion.model_validate(raw)


def reconstruct_event(event: dict):
    """Rebuild the recorded response for a replayed event, with a clear error on failure."""
    from agentsnap.exceptions import ReplayError

    try:
        return reconstruct(event["raw_response"])
    except ReplayError:
        raise
    except Exception as e:
        raise ReplayError(
            f"Failed to reconstruct the recorded response for llm_call step "
            f"{event.get('step', '?')} — the snapshot may be corrupt or recorded "
            f"under a different SDK version ({e}). "
            "Re-record the golden: pytest --agentsnap-record"
        ) from e


class OpenAIRecordingStream:
    """Tees a streaming response: yields chunks unchanged, records the assembled call."""

    def __init__(self, inner, messages, acc) -> None:
        self._inner = inner
        self._messages = messages
        self._acc = acc
        self._chunks: list = []
        self._text: list[str] = []
        self._tokens = 0
        self._recorded = False

    def __iter__(self):
        # finally covers natural exhaustion, consumer break (GeneratorExit),
        # and mid-stream exceptions — the partial call is always recorded.
        try:
            for chunk in self._inner:
                self._capture(chunk)
                yield chunk
        finally:
            self._record()

    def _capture(self, chunk) -> None:
        self._chunks.append(chunk)
        if getattr(chunk, "choices", None):
            delta = chunk.choices[0].delta
            if getattr(delta, "content", None):
                self._text.append(delta.content)
        usage = getattr(chunk, "usage", None)
        if usage is not None:
            self._tokens = getattr(usage, "total_tokens", 0) or 0

    def _record(self) -> None:
        if self._recorded:
            return
        self._recorded = True
        self._acc.push(
            {
                "type": "llm_call",
                "messages": self._messages,
                "response": "".join(self._text),
                "tokens": self._tokens,
                "raw_response": {
                    "__stream__": True,
                    "chunks": [dump_raw(c) for c in self._chunks],
                },
            }
        )

    def close(self) -> None:
        try:
            self._inner.close()
        finally:
            self._record()

    def __enter__(self):
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def __getattr__(self, name):
        return getattr(self._inner, name)


class _CompletionsProxy:
    def __init__(self, original) -> None:
        self._original = original

    def create(self, **kwargs):
        acc = TraceAccumulator.current()
        if acc is None:
            return self._original.create(**kwargs)

        messages = kwargs.get("messages", [])

        if acc.replay is not None:
            event = acc.replay.next_llm_event()
            acc.push(
                {
                    "type": "llm_call",
                    "messages": messages,
                    "response": event.get("response", ""),
                    "tokens": event.get("tokens", 0),
                    "raw_response": event.get("raw_response"),
                }
            )
            return reconstruct_event(event)

        if kwargs.get("stream"):
            response = self._original.create(**kwargs)
            return OpenAIRecordingStream(response, messages, acc)

        # Force non-streaming so we always get a complete ChatCompletion object.
        # Streaming responses expose deltas, not the full message content.
        kwargs["stream"] = False
        response = self._original.create(**kwargs)

        response_text = ""
        tokens = 0
        if hasattr(response, "choices") and response.choices:
            msg = response.choices[0].message
            response_text = msg.content or ""
        if hasattr(response, "usage"):
            tokens = getattr(response.usage, "total_tokens", 0)

        acc.push(
            {
                "type": "llm_call",
                "messages": messages,
                "response": response_text,
                "tokens": tokens,
                "raw_response": dump_raw(response),
            }
        )
        return response

    def __getattr__(self, name: str):
        return getattr(self._original, name)


class _ChatProxy:
    def __init__(self, chat) -> None:
        self._chat = chat
        self.completions = _CompletionsProxy(chat.completions)

    def __getattr__(self, name: str):
        return getattr(self._chat, name)


class OpenAIAdapter:
    """Wraps an openai.OpenAI() client to intercept .chat.completions.create()."""

    def __init__(self, client) -> None:
        self._client = client
        self.chat = _ChatProxy(client.chat)

    def __getattr__(self, name: str):
        return getattr(self._client, name)
