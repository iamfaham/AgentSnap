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
        return dump()
    except Exception:
        return None


def reconstruct(raw: dict):
    """Rebuild an openai ChatCompletion object from a recorded raw_response dict."""
    from openai.types.chat import ChatCompletion

    return ChatCompletion.model_validate(raw)


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
            return reconstruct(event["raw_response"])

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
