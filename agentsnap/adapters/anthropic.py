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
    """Rebuild an anthropic Message object from a recorded raw_response dict."""
    from anthropic.types import Message

    return Message.model_validate(raw)


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


class _MessagesProxy:
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

        response = self._original.create(**kwargs)

        response_text = ""
        tokens = 0
        if hasattr(response, "content"):
            for block in response.content:
                if hasattr(block, "text"):
                    response_text += block.text
        if hasattr(response, "usage"):
            tokens = getattr(response.usage, "input_tokens", 0) + getattr(
                response.usage, "output_tokens", 0
            )

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


class AnthropicAdapter:
    """Wraps an anthropic.Anthropic() client to intercept .messages.create()."""

    def __init__(self, client) -> None:
        self._client = client
        self.messages = _MessagesProxy(client.messages)

    def __getattr__(self, name: str):
        return getattr(self._client, name)
