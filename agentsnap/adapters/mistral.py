from __future__ import annotations

from agentsnap.core.recorder import TraceAccumulator
from agentsnap.exceptions import ReplayError


class _MistralCompletionsProxy:
    def __init__(self, original) -> None:
        self._original = original

    def complete(self, **kwargs):
        acc = TraceAccumulator.current()
        if acc is None:
            return self._original.complete(**kwargs)
        if acc.replay is not None:
            raise ReplayError(
                "replay mode does not yet support Mistral; "
                "use mode='live' for this test."
            )

        messages = kwargs.get("messages", [])
        kwargs["stream"] = False
        response = self._original.complete(**kwargs)

        response_text = ""
        tokens = 0
        if hasattr(response, "choices") and response.choices:
            msg = response.choices[0].message
            response_text = msg.content or ""
        if hasattr(response, "usage"):
            tokens = getattr(response.usage, "total_tokens", 0) or 0

        acc.push(
            {
                "type": "llm_call",
                "messages": messages,
                "response": response_text,
                "tokens": tokens,
            }
        )
        return response

    def __getattr__(self, name: str):
        return getattr(self._original, name)


class _MistralChatProxy:
    def __init__(self, chat) -> None:
        self._chat = chat
        self.complete = _MistralCompletionsProxy(chat).complete

    def __getattr__(self, name: str):
        return getattr(self._chat, name)


class MistralAdapter:
    """Wraps a mistralai.Mistral() client to intercept .chat.complete().

    Usage:
        from mistralai import Mistral
        from agentsnap.adapters.mistral import MistralAdapter
        client = MistralAdapter(Mistral(api_key="..."))
        response = client.chat.complete(model="mistral-large-latest", messages=[...])
    """

    def __init__(self, client) -> None:
        self._client = client
        self.chat = _MistralChatProxy(client.chat)

    def __getattr__(self, name: str):
        return getattr(self._client, name)
