from __future__ import annotations

from agenttest.core.recorder import TraceAccumulator


class _CompletionsProxy:
    def __init__(self, original) -> None:
        self._original = original

    def create(self, **kwargs):
        acc = TraceAccumulator.current()
        if acc is None:
            return self._original.create(**kwargs)

        messages = kwargs.get("messages", [])
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
            }
        )
        return response

    def __getattr__(self, name: str):
        return getattr(self._original, name)


class _ChatProxy:
    def __init__(self, chat) -> None:
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
