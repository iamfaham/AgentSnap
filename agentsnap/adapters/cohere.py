from __future__ import annotations

from agentsnap.core.recorder import TraceAccumulator
from agentsnap.exceptions import ReplayError


class _CohereV2ChatProxy:
    """Intercepts cohere.ClientV2().chat()."""

    def __init__(self, original_chat_fn) -> None:
        self._original = original_chat_fn

    def __call__(self, **kwargs):
        acc = TraceAccumulator.current()
        if acc is None:
            return self._original(**kwargs)
        if acc.replay is not None:
            raise ReplayError(
                "replay mode does not yet support Cohere; "
                "use mode='live' for this test."
            )

        messages = kwargs.get("messages", [])
        response = self._original(**kwargs)

        response_text = ""
        tokens = 0
        if hasattr(response, "message") and hasattr(response.message, "content"):
            for block in response.message.content:
                if hasattr(block, "text"):
                    response_text += block.text
        if hasattr(response, "usage"):
            tokens = getattr(response.usage, "tokens", None)
            if tokens is None:
                inp = getattr(response.usage, "input_tokens", 0) or 0
                out = getattr(response.usage, "output_tokens", 0) or 0
                tokens = inp + out

        acc.push(
            {
                "type": "llm_call",
                "messages": messages,
                "response": response_text,
                "tokens": tokens or 0,
            }
        )
        return response


class CohereAdapter:
    """Wraps a cohere.ClientV2() to intercept .chat().

    Usage:
        import cohere
        from agentsnap.adapters.cohere import CohereAdapter
        client = CohereAdapter(cohere.ClientV2(api_key="..."))
        response = client.chat(model="command-r-plus", messages=[...])
    """

    def __init__(self, client) -> None:
        self._client = client
        self.chat = _CohereV2ChatProxy(client.chat)

    def __getattr__(self, name: str):
        return getattr(self._client, name)
