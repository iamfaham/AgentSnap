from __future__ import annotations

from agentsnap.core.recorder import TraceAccumulator
from agentsnap.exceptions import ReplayError


class _ModelsProxy:
    def __init__(self, original) -> None:
        self._original = original

    def generate_content(self, model: str, contents, **kwargs):
        acc = TraceAccumulator.current()
        if acc is None:
            return self._original.generate_content(model=model, contents=contents, **kwargs)
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

        response = self._original.generate_content(model=model, contents=contents, **kwargs)

        response_text = ""
        tokens = 0
        if hasattr(response, "text"):
            response_text = response.text or ""
        elif hasattr(response, "candidates") and response.candidates:
            for part in response.candidates[0].content.parts:
                if hasattr(part, "text"):
                    response_text += part.text
        if hasattr(response, "usage_metadata"):
            tokens = getattr(response.usage_metadata, "total_token_count", 0)

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


class GeminiAdapter:
    """Wraps a google.genai.Client() to intercept .models.generate_content().

    Usage:
        from google import genai
        from agentsnap.adapters.google import GeminiAdapter
        client = GeminiAdapter(genai.Client(api_key="..."))
        response = client.models.generate_content(model="gemini-2.0-flash", contents="Hello")
    """

    def __init__(self, client) -> None:
        self._client = client
        self.models = _ModelsProxy(client.models)

    def __getattr__(self, name: str):
        return getattr(self._client, name)
