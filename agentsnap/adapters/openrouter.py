from __future__ import annotations

from agentsnap.adapters.openai import OpenAIAdapter

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterAdapter(OpenAIAdapter):
    """Wraps an openai.OpenAI() client pointed at OpenRouter.

    OpenRouter exposes an OpenAI-compatible API that proxies 300+ models
    (Claude, GPT, Gemini, Llama, Mistral, etc.) under one key.

    Usage:
        import openai
        from agentsnap.adapters.openrouter import OpenRouterAdapter, OPENROUTER_BASE_URL

        client = OpenRouterAdapter(
            openai.OpenAI(
                api_key="sk-or-...",
                base_url=OPENROUTER_BASE_URL,
            )
        )
        response = client.chat.completions.create(
            model="anthropic/claude-haiku-4-5",   # or any OpenRouter model slug
            messages=[{"role": "user", "content": "Hello"}],
        )
    """
