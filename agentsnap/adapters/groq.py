from __future__ import annotations

from agentsnap.adapters.openai import OpenAIAdapter


class GroqAdapter(OpenAIAdapter):
    """Wraps a groq.Groq() client (OpenAI-compatible interface).

    Groq uses the same .chat.completions.create() interface as OpenAI,
    so this is a thin alias over OpenAIAdapter.

    Usage:
        from groq import Groq
        from agentsnap.adapters.groq import GroqAdapter
        client = GroqAdapter(Groq(api_key="..."))
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile", messages=[...]
        )
    """
