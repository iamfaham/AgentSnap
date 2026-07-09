"""
agentsnap.wrap() -- auto-detect and wrap any supported client or callable.

Instead of importing a specific adapter:
    from agentsnap.adapters.anthropic import AnthropicAdapter
    client = AnthropicAdapter(anthropic.Anthropic())

Just write:
    import agentsnap
    client = agentsnap.wrap(anthropic.Anthropic())

Supported types (detected by class name, no hard SDK dependency):
    anthropic.Anthropic          -> AnthropicAdapter
    openai.OpenAI (OpenRouter)   -> OpenRouterAdapter
    openai.OpenAI (default)      -> OpenAIAdapter
    google.genai.Client          -> GeminiAdapter
    cohere.ClientV2              -> CohereAdapter
    mistralai.Mistral            -> MistralAdapter
    groq.Groq                    -> GroqAdapter
    any callable                 -> ToolAdapter
"""

from __future__ import annotations

from typing import Any

from agentsnap.adapters.tool import ToolAdapter

_OPENROUTER_HOST = "openrouter.ai"


def _class_name(obj: Any) -> str:
    t = type(obj)
    return f"{t.__module__}.{t.__qualname__}"


def wrap(obj: Any, name: str | None = None) -> Any:
    """Wrap a provider client or callable with the appropriate agentsnap adapter.

    Parameters
    ----------
    obj:
        An SDK client (anthropic.Anthropic, openai.OpenAI, etc.) or any callable.
    name:
        Tool name — only used when wrapping a callable. Defaults to the
        function's __name__.

    Returns
    -------
    The appropriate adapter instance, or a ToolAdapter for callables.

    Raises
    ------
    TypeError if the object is not a recognised client and not callable.
    """
    cls = _class_name(obj)

    # -- Anthropic ------------------------------------------------------------
    if "anthropic" in cls and "Anthropic" in cls:
        from agentsnap.adapters.anthropic import AnthropicAdapter
        return AnthropicAdapter(obj)

    # -- OpenAI / OpenRouter / Groq (all share openai.OpenAI) ----------------
    if "openai" in cls and "OpenAI" in cls:
        base_url = str(getattr(obj, "base_url", ""))
        if _OPENROUTER_HOST in base_url:
            from agentsnap.adapters.openrouter import OpenRouterAdapter
            return OpenRouterAdapter(obj)
        from agentsnap.adapters.openai import OpenAIAdapter
        return OpenAIAdapter(obj)

    # -- Groq (uses its own client class, not openai.OpenAI) -----------------
    if "groq" in cls and "Groq" in cls:
        from agentsnap.adapters.groq import GroqAdapter
        return GroqAdapter(obj)

    # -- Google Gemini --------------------------------------------------------
    if "genai" in cls and "Client" in cls:
        from agentsnap.adapters.google import GeminiAdapter
        return GeminiAdapter(obj)

    # -- Cohere ---------------------------------------------------------------
    if "cohere" in cls and "Client" in cls:
        from agentsnap.adapters.cohere import CohereAdapter
        return CohereAdapter(obj)

    # -- Mistral --------------------------------------------------------------
    if "mistral" in cls and "Mistral" in cls:
        from agentsnap.adapters.mistral import MistralAdapter
        return MistralAdapter(obj)

    # -- LangGraph compiled graph ---------------------------------------------
    if hasattr(obj, "invoke") and hasattr(obj, "get_graph"):
        from agentsnap.adapters.langgraph import LangGraphAdapter
        return LangGraphAdapter(obj)

    # -- Any callable -> ToolAdapter ------------------------------------------
    if callable(obj):
        tool_name = name or getattr(obj, "__name__", "unknown_tool")
        return ToolAdapter(obj, name=tool_name)

    raise TypeError(
        f"agentsnap.wrap(): cannot wrap {cls!r}. "
        "Pass a supported SDK client or a callable."
    )
