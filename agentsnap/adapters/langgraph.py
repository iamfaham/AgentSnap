from __future__ import annotations

from agentsnap.core.recorder import TraceAccumulator

try:
    from langchain_core.callbacks import BaseCallbackHandler as _Base
except ImportError:
    _Base = object  # type: ignore[assignment,misc]


class AgentSnapCallback(_Base):
    """LangChain callback handler that records LLM and tool events into TraceAccumulator.

    Injected automatically by LangGraphAdapter. Also works standalone with any
    LangChain model or chain that supports the callbacks API.

    When langchain_core is not installed, _Base is object — the class is still
    importable and its methods callable; it just won't be registered as a real
    LangChain callback (duck-typing makes it work with our mock graphs in tests).
    """

    def on_llm_end(self, response, **kwargs) -> None:
        acc = TraceAccumulator.current()
        if acc is None:
            return
        text = ""
        if hasattr(response, "generations") and response.generations:
            gen = response.generations[0][0]
            if hasattr(gen, "text") and gen.text:
                text = gen.text
            elif hasattr(gen, "message") and hasattr(gen.message, "content"):
                text = gen.message.content or ""
            else:
                text = str(gen)
        acc.push({"type": "llm_call", "messages": [], "response": text, "tokens": 0})

    def on_tool_end(self, output, *, name: str = "", **kwargs) -> None:
        acc = TraceAccumulator.current()
        if acc is None:
            return
        acc.push({"type": "tool_call", "name": name, "args": {}, "result": str(output)})


class LangGraphAdapter:
    """Wraps a CompiledGraph to capture traces via the LangChain callbacks system.

    Injects AgentSnapCallback into every invoke() call so that LLM and tool
    events from individual graph nodes are recorded into the active
    TraceAccumulator.

    Falls back to recording the top-level result as a single llm_call event
    when langchain_core is not installed (no callbacks support available).
    """

    def __init__(self, graph) -> None:
        self._graph = graph

    def invoke(self, input_data, **kwargs):
        acc = TraceAccumulator.current()
        if acc is None:
            return self._graph.invoke(input_data, **kwargs)

        # Inject AgentSnapCallback via config so node-level events are captured.
        # AgentSnapCallback is always usable via duck-typing even without langchain_core.
        # When langchain_core is absent (_Base is object), we still inject the callback
        # so that graphs that accept a callbacks list (e.g. LangGraph) fire it correctly.
        # The top-level fallback push only runs when langchain_core is absent, because
        # in that case a real CompiledGraph won't know to call on_llm_end/on_tool_end.
        config = dict(kwargs.pop("config", None) or {})
        callbacks = list(config.get("callbacks") or [])
        callbacks.append(AgentSnapCallback())
        config["callbacks"] = callbacks

        if _Base is not object:
            # langchain_core available: callback will be invoked by the real runtime
            return self._graph.invoke(input_data, config=config, **kwargs)

        # Fallback: langchain_core absent — real CompiledGraph won't fire callbacks,
        # so record the top-level invocation as a single llm_call event instead.
        result = self._graph.invoke(input_data, config=config, **kwargs)
        if not any(e["type"] == "llm_call" for e in acc.trace):
            acc.push({
                "type": "llm_call",
                "messages": [{"role": "user", "content": str(input_data)}],
                "response": str(result),
                "tokens": 0,
            })
        return result

    def stream(self, input_data, **kwargs):
        return self._graph.stream(input_data, **kwargs)

    def __getattr__(self, name: str):
        return getattr(self._graph, name)
