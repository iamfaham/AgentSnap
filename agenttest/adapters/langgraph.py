from __future__ import annotations

from agenttest.core.recorder import TraceAccumulator


class LangGraphAdapter:
    """Wraps a CompiledGraph to intercept .invoke() calls.

    Tool calls within the graph should be individually wrapped with ToolAdapter
    for fine-grained tracing. This adapter records the overall graph invocation.
    """

    def __init__(self, graph) -> None:
        self._graph = graph

    def invoke(self, input_data, **kwargs):
        acc = TraceAccumulator.current()
        if acc is None:
            return self._graph.invoke(input_data, **kwargs)

        result = self._graph.invoke(input_data, **kwargs)

        acc.push(
            {
                "type": "llm_call",
                "messages": [{"role": "user", "content": str(input_data)}],
                "response": str(result),
                "tokens": 0,
            }
        )
        return result

    def stream(self, input_data, **kwargs):
        return self._graph.stream(input_data, **kwargs)

    def __getattr__(self, name: str):
        return getattr(self._graph, name)
