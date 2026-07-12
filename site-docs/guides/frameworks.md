# Frameworks

Frameworks build their own SDK clients internally, so there's nothing to wrap — `PatchSet` patches the underlying SDK classes (sync and async Anthropic/OpenAI chat, plus the OpenAI Responses API), so any framework built on top of them is captured automatically.

| Framework | How | CI-verified |
|-----------|-----|-------------|
| Pydantic AI | `PatchSet` — async OpenAI/Anthropic clients | Yes |
| OpenAI Agents SDK | `PatchSet` — Responses API | Yes |
| LangChain | `PatchSet` — sync + async chat | Yes |
| LangGraph | `LangGraphAdapter` for node-level events, or `PatchSet` | Yes (existing) |
| CrewAI | Works via LiteLLM's OpenAI-compatible sync path | Documented, not CI-verified |

The universal pattern — wrap the framework call, nothing else changes:

```python
from agentsnap import PatchSet
from agentsnap.core.asserter import AgentAsserter

with PatchSet():
    with AgentAsserter("my_framework_agent") as a:
        a.output = my_pydantic_ai_agent.run_sync("What is Python?").output
```

## Caveats

- Streamed OpenAI Responses-API runs (`responses.create(stream=True)`) pass through unrecorded this iteration — non-streaming Responses calls and all chat-completions streaming (sync + async) are recorded and replayable.
- The model-tools check (see [Model tools](model-tools.md)) is gated trace-wide: if any call in the trace is a streamed call or a non-Anthropic/OpenAI provider, the whole run's `model_tools`/`model_tool_args` comparison is skipped.

Real-framework verification tests live in `tests/frameworks/` (marker `frameworks`, `pytest.importorskip`-guarded, run via a separate CI job with `.[dev,frameworks]` installed) — they drive each framework's real code path through an offline mock transport, asserting on agentsnap's recorded trace, not framework internals.

## LangGraph

Wrap your compiled graph with `LangGraphAdapter`. It injects a callback handler that captures LLM and tool events at the node level, not just the top-level graph invocation:

```python
from agentsnap.adapters.langgraph import LangGraphAdapter

graph = build_my_graph()   # your compiled StateGraph
agent = LangGraphAdapter(graph)

with AgentRecorder("langgraph_agent") as rec:
    result = agent.invoke({"messages": [HumanMessage(content="Hello")]})
    rec.output = str(result)
```

Node-level capture means the snapshot reflects what each node in your graph actually called — tool calls within nodes, intermediate LLM calls, and their responses — not just the final output.
