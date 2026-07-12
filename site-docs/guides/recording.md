# Recording

## Zero-instrumentation with PatchSet

`PatchSet` patches the SDK classes directly so you do not need to change any agent code:

```python
from agentsnap import PatchSet, AgentRecorder

with PatchSet():
    with AgentRecorder("my_agent") as rec:
        client = anthropic.Anthropic()    # raw client — no adapter
        result = my_agent(client, "What is Python?")
        rec.output = result
```

`PatchSet` covers all installed SDKs simultaneously. SDKs that are not installed are silently skipped.

**In pytest — per test:**

```python
def test_my_agent(snapshot, agentsnap_instrument):
    with snapshot.run("my_agent") as s:
        client = anthropic.Anthropic()
        s.output = my_agent(client, "query")
```

**In pytest — all tests in a session:**

```bash
pytest --agentsnap-instrument
```

**Project-wide (autouse in conftest.py):**

```python
# conftest.py
import pytest

@pytest.fixture(autouse=True)
def _(agentsnap_instrument):
    pass
```

!!! warning
    Do not use `PatchSet` together with an adapter wrapper on the same client. Both interceptors will fire and events will be recorded twice, producing a corrupted trace. Use one or the other.

## Using adapters (alternative)

Each supported provider has an adapter that wraps the SDK client explicitly. Use it everywhere you would use the raw client — useful when you want to be explicit about what is captured, or when you control the agent code directly:

```python
import anthropic
from agentsnap.adapters.anthropic import AnthropicAdapter
from agentsnap.adapters.tool import ToolAdapter

# Wrap the client
client = AnthropicAdapter(anthropic.Anthropic())

# Wrap tools
def search(query: str) -> str:
    return f"Results for: {query}"

search_tool = ToolAdapter(search, name="search")
```

Now use `client` and `search_tool` exactly as you would use the unwrapped versions. The adapters are transparent pass-throughs when no recording context is active.

**Available adapters:**

```python
from agentsnap.adapters.anthropic  import AnthropicAdapter
from agentsnap.adapters.openai     import OpenAIAdapter
from agentsnap.adapters.gemini     import GeminiAdapter
from agentsnap.adapters.cohere     import CohereAdapter
from agentsnap.adapters.mistral    import MistralAdapter
from agentsnap.adapters.groq       import GroqAdapter      # subclass of OpenAIAdapter
from agentsnap.adapters.openrouter import OpenRouterAdapter  # subclass of OpenAIAdapter
from agentsnap.adapters.langgraph  import LangGraphAdapter
from agentsnap.adapters.tool       import ToolAdapter
```

**Recording with adapters:**

```python
from agentsnap import AgentRecorder

with AgentRecorder("my_agent", model="claude-haiku-4-5") as rec:
    rec.input_data = {"query": "What is Python?"}   # optional — stored in snapshot
    result = my_agent(client, search_tool, "What is Python?")
    rec.output = result
```

**Asserting with adapters:**

```python
from agentsnap import AgentAsserter

with AgentAsserter("my_agent") as a:
    result = my_agent(client, search_tool, "What is Python?")
    a.output = result
```

If the trace matches the golden snapshot, the context manager exits cleanly. If anything drifted, it raises `AgentRegressionError`.

## pytest fixture

The `snapshot` fixture is registered automatically when agentsnap is installed — no imports needed in your test files.

```python
def test_agent(snapshot):
    # Auto mode (recommended): records on first run, asserts on every run after
    with snapshot.run("test_name") as s:
        s.output = my_agent(...)

    # Explicit record
    with snapshot.record_agent("test_name") as rec:
        rec.output = my_agent(...)

    # Explicit assert
    with snapshot.assert_agent("test_name") as a:
        a.output = my_agent(...)
```

Per-test overrides:

```python
def test_agent(snapshot):
    # Tighter threshold for a critical agent
    with snapshot.assert_agent("name", semantic_threshold=0.98) as a:
        a.output = my_agent(...)

    # Force embeddings even when LLM judge is configured
    with snapshot.assert_agent("name", judge=False) as a:
        a.output = my_agent(...)

    # Ignore volatile fields in tool arguments
    with snapshot.assert_agent("name", ignored_fields=["timestamp", "request_id"]) as a:
        a.output = my_agent(...)
```

By default agentsnap places snapshots in `__agent_snapshots__/` relative to the nearest `conftest.py`. To use a different location:

```python
with AgentRecorder("name", snapshot_dir="/path/to/snapshots") as rec:
    ...
```

See [Configuration](configuration.md) for the full list of pytest flags and ini options.

## Snapshot files and format

Golden snapshots live in `__agent_snapshots__/` and are committed to git. The `.last_run/` subdirectory is written on every assert run and is gitignored — it is only used by `agentsnap update`.

```json
{
  "version": "1.1",
  "recorded_at": "2026-01-01T00:00:00+00:00",
  "model": "claude-haiku-4-5",
  "input": { "query": "What is Python?" },
  "trace": [
    { "step": 0, "type": "llm_call", "messages": [...], "response": "...", "tokens": 350, "raw_response": {...} },
    { "step": 1, "type": "tool_call", "name": "search", "args": {"query": "Python"}, "result": "..." }
  ],
  "output": "Python is a high-level programming language..."
}
```

## Scenarios and input binding

Multiple snapshots per test function are namespaced by a scenario name. If no explicit scenario is given, agentsnap derives one automatically from an `input_sha8` hash of whatever you assign to `rec.input_data` / `a.input`:

```python
with AgentRecorder("my_agent", scenario="short_question") as rec:
    ...
```

If you don't set `input_data`/`input` and don't pass `scenario=`, all runs of that test share a single unnamed snapshot. `agentsnap update <test_name>` promotes all scenario variants for a test at once (wildcard), not just one.

When the input captured inside the `with` block differs from what was recorded, `AgentAsserter` prints a warning (comparison may be against the wrong baseline) rather than failing silently.
