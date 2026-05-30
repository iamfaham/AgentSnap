# agentsnap

Deterministic snapshot testing for AI agents.

`agentsnap` wraps your agent's LLM and tool calls during a **golden run** to produce a committed snapshot file. On every subsequent run it replays the same inputs and compares the new trace against the snapshot across three dimensions:

| Dimension | What it checks |
|-----------|----------------|
| **Structural** | Tool call names and order match exactly |
| **Arguments** | JSON diff on tool call arguments (configurable ignored fields) |
| **Semantic** | Cosine similarity of LLM outputs via `all-MiniLM-L6-v2` (default threshold: 0.92) |

If any dimension drifts beyond its threshold, `agentsnap` raises `AgentRegressionError` with a structured diff report.

---

## 3-minute quickstart

### 1 — Install

```bash
pip install agentsnap
```

### 2 — Record a golden run

```python
# my_agent_test.py
from agentsnap import AgentRecorder
from agentsnap.adapters.anthropic import AnthropicAdapter
from agentsnap.adapters.tool import ToolAdapter
import anthropic

def search(query: str) -> str:
    return f"Results for: {query}"

client = AnthropicAdapter(anthropic.Anthropic())
search_tool = ToolAdapter(search, name="search")

with AgentRecorder("my_agent_test") as recorder:
    result = my_agent(client, search_tool, input="What is Python?")
    recorder.output = result
```

```bash
python my_agent_test.py
# Writes __agent_snapshots__/my_agent_test.json
```

Commit the snapshot file to version control.

### 3 — Assert on future runs

```python
from agentsnap import AgentAsserter

with AgentAsserter("my_agent_test") as asserter:
    result = my_agent(client, search_tool, input="What is Python?")
    asserter.output = result
# Raises AgentRegressionError if behavior drifted
```

### 4 — Use the pytest fixture

```python
# test_my_agent.py
def test_my_agent_behavior(snapshot):
    with snapshot.assert_agent("my_agent_test") as asserter:
        result = my_agent(client, search_tool, input="What is Python?")
        asserter.output = result
```

```bash
pytest test_my_agent.py
```

---

## API reference

### `AgentRecorder(test_name, snapshot_dir="__agent_snapshots__", model="unknown")`

Context manager. Intercepts all adapter calls inside the `with` block and writes a snapshot on clean exit.

```python
with AgentRecorder("name") as rec:
    rec.input_data = {"query": "hello"}   # optional, stored in snapshot
    result = my_agent(wrapped_client, ...)
    rec.output = result
```

### `AgentAsserter(test_name, snapshot_dir="__agent_snapshots__", semantic_threshold=0.92, ignored_fields=None)`

Context manager. Reads the snapshot, intercepts calls, then runs the three-layer diff on exit.

```python
with AgentAsserter("name", semantic_threshold=0.95, ignored_fields=["timestamp"]) as a:
    result = my_agent(wrapped_client, ...)
    a.output = result
```

### `AnthropicAdapter(client)`

Wraps `anthropic.Anthropic()` — intercepts `.messages.create()`.

### `OpenAIAdapter(client)`

Wraps `openai.OpenAI()` — intercepts `.chat.completions.create()`.

### `LangGraphAdapter(graph)`

Wraps a compiled LangGraph — intercepts `.invoke()`.

### `ToolAdapter(func, name=None)`

Wraps any callable — records `type=tool_call` events.

```python
search = ToolAdapter(my_search_fn, name="search")
result = search(query="hello")
```

### `snapshot` pytest fixture

Provides two methods:

- `snapshot.record_agent(test_name, model="unknown")` → `AgentRecorder`
- `snapshot.assert_agent(test_name, semantic_threshold=0.92, ignored_fields=None)` → `AgentAsserter`

Auto-discovers `__agent_snapshots__/` relative to the nearest `conftest.py`.

### Exceptions

| Exception | When raised |
|-----------|-------------|
| `AgentRegressionError(message, diff_report)` | Behavior drifted beyond threshold |
| `SnapshotNotFoundError(test_name)` | No snapshot file found — run record first |
| `AdapterNotWrappedError` | Unwrapped client used inside a recording context |

---

## CLI

```bash
agentsnap record <test_file>          # run file in record mode
agentsnap run <test_file>             # run file in assert mode
agentsnap diff <snapshot_file>        # pretty-print snapshot
agentsnap update <test_name>          # approve last run as new golden
agentsnap list [--snapshot-dir DIR]   # list all snapshots
```

---

## Snapshot format

Snapshots are stored in `__agent_snapshots__/<test_name>.json`:

```json
{
  "version": "1.0",
  "recorded_at": "2026-01-01T00:00:00+00:00",
  "model": "claude-opus-4-7",
  "input": { "query": "What is Python?" },
  "trace": [
    { "step": 0, "type": "llm_call", "messages": [...], "response": "...", "tokens": 350 },
    { "step": 1, "type": "tool_call", "name": "search", "args": {"query": "Python"}, "result": "..." }
  ],
  "output": "Python is a high-level programming language..."
}
```

Commit snapshot files. They are the source of truth for agent behavior.

---

## CI integration (GitHub Actions)

```yaml
name: Agent regression tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: pip

      - name: Install dependencies
        run: pip install -e ".[dev]"

      - name: Run agent snapshot tests
        run: pytest tests/ -v
        env:
          # Real API calls are not made in tests — adapters are mocked.
          # Set keys only if your integration tests hit live APIs.
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
```

---

## How to approve an intentional regression

When you intentionally change agent behavior (new prompt, new tool, new model), update the golden snapshot:

```bash
# 1. Run the asserter — it fails and saves the new trace to .last_run/
pytest tests/test_my_agent.py  # fails with AgentRegressionError

# 2. Inspect the diff
agentsnap diff __agent_snapshots__/my_agent_test.json

# 3. Approve the new behavior
agentsnap update my_agent_test

# 4. Commit the updated snapshot
git add __agent_snapshots__/my_agent_test.json
git commit -m "approve: updated agent behavior for my_agent_test"
```

The `update` command copies `.last_run/<test_name>.json` over the golden snapshot. The next CI run will use the new snapshot as the baseline.

---

## Configuring the semantic threshold

Threshold is per-assertion — tighten it for critical paths, loosen it for exploratory agents:

```python
# Tight — must be nearly identical
with AgentAsserter("critical_path", semantic_threshold=0.98) as a:
    ...

# Loose — allow more paraphrasing
with AgentAsserter("creative_output", semantic_threshold=0.80) as a:
    ...
```

The default `0.92` works well for factual, structured agents. Lower it for creative or stochastic agents.
