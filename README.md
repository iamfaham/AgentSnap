# agentsnap

Deterministic snapshot testing for AI agents.

`agentsnap` wraps your agent's LLM and tool calls during a **golden run** to produce a committed snapshot file. On every subsequent run it replays the same inputs and compares the new trace against the snapshot across three dimensions:

| Dimension | What it checks | How |
|-----------|----------------|-----|
| **Structural** | Tool call names and order | Levenshtein edit distance on the tool sequence |
| **Arguments** | Tool call arguments | `deepdiff` (if installed) or plain dict diff, with configurable ignored fields |
| **Semantic** | LLM responses and final output | Cosine similarity via `all-MiniLM-L6-v2`, or an LLM judge for higher accuracy |

If any dimension drifts beyond its threshold, `agentsnap` raises `AgentRegressionError` with a structured diff report.

---

## 3-minute quickstart

### 1 — Install

```bash
pip install agentsnap
```

### 2 — Wrap your client and record a golden run

```python
from agentsnap import AgentRecorder
from agentsnap.adapters.anthropic import AnthropicAdapter
from agentsnap.adapters.tool import ToolAdapter
import anthropic

def search(query: str) -> str:
    return f"Results for: {query}"

client = AnthropicAdapter(anthropic.Anthropic())
search_tool = ToolAdapter(search, name="search")

with AgentRecorder("my_agent") as rec:
    result = my_agent(client, search_tool, input="What is Python?")
    rec.output = result
# Writes __agent_snapshots__/my_agent.json
```

Commit the snapshot file. It is the contract for what the agent does.

### 3 — Assert on future runs

```python
from agentsnap import AgentAsserter

with AgentAsserter("my_agent") as a:
    result = my_agent(client, search_tool, input="What is Python?")
    a.output = result
# Raises AgentRegressionError if behavior drifted
```

### 4 — Use the pytest fixture

```python
def test_my_agent(snapshot):
    with snapshot.assert_agent("my_agent") as a:
        result = my_agent(client, search_tool, input="What is Python?")
        a.output = result
```

```bash
pytest
```

---

## Supported providers

| Provider | Adapter | Intercepts |
|----------|---------|-----------|
| Anthropic | `AnthropicAdapter` | `.messages.create()` |
| OpenAI | `OpenAIAdapter` | `.chat.completions.create()` |
| Google Gemini | `GeminiAdapter` | `.models.generate_content()` |
| Cohere | `CohereAdapter` | `.chat()` |
| Mistral | `MistralAdapter` | `.chat.complete()` |
| Groq | `GroqAdapter` | `.chat.completions.create()` |
| OpenRouter | `OpenRouterAdapter` | `.chat.completions.create()` |
| LangGraph | `LangGraphAdapter` | `.invoke()` |
| Any callable | `ToolAdapter` | direct call |

Install provider SDKs as needed:

```bash
pip install agentsnap[google]    # google-genai
pip install agentsnap[cohere]    # cohere
pip install agentsnap[mistral]   # mistralai
pip install agentsnap[groq]      # groq
pip install agentsnap[all-providers]
```

---

## Configuration

### API key for the LLM judge (optional)

The LLM judge uses a small language model to compare outputs instead of embeddings — more accurate for factual content. Set the key as an environment variable (never in a file):

```bash
export AGENTSNAP_JUDGE_API_KEY=sk-or-...   # OpenRouter, OpenAI, or any compatible key
export AGENTSNAP_JUDGE_MODEL=openai/gpt-4o-mini      # optional
export AGENTSNAP_JUDGE_BASE_URL=https://openrouter.ai/api/v1  # optional
```

Once set, the `snapshot` pytest fixture enables the LLM judge automatically — no code changes needed in tests.

### Project settings (`pyproject.toml`)

```toml
[tool.agentsnap]
judge_model        = "openai/gpt-4o-mini"
judge_base_url     = "https://openrouter.ai/api/v1"
semantic_threshold = 0.92   # final agent output (strict)
llm_threshold      = 0.75   # intermediate LLM responses (tolerant)
```

These can also be set as pytest ini options:

```toml
[tool.pytest.ini_options]
agentsnap_judge_model        = "openai/gpt-4o-mini"
agentsnap_judge_base_url     = "https://openrouter.ai/api/v1"
agentsnap_semantic_threshold = "0.92"
agentsnap_llm_threshold      = "0.75"
```

---

## API reference

### `AgentRecorder(test_name, snapshot_dir="__agent_snapshots__", model="unknown")`

Context manager. Intercepts all adapter calls and writes a snapshot on clean exit.

```python
with AgentRecorder("name", model="claude-haiku-4-5") as rec:
    rec.input_data = {"query": "hello"}   # optional metadata
    result = my_agent(wrapped_client, ...)
    rec.output = result
```

### `AgentAsserter(test_name, snapshot_dir, semantic_threshold, llm_threshold, ignored_fields, embed_fn, judge)`

Context manager. Reads the snapshot, intercepts calls, runs the three-layer diff on exit.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `semantic_threshold` | `0.92` | Min similarity for final output |
| `llm_threshold` | `0.75` | Min similarity for intermediate LLM responses |
| `ignored_fields` | `None` | Tool arg keys to exclude from argument diff |
| `embed_fn` | `None` | Custom embedding function (for testing) |
| `judge` | `None` | `LLMJudge` instance; overrides embedding comparison |

```python
with AgentAsserter("name", semantic_threshold=0.95, ignored_fields=["timestamp"]) as a:
    result = my_agent(wrapped_client, ...)
    a.output = result
```

### `LLMJudge(api_key, model, base_url)`

Uses an LLM to score semantic equivalence instead of embeddings. Returns a `0.0–1.0` score and a one-sentence reason explaining any difference.

```python
from agentsnap import LLMJudge

# Explicit construction
judge = LLMJudge(api_key="sk-or-...", model="openai/gpt-4o-mini")

# From environment / pyproject.toml
judge = LLMJudge.from_env()  # returns None if AGENTSNAP_JUDGE_API_KEY is not set

with AgentAsserter("name", judge=judge) as a:
    ...
```

### `snapshot` pytest fixture

Auto-wired from `[tool.agentsnap]` and `AGENTSNAP_JUDGE_API_KEY`. No imports needed.

```python
def test_agent(snapshot):
    # Record
    with snapshot.record_agent("name") as rec:
        rec.output = run_agent(...)

    # Assert — judge enabled automatically if API key is set
    with snapshot.assert_agent("name") as a:
        a.output = run_agent(...)

    # Override per-test
    with snapshot.assert_agent("name", judge=False) as a:      # force embeddings
        a.output = run_agent(...)

    with snapshot.assert_agent("name", semantic_threshold=0.98) as a:  # tighter
        a.output = run_agent(...)
```

### Exceptions

| Exception | When raised |
|-----------|-------------|
| `AgentRegressionError(message, diff_report)` | Behavior drifted beyond threshold |
| `SnapshotNotFoundError(test_name)` | No snapshot found — record first |
| `AdapterNotWrappedError` | Unwrapped client used inside a recording context |

`AgentRegressionError.diff_report` is a `DiffReport` dataclass with `structural_diff`, `argument_diffs`, `semantic_scores`, `semantic_reasons`, and `failed_checks`.

---

## CLI

```bash
agentsnap list                        # list all snapshots
agentsnap diff __agent_snapshots__/my_agent.json   # pretty-print a snapshot
agentsnap update my_agent            # approve last run as new golden
agentsnap record <test_file>         # run file in record mode
agentsnap run <test_file>            # run file in assert mode
```

---

## Snapshot format

```json
{
  "version": "1.0",
  "recorded_at": "2026-01-01T00:00:00+00:00",
  "model": "claude-haiku-4-5",
  "input": { "query": "What is Python?" },
  "trace": [
    { "step": 0, "type": "llm_call", "messages": [...], "response": "...", "tokens": 350 },
    { "step": 1, "type": "tool_call", "name": "search", "args": {"query": "Python"}, "result": "..." }
  ],
  "output": "Python is a high-level programming language..."
}
```

Golden snapshots live in `__agent_snapshots__/` and are committed to git. The `.last_run/` subdirectory is written on every assert run and should be gitignored — it is only used by `agentsnap update`.

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

      - name: Install
        run: pip install -e ".[dev]"

      - name: Run agent snapshot tests
        run: pytest tests/ -v
        env:
          # Optional: enables LLM judge for higher-accuracy semantic comparison
          AGENTSNAP_JUDGE_API_KEY: ${{ secrets.AGENTSNAP_JUDGE_API_KEY }}
```

Snapshots are committed to the repo. CI only runs the asserter — no real agent API calls needed unless your tests explicitly make them.

---

## How to approve an intentional regression

When you intentionally change agent behavior (new prompt, model upgrade, new tool):

```bash
# 1. Run tests — they fail, new trace saved to .last_run/
pytest tests/test_my_agent.py

# 2. Inspect what changed
agentsnap diff __agent_snapshots__/my_agent.json

# 3. Approve — promote last run to golden
agentsnap update my_agent

# 4. Commit the new baseline
git add __agent_snapshots__/my_agent.json
git commit -m "approve: updated golden after Sonnet upgrade"
```

---

## Thresholds

Two independent thresholds control the semantic layer:

| Threshold | Default | Applies to |
|-----------|---------|-----------|
| `semantic_threshold` | `0.92` | Final `output` — the agent's actual answer |
| `llm_threshold` | `0.75` | Intermediate `llm_call[n]` responses — tolerates natural phrasing variance |

Tune per-test:

```python
# Critical factual agent — hold output tightly
with AgentAsserter("rag_agent", semantic_threshold=0.97) as a: ...

# Creative agent — allow more paraphrasing
with AgentAsserter("writer_agent", semantic_threshold=0.75) as a: ...
```
