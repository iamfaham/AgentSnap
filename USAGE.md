# Using agentsnap

This guide walks through everything you need to use agentsnap effectively — from your first snapshot to CI integration and approving intentional changes.

---

## Table of contents

1. [How it works](#how-it-works)
2. [Installation](#installation)
3. [Setup](#setup)
4. [Your first snapshot](#your-first-snapshot)
5. [Choosing an instrumentation style](#choosing-an-instrumentation-style)
6. [Zero-instrumentation with PatchSet](#zero-instrumentation-with-patchset)
7. [Using adapters (alternative)](#using-adapters-alternative)
8. [pytest integration](#pytest-integration)
9. [LangGraph](#langgraph)
10. [Reviewing and approving changes](#reviewing-and-approving-changes)
11. [Configuration](#configuration)
12. [Understanding diff results](#understanding-diff-results)
13. [Tuning thresholds](#tuning-thresholds)
14. [LLM judge](#llm-judge)
15. [CI setup](#ci-setup)

---

## How it works

agentsnap works in two phases:

**Record** — Run your agent once with agentsnap active. It intercepts every LLM call and tool call, records the inputs and outputs into a JSON snapshot file, and saves it alongside your code. Commit this file — it is the contract for what the agent does.

**Assert** — On every subsequent run (in CI, in tests, locally), agentsnap replays the same execution and compares the new trace against the committed snapshot across three layers:

1. **Structural** — Did the tool sequence change? (e.g. added a step, skipped a step, reordered)
2. **Arguments** — Did the arguments to tool calls change?
3. **Semantic** — Did the LLM responses or final output drift in meaning?

If anything drifts beyond its threshold, agentsnap raises `AgentRegressionError` with a detailed report of exactly what changed.

---

## Installation

```bash
pip install agentsnap
```

With optional provider SDKs:

```bash
pip install agentsnap[google]          # Google Gemini
pip install agentsnap[cohere]          # Cohere
pip install agentsnap[mistral]         # Mistral
pip install agentsnap[groq]            # Groq
pip install agentsnap[all-providers]   # everything
```

For development (includes test dependencies):

```bash
pip install -e ".[dev]"
```

---

## Setup

After installing, run the setup wizard to choose your semantic comparison backend:

```bash
agentsnap init
```

The wizard presents three options:

**[1] LLM judge — API (recommended, default)**
Calls a small LLM to score whether two responses are semantically equivalent. More accurate for factual agents. Requires an API key from one of:
- OpenRouter (recommended — one key gives access to many models)
- OpenAI, Anthropic, or any OpenAI-compatible provider

The wizard saves your key to `.env` — never to `pyproject.toml` (which gets committed to git).

**[2] Offline embeddings — all-MiniLM-L6-v2**
Uses cosine similarity between sentence embeddings. No API key, no internet after first use. The 22 MB model downloads once and is cached permanently. Runs on any machine including budget cloud VMs (CPU only, ~500 MB RAM).

Requires `sentence-transformers` to be installed. Choose this if you have no API key or need fully air-gapped operation.

**[3] Local LLM judge — coming soon**
Run the judge on your own machine using a locally hosted model (Ollama, llama.cpp, or any OpenAI-compatible local server). This option is visible in the menu but not yet selectable — it will be available in a future release.

---

Verify your setup at any time:

```bash
agentsnap check
```

Output example (LLM judge):
```
Backend : LLM judge
Provider: https://openrouter.ai/api/v1
Model   : openai/gpt-4o-mini
API key : found
Status  : ok (0.42s)
```

Output example (offline embeddings):
```
Backend : offline embeddings (all-MiniLM-L6-v2)
Model   : cached at ~/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2
Status  : ok
```

`agentsnap check` exits 0 on success and 1 on failure, making it safe to use in CI health checks.

---

## Your first snapshot

The simplest path uses the `snapshot` pytest fixture's `run()` method. It auto-records on the first call and auto-asserts on every run after — no manual switching between record and assert mode:

```python
# test_my_agent.py
def test_my_agent(snapshot):
    with snapshot.run("my_agent") as s:
        result = my_agent(client, "What is Python?")
        s.output = result
```

Run once:
```bash
pytest test_my_agent.py
# -> Records __agent_snapshots__/my_agent.json
```

Commit the snapshot:
```bash
git add __agent_snapshots__/my_agent.json
git commit -m "feat: add golden snapshot for my_agent"
```

From now on `pytest` asserts the snapshot on every run. If behavior drifts, the test fails with a clear report.

---

## Choosing an instrumentation style

agentsnap offers two ways to intercept LLM calls:

**PatchSet (recommended)** — Patch SDK classes at the Python level. Any client created anywhere in the process is captured automatically. No code changes to existing agent code.

**Adapters** — Wrap your SDK client explicitly. agentsnap intercepts calls through the wrapper. Use when you want explicit control over exactly what is captured.

Use `PatchSet` by default. Switch to adapters only when you need to be explicit about which clients are captured or when you are working with a provider that does not yet have a `PatchSet` patch.

---

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

> **Warning:** Do not use `PatchSet` together with an adapter wrapper on the same client. Both interceptors will fire and events will be recorded twice, producing a corrupted trace. Use one or the other.

---

## Using adapters (alternative)

Each supported provider has an adapter that wraps the SDK client explicitly. Use it everywhere you would use the raw client:

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

---

## pytest integration

### The `snapshot` fixture

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

### Per-test overrides

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

### Pytest flags

```bash
# Force re-record all snapshots (overwrites existing goldens)
pytest --agentsnap-record

# Zero-instrumentation: patch all SDK clients automatically
pytest --agentsnap-instrument
```

### Snapshot directory

By default agentsnap places snapshots in `__agent_snapshots__/` relative to the nearest `conftest.py`. To use a different location:

```python
with AgentRecorder("name", snapshot_dir="/path/to/snapshots") as rec:
    ...
```

---

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

---

## Reviewing and approving changes

When a test fails because agent behavior changed intentionally (new prompt, model upgrade, new tool added), use the CLI to approve the change:

```bash
# 1. Run tests — they fail, the new trace is saved to .last_run/
pytest tests/test_my_agent.py

# 2. Review what changed
agentsnap update my_agent
# -> Shows a diff: output changes, tool sequence changes, model changes
# -> Prompts: "Accept this as the new golden? [y/N]"

# 3. Or approve immediately without the prompt
agentsnap update my_agent --yes

# 4. Commit the new golden
git add __agent_snapshots__/my_agent.json
git commit -m "approve: updated golden after model upgrade"
```

Other useful CLI commands:

```bash
agentsnap list                                     # list all snapshot files
agentsnap diff __agent_snapshots__/my_agent.json   # pretty-print a snapshot
```

---

## Configuration

### `pyproject.toml`

```toml
[tool.agentsnap]
judge_model        = "openai/gpt-4o-mini"          # model used for LLM judge
judge_base_url     = "https://openrouter.ai/api/v1" # provider for LLM judge
semantic_threshold = 0.92                           # output similarity threshold
llm_threshold      = 0.75                           # intermediate LLM similarity threshold
```

### Environment variables

| Variable | Purpose |
|----------|---------|
| `AGENTSNAP_JUDGE_API_KEY` | API key for the LLM judge — always takes priority |
| `AGENTSNAP_JUDGE_MODEL` | Model override |
| `AGENTSNAP_JUDGE_BASE_URL` | Base URL override |
| `OPENROUTER_API_KEY` | Auto-used when `judge_base_url` is the default (openrouter.ai) |
| `OPENAI_API_KEY` | Auto-used when `judge_base_url` points to api.openai.com |
| `ANTHROPIC_API_KEY` | Auto-used when `judge_base_url` points to anthropic.com |

---

## Understanding diff results

When `AgentRegressionError` is raised, `str(e)` (or just letting pytest print it) gives a formatted report with percentage scores:

```
Agent regression detected in 'my_agent'

-- Diff Report ------------------------------------------
  [STRUCTURAL] 50% tool match  (Tool sequence changed: ['lookup'] -> ['lookup', 'summarize'])
  [SEMANTIC] llm_call[0]: 91% (PASS)
  [SEMANTIC] output: 71% (FAIL)  "responses differ in topic"
  Failed checks: ['structural', 'semantic:output']
---------------------------------------------------------
```

For programmatic access, inspect `e.diff_report` directly:

```python
try:
    with AgentAsserter("my_agent") as a:
        a.output = my_agent(...)
except AgentRegressionError as e:
    print(e.diff_report.failed_checks)      # ['structural', 'semantic:output']
    print(e.diff_report.structural_diff)    # tool sequence diff string
    print(e.diff_report.argument_diffs)     # per-tool argument diffs
    print(e.diff_report.semantic_scores)    # {'output': 0.71, 'llm_call[0]': 0.91}
    print(e.diff_report.semantic_reasons)   # LLM judge explanations (if enabled)
```

`failed_checks` contains strings like `'structural'`, `'semantic:output'`, or `'semantic:llm_call[0]'` — one entry per layer and step that failed.

---

## Tuning thresholds

Two independent thresholds control the semantic layer:

| Threshold | Default | Applies to |
|-----------|---------|-----------|
| `semantic_threshold` | `0.92` | Final `output` — what the agent returned |
| `llm_threshold` | `0.75` | Intermediate `llm_call[n]` responses — naturally vary between runs |

The lower `llm_threshold` is intentional: LLM phrasing varies even for identical prompts, so intermediate responses are held loosely. The final output is held tightly.

**Tune per-test:**

```python
# Factual RAG agent — output must match closely
with AgentAsserter("rag_agent", semantic_threshold=0.97) as a: ...

# Creative writing agent — allow paraphrasing
with AgentAsserter("writer", semantic_threshold=0.75) as a: ...

# Tighten intermediate LLM check
with AgentAsserter("strict_agent", llm_threshold=0.85) as a: ...
```

**Tune globally in `pyproject.toml`:**

```toml
[tool.agentsnap]
semantic_threshold = 0.95
llm_threshold      = 0.80
```

---

## LLM judge

agentsnap requires a configured backend before it can compare responses. Run `agentsnap init` once per project to choose between LLM judge and offline embeddings. The wizard defaults to the LLM judge because it gives more accurate results for factual agents — choose offline embeddings if you have no API key or need air-gapped operation.

The easiest way to configure the LLM judge is via the setup wizard:

```bash
agentsnap init   # choose [1] LLM judge and enter your API key
agentsnap check  # verify the connection is working
```

You can also configure it manually:

```python
from agentsnap import LLMJudge

# Explicit
judge = LLMJudge(api_key="sk-or-...", model="openai/gpt-4o-mini")

# From environment
judge = LLMJudge.from_env()   # None if no key is set

with AgentAsserter("my_agent", judge=judge) as a:
    a.output = my_agent(...)
```

In pytest, if any matching API key is found in the environment, the judge is enabled automatically for all `snapshot.assert_agent()` and `snapshot.run()` calls. Override per-test with `judge=False` to force embeddings:

```python
def test_agent(snapshot):
    with snapshot.assert_agent("name", judge=False) as a:   # always use embeddings
        a.output = my_agent(...)
```

---

## CI setup

Commit your `__agent_snapshots__/` directory. CI only runs the asserter — no real LLM calls required unless your agent code itself makes them.

```yaml
# .github/workflows/test.yml
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
      - run: pip install -e ".[dev]"
      - run: pytest tests/ -v
        env:
          AGENTSNAP_JUDGE_API_KEY: ${{ secrets.AGENTSNAP_JUDGE_API_KEY }}
```

If `AGENTSNAP_JUDGE_API_KEY` is not set, agentsnap uses offline embedding comparison — provided you ran `agentsnap init` with option [2] (offline embeddings) and committed the resulting `pyproject.toml`. CI works without any secrets once that setup is done.
