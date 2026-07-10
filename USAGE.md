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
8. [Replay vs live mode](#replay-vs-live-mode)
9. [pytest integration](#pytest-integration)
10. [LangGraph](#langgraph)
11. [Reviewing and approving changes](#reviewing-and-approving-changes)
12. [Configuration](#configuration)
13. [Understanding diff results](#understanding-diff-results)
14. [Tuning thresholds](#tuning-thresholds)
15. [LLM judge](#llm-judge)
16. [CI setup](#ci-setup)

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

After you make your backend choice, `agentsnap init` also scaffolds your project: it adds `__agent_snapshots__/.last_run/` to `.gitignore` (creating the file if it doesn't exist, idempotent on repeat runs), and offers to write an example snapshot test to `tests/test_agentsnap_example.py` (opt-in, skipped by default so pytest stays green until you replace the fake agent).

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

## Replay vs live mode

Every assert can run in one of two modes:

| Mode | LLM calls | Catches | Best for |
|------|-----------|---------|----------|
| `live` (default) | Real API | Model/behavior drift | Nightly runs, pre-release |
| `replay` | None — recorded responses are replayed | Code regressions: prompt edits, tool wiring, loop logic | Every PR / CI |

In replay mode the recorded response for each LLM call is fed back to your agent — no API key, no cost, no flakes. The comparison flips to the **request side**: agentsnap fails the test if your code sends different prompts, makes a different number of LLM calls, or changes the tool sequence.

```python
from agentsnap import AgentAsserter

# per test
with AgentAsserter("my_agent", mode="replay") as a:
    a.output = my_agent(client, search_tool, "What is Python?")
```

```bash
# whole suite
pytest --agentsnap-replay        # force replay
pytest --agentsnap-live          # force live
```

```toml
[tool.agentsnap]
mode = "replay"   # make replay the project default
```

Tool calls still execute for real in replay mode. Pass `replay_tools=True` to stub them from the recording too (no side effects at all):

```python
with AgentAsserter("my_agent", mode="replay", replay_tools=True) as a:
    a.output = my_agent(client, search_tool, "What is Python?")
```

Notes:
- Replay needs snapshots recorded with agentsnap >= 0.2.0 (they include `raw_response`). Older snapshots raise `SnapshotFormatError` — re-record with `pytest --agentsnap-record`.
- Replay currently supports Anthropic, OpenAI, Groq, and OpenRouter. Other providers raise `ReplayError` — use live mode for those tests.
- With scenarios, pass `scenario=` explicitly in replay mode (input auto-hash is not available because the snapshot is read before the test body runs).
- If the replayed final output isn't byte-identical to the golden, scoring it needs a semantic backend — install the embeddings extra (`pip install agentsnap[offline]`) or configure a judge (`AGENTSNAP_JUDGE_API_KEY`).
- Async clients (`AsyncAnthropic`, `AsyncOpenAI`) aren't intercepted yet — replay's no-network guarantee currently covers sync clients only.

See `examples/demo_replay.py` for a full runnable walkthrough: record a golden run, replay it with the network disabled to prove zero live calls, then watch replay catch a prompt edit instantly.

### Streaming agents

`AnthropicAdapter` and `OpenAIAdapter` tee `stream=True` calls instead of forcing non-streaming (Groq and OpenRouter inherit this since they subclass `OpenAIAdapter`). Chunks flow through to your agent unmodified while the assembled text/tokens are recorded, with `raw_response={"__stream__": True, "chunks": [...]}`.

Replay rebuilds the recorded chunks into real SDK chunk/event objects and yields them back incrementally — the agent consumes them exactly like a live stream, with zero API calls. Replaying a streaming recording against a non-streaming request (or vice versa) raises `ReplayError` with a "shape mismatch" message.

Not yet supported: the `client.messages.stream()` context-manager helper, and async streams. Mistral still forces `stream=False` on every call.

A stream that is never iterated and never closed is finalized automatically at recorder/asserter exit, but consuming or closing it promptly is still recommended so events appear in call order.

See `examples/demo_streaming.py` for a full runnable walkthrough of recording and replaying a streaming agent.

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

# Force replay mode for every test in the session
pytest --agentsnap-replay

# Force live mode for every test in the session
pytest --agentsnap-live
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

For multiple failures at once, run `agentsnap status` first to see what changed across every snapshot, then `agentsnap update --all` to batch-approve every failing or new snapshot in one pass (shows a diff per file, then asks for one confirmation for the whole batch, unless `--yes` is passed).

Other useful CLI commands:

```bash
agentsnap list                                     # list all snapshot files
agentsnap status                                   # pass/fail/stale status for every snapshot (CI-friendly, exits 0/1)
agentsnap diff __agent_snapshots__/my_agent.json   # pretty-print a snapshot
agentsnap update --all                             # batch-approve every failing or new snapshot
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
mode               = "live"                         # "live" (default) or "replay"
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
