# agentsnap

[![CI](https://github.com/iamfaham/AgentSnap/actions/workflows/ci.yml/badge.svg)](https://github.com/iamfaham/AgentSnap/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/agentsnap)](https://pypi.org/project/agentsnap/)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://pypi.org/project/agentsnap/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Deterministic snapshot testing for AI agents.

`agentsnap` records your agent's LLM and tool calls during a **golden run** and produces a committed snapshot file. On every subsequent run it replays the same inputs and compares the new trace against the snapshot across three dimensions:

| Dimension | What it checks | How |
|-----------|----------------|-----|
| **Structural** | Tool call names and order | Levenshtein edit distance on the tool sequence |
| **Arguments** | Tool call arguments | `deepdiff` (if installed) or plain dict diff, with configurable ignored fields |
| **Semantic** | LLM responses and final output | Cosine similarity via `all-MiniLM-L6-v2`, or an LLM judge for higher accuracy |

If any dimension drifts beyond its threshold, `agentsnap` raises `AgentRegressionError` with a structured diff report.

---

## Why agentsnap

Agents regress silently. A prompt tweak, a model swap, a tool wired to the wrong argument — nothing throws an exception, nothing fails CI, and you find out in production when the agent quietly starts giving worse answers.

`agentsnap` gives you two modes for two different jobs:

- **Replay, on every PR** — recorded responses are replayed instead of calling a real API. Deterministic, zero cost, catches code regressions (prompt edits, broken tool wiring, changed call counts).
- **Live, nightly** — real API calls against the current model, catching drift that only shows up when the model itself changes.

A prompt edit caught by replay mode, no API call required:

```
Agent regression in 'demo_replay'
=================================

[ARGS] llm_call[0].messages:
  messages: [{'content': 'Answer concisely: What is Python?', ...}] ->
            [{'role': 'user', 'content': 'You are a pirate. Answer: ...'}]

[SEMANTIC] llm_call[0]: 100% PASS
[SEMANTIC] output: 100% PASS

Failed checks: ['llm_requests']
```

---

## 3-minute quickstart

### 1 — Install

```bash
pip install agentsnap
```

### 2 — Run setup

```bash
agentsnap init
```

Asks you to choose a semantic comparison backend:

| Option | What it needs | Best for |
|--------|--------------|----------|
| **[1] LLM judge** (default) | API key (OpenRouter, OpenAI, Anthropic, or custom) | Factual agents, highest accuracy |
| **[2] Offline embeddings** | Nothing — ~22 MB model download, runs anywhere | Any machine, no API key |
| **[3] Local LLM judge** | *(coming soon)* | Strong local machine, no cloud |

The wizard saves your choice to `pyproject.toml` and your API key (if any) to `.env`. Keys are never written to `pyproject.toml`. It also adds `__agent_snapshots__/.last_run/` to `.gitignore` (creating the file if needed) and offers to scaffold an example snapshot test at `tests/test_agentsnap_example.py`.

```bash
agentsnap check   # verify your setup at any time
```

### 3 — Record your agent (no code changes needed)

`PatchSet` patches all installed LLM SDKs at the class level — any raw client created anywhere is captured automatically. No need to wrap your clients:

```python
from agentsnap import PatchSet, AgentRecorder
import anthropic

# your existing agent — untouched
def my_agent(question):
    client = anthropic.Anthropic()   # raw client, no wrapper needed
    return client.messages.create(...).content[0].text

# First run: records the golden snapshot
with PatchSet():
    with AgentRecorder("my_agent") as rec:
        result = my_agent("What is Python?")
        rec.output = result
# Writes __agent_snapshots__/my_agent.json — commit this file
```

### 4 — Assert on future runs

```python
from agentsnap import PatchSet, AgentAsserter

with PatchSet():
    with AgentAsserter("my_agent") as a:
        result = my_agent("What is Python?")
        a.output = result
# Raises AgentRegressionError if behavior drifted
```

### 5 — Use the pytest fixture (simplest)

`snapshot.run()` auto-records on first call and auto-asserts on every run after — no switching needed. Add `agentsnap_instrument` to activate `PatchSet` automatically:

```python
def test_my_agent(snapshot, agentsnap_instrument):
    with snapshot.run("my_agent") as s:
        result = my_agent("What is Python?")   # raw client — captured automatically
        s.output = result
```

```bash
pytest
# or enable PatchSet for every test in a session:
pytest --agentsnap-instrument
```

---

## Replay vs live mode

Every assert can run in one of two modes:

| Mode | LLM calls | Catches | Best for |
|------|-----------|---------|----------|
| `live` (default) | Real API | Model/behavior drift | Nightly runs, pre-release |
| `replay` | None — recorded responses are replayed | Code regressions: prompt edits, tool wiring, loop logic | Every PR / CI |

In replay mode the recorded response for each LLM call is fed back to your
agent — no API key, no cost, no flakes. The comparison flips to the **request
side**: agentsnap fails the test if your code sends different prompts, makes a
different number of LLM calls, or changes the tool sequence.

```python
# per test
with AgentAsserter("my_agent", mode="replay") as a: ...

# whole suite
pytest --agentsnap-replay        # force replay
pytest --agentsnap-live          # force live
```

```toml
[tool.agentsnap]
mode = "replay"   # make replay the project default
```

Tool calls still execute for real in replay mode. Pass `replay_tools=True` to
stub them from the recording too (no side effects at all).

Notes:
- Replay needs snapshots recorded with agentsnap >= 0.2.0 (they include
  `raw_response`). Older snapshots raise `SnapshotFormatError` — re-record
  with `pytest --agentsnap-record`.
- Replay currently supports Anthropic, OpenAI, Groq, and OpenRouter.
  Other providers raise `ReplayError` — use live mode for those tests.
- With scenarios, pass `scenario=` explicitly in replay mode (input auto-hash
  is not available because the snapshot is read before the test body runs).
- If the replayed final output isn't byte-identical to the golden, scoring it
  needs a semantic backend — install the embeddings extra
  (`pip install agentsnap[offline]`) or configure a judge
  (`AGENTSNAP_JUDGE_API_KEY`).
- Async clients (`AsyncAnthropic`, `AsyncOpenAI`) aren't intercepted yet —
  replay's no-network guarantee currently covers sync clients only.

### Streaming agents

`AnthropicAdapter` and `OpenAIAdapter` tee `stream=True` calls (Groq and
OpenRouter inherit this from `OpenAIAdapter`): chunks are forwarded to your
agent unmodified while the assembled response is recorded for replay, with
`raw_response={"__stream__": True, "chunks": [...]}`.

In replay mode the recorded chunks are rebuilt into real SDK chunk/event
objects and yielded back incrementally — your agent consumes them exactly
like a live stream, with zero API calls. A recording made from a streaming
call cannot replay as a non-streaming request (or vice versa) — that raises
`ReplayError` with a clear "shape mismatch" message.

Not yet supported: the `client.messages.stream()` context-manager helper,
and async streams. Mistral still forces `stream=False` on every call. See
`examples/demo_streaming.py` for a full runnable walkthrough.

A stream that is never iterated and never closed is finalized automatically
at recorder/asserter exit, but consuming or closing it promptly is still
recommended so events appear in call order.

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
| LangGraph | `LangGraphAdapter` | `.invoke()` + node-level LLM/tool events via callbacks |
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

## Using adapters (alternative)

If you prefer explicit interception, wrap your SDK client with the matching adapter instead of using `PatchSet`. Useful when you want to be explicit about what is captured or when you control the agent code directly:

```python
from agentsnap import AgentRecorder
from agentsnap.adapters.anthropic import AnthropicAdapter

client = AnthropicAdapter(anthropic.Anthropic())   # explicit wrapper

with AgentRecorder("my_agent") as rec:
    result = my_agent(client, "What is Python?")
    rec.output = result
```

> **Note:** Do not combine adapters with `PatchSet` on the same client — both interceptors will fire and events will be recorded twice.

---

## Configuration

### API key for the LLM judge (optional)

The LLM judge uses a small language model to compare outputs instead of embeddings — more accurate for factual content.

agentsnap resolves the API key automatically — **you do not need a separate key**. It checks in this order:

1. `AGENTSNAP_JUDGE_API_KEY` — explicit override, always wins
2. The provider-specific key that matches `judge_base_url`:

| `judge_base_url` contains | Key used automatically |
|--------------------------|------------------------|
| `openrouter.ai` (default) | `OPENROUTER_API_KEY` |
| `api.openai.com` | `OPENAI_API_KEY` |
| `anthropic.com` | `ANTHROPIC_API_KEY` |
| `api.groq.com` | `GROQ_API_KEY` |
| `api.mistral.ai` | `MISTRAL_API_KEY` |
| `api.cohere.com` | `COHERE_API_KEY` |

Once any matching key is found, the `snapshot` pytest fixture enables the LLM judge automatically — no code changes needed in tests.

To use a different provider, change `judge_base_url` in `pyproject.toml` and set the matching env var:

```bash
export OPENAI_API_KEY=sk-...
```
```toml
[tool.agentsnap]
judge_base_url = "https://api.openai.com/v1"
judge_model    = "gpt-4o-mini"
```

### Project settings (`pyproject.toml`)

```toml
[tool.agentsnap]
judge_model        = "openai/gpt-4o-mini"
judge_base_url     = "https://openrouter.ai/api/v1"
semantic_threshold = 0.92   # final agent output (strict)
llm_threshold      = 0.75   # intermediate LLM responses (tolerant)
mode               = "live"   # "live" (default) or "replay"
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

Context manager. Reads the snapshot, intercepts calls, runs the three-layer diff on exit. If no snapshot exists yet, automatically switches to record mode and writes the golden run.

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

### `PatchSet`

Context manager that monkey-patches all installed LLM SDK classes so any client — wrapped or unwrapped — is captured by an active `AgentRecorder` or `AgentAsserter`.

```python
from agentsnap import PatchSet

with PatchSet():
    # all anthropic.Anthropic(), openai.OpenAI(), etc. clients are auto-captured
    ...
```

### `LLMJudge(api_key, model, base_url)`

Uses an LLM to score semantic equivalence instead of embeddings. Returns a `0.0–1.0` score and a one-sentence reason explaining any difference.

```python
from agentsnap import LLMJudge

judge = LLMJudge(api_key="sk-or-...", model="openai/gpt-4o-mini")
judge = LLMJudge.from_env()   # returns None if no key found

with AgentAsserter("name", judge=judge) as a:
    ...
```

### `snapshot` pytest fixture

Auto-wired from `[tool.agentsnap]` and environment variables. No imports needed.

```python
def test_agent(snapshot):
    # Auto mode: records first time, asserts every run after
    with snapshot.run("name") as s:
        s.output = run_agent(...)

    # Explicit record
    with snapshot.record_agent("name") as rec:
        rec.output = run_agent(...)

    # Explicit assert
    with snapshot.assert_agent("name") as a:
        a.output = run_agent(...)

    # Per-test overrides
    with snapshot.assert_agent("name", judge=False) as a:          # force embeddings
        a.output = run_agent(...)
    with snapshot.assert_agent("name", semantic_threshold=0.98) as a:
        a.output = run_agent(...)
```

### Pytest flags

| Flag | Description |
|------|-------------|
| `--agentsnap-record` | Force re-record all snapshots, overwriting existing goldens |
| `--agentsnap-instrument` | Auto-patch all installed LLM SDKs (zero-instrumentation mode) |
| `--agentsnap-replay` | Force replay mode for every test in the session |
| `--agentsnap-live` | Force live mode for every test in the session |

```bash
pytest --agentsnap-record        # re-record everything
pytest --agentsnap-instrument    # capture raw clients without adapters
pytest --agentsnap-replay        # force replay mode
pytest --agentsnap-live          # force live mode
```

### `agentsnap_instrument` fixture

Standalone fixture for zero-instrumentation capture within a single test:

```python
def test_agent(snapshot, agentsnap_instrument):
    with snapshot.run("name") as s:
        client = anthropic.Anthropic()   # no adapter needed
        s.output = my_agent(client, "query")
```

### Exceptions

| Exception | When raised |
|-----------|-------------|
| `AgentRegressionError(message, diff_report)` | Behavior drifted beyond threshold |
| `SnapshotNotFoundError(test_name)` | No snapshot found (only from direct SDK use; `AgentAsserter` auto-records instead) |
| `AdapterNotWrappedError` | Unwrapped client used inside a recording context without `PatchSet` |

`AgentRegressionError.diff_report` is a `DiffReport` dataclass with `structural_diff`, `argument_diffs`, `semantic_scores`, `semantic_reasons`, and `failed_checks`.

---

## CLI

```bash
agentsnap init                                     # interactive setup wizard — choose backend and save config
agentsnap check                                    # verify current backend is working (exits 0/1)
agentsnap list                                     # list all snapshots
agentsnap status                                   # pass/fail/stale status for every snapshot (CI-friendly, exits 0/1)
agentsnap diff __agent_snapshots__/my_agent.json   # pretty-print a snapshot
agentsnap update my_agent                          # show diff and approve last run as new golden
agentsnap update my_agent --yes                    # approve without confirmation prompt
agentsnap update --all                             # batch-approve every failing or new snapshot
```

---

## Snapshot format

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

Golden snapshots live in `__agent_snapshots__/` and are committed to git. The `.last_run/` subdirectory is written on every assert run and is gitignored — it is only used by `agentsnap update`.

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

# 2. Approve — shows a diff and prompts for confirmation
agentsnap update my_agent

# 3. Commit the new baseline
git add __agent_snapshots__/my_agent.json
git commit -m "approve: updated golden after Sonnet upgrade"
```

For multiple failures at once, run `agentsnap status` to see what changed, then `agentsnap update --all` to batch-approve every failing or new snapshot in one pass.

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
