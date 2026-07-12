# agentsnap

Deterministic snapshot testing for AI agents.

`agentsnap` records your agent's LLM and tool calls during a **golden run** and produces a committed snapshot file. On every subsequent run it replays the same inputs and compares the new trace against the snapshot across four dimensions:

| Dimension | What it checks | How |
|-----------|----------------|-----|
| **Structural** | Tool call names and order | Levenshtein edit distance on the tool sequence |
| **Arguments** | Tool call arguments | `deepdiff` (if installed) or plain dict diff, with configurable ignored fields |
| **Model tools** | Which tool the *model itself* requested (not just what your code executed) | Levenshtein edit distance on `tool_requests`, plus per-request argument diffs |
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

## Install matrix

```bash
pip install agentsnap                  # base install (OpenAI SDK included)
pip install agentsnap[google]          # Google Gemini
pip install agentsnap[cohere]          # Cohere
pip install agentsnap[mistral]         # Mistral
pip install agentsnap[groq]            # Groq
pip install agentsnap[anthropic]       # Anthropic
pip install agentsnap[langgraph]       # LangGraph adapter
pip install agentsnap[offline]         # offline embeddings backend (sentence-transformers)
pip install agentsnap[all-providers]   # every provider SDK + offline embeddings
pip install -e ".[dev]"                # development install (test tooling)
```

---

## Guides

- [Recording](guides/recording.md) — `PatchSet`, adapters, the pytest fixture, and the snapshot file format
- [Replay](guides/replay.md) — replay vs live mode, `raw_response`, re-recording, caveats
- [Streaming](guides/streaming.md) — tee behavior, replayed streams, current limitations
- [Model tools](guides/model-tools.md) — capturing what the model itself decided to call
- [Frameworks](guides/frameworks.md) — Pydantic AI, OpenAI Agents SDK, LangChain, LangGraph, CrewAI
- [Configuration](guides/configuration.md) — `pyproject.toml`, pytest ini options, thresholds, env vars
- [CLI](guides/cli.md) — the full command reference and the approval workflow
- [CI](guides/ci.md) — wiring agentsnap into GitHub Actions

See the [API reference](reference.md) for the full class and exception documentation, and the [changelog](changelog.md) for release history.
