# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install (editable, with dev tools)
pip install -e ".[dev]"

# Install with specific provider SDKs
pip install -e ".[google,cohere,mistral,groq]"

# Run all tests
python -m pytest tests/

# Run a single test
python -m pytest tests/unit/test_diff.py::test_structural_catches_reordering

# Run only unit or integration tests
python -m pytest tests/unit/
python -m pytest tests/integration/

# Lint
ruff check agentsnap/

# CLI
python -m agentsnap.cli list
python -m agentsnap.cli diff __agent_snapshots__/<name>.json
python -m agentsnap.cli update <test_name>

# Run demo (no API keys needed)
python examples/demo_mock.py
python examples/demo_mock.py --snapshot-dir /tmp/snaps
```

## Architecture

The package name is `agentsnap`. The project root is also named `agenttest` (legacy directory name) — don't confuse the two.

### Data flow

```
AgentRecorder / AgentAsserter
        |
        | sets ContextVar
        v
  TraceAccumulator          (recorder.py — thread-safe, contextvar-based singleton)
        ^
        | .push(event)
        |
  Adapter wrappers          (adapters/*.py — one per provider)
        |
        | forward call to real SDK
        v
  Real LLM / tool
```

Every adapter follows the same pattern: check `TraceAccumulator.current()`, forward the real call, push a `{"type": "llm_call"|"tool_call", ...}` event, return the response unchanged. If no accumulator is active (outside a recorder/asserter context), the adapter is a transparent pass-through.

### Key design points

**ContextVar isolation** — `_accumulator_var` in `recorder.py` is a module-level `ContextVar`. Each `AgentRecorder`/`AgentAsserter` entry pushes a new accumulator and resets to the previous on exit. This means nested agents and async code each get their own accumulator automatically.

**Diff engine** (`core/diff.py`) — three layers run in order:
1. Structural: tool call names+order must match exactly. If this fails, layer 2 is skipped entirely.
2. Arguments: JSON diff per tool call, with configurable `ignored_fields`.
3. Semantic: cosine similarity via `all-MiniLM-L6-v2` (lazy-loaded on first use). Both LLM call responses and the final `output` string are compared. Default threshold: 0.92.

`compute_diff()` accepts an `embed_fn` parameter — pass a stub in tests to avoid loading the model.

**Snapshot files** (`__agent_snapshots__/*.json`) are the committed source of truth. `__agent_snapshots__/.last_run/*.json` is written on every assert run and is the source for `agentsnap update`. The `.last_run/` directory should be gitignored in production use.

**pytest plugin** — registered via the `pytest11` entry point in `pyproject.toml`. The `snapshot` fixture auto-discovers `__agent_snapshots__/` by walking up from the test file to the nearest `conftest.py`.

### Adding a new provider adapter

1. Create `agentsnap/adapters/<provider>.py`
2. Wrap the method that makes LLM calls — check `TraceAccumulator.current()`, push `{"type": "llm_call", "messages": [...], "response": str, "tokens": int}` after the real call
3. Add optional dep to `pyproject.toml`
4. Add a mock client class to `tests/fixtures/mock_agents.py` and a demo function to `examples/demo_mock.py`

Groq (`adapters/groq.py`) is the minimal example — it's a one-liner subclass of `OpenAIAdapter` because Groq's SDK is OpenAI-compatible.

### Test conventions

Unit tests (`tests/unit/`) never load `sentence-transformers`. They pass a deterministic `embed_fn` stub directly to `compute_diff()` / `semantic_scores()`.

Integration tests (`tests/integration/`) use `MockAnthropicClient` / `MockAnthropicResponse` from `tests/fixtures/mock_agents.py` — no real API calls. They also pass `embed_fn` stubs.

`_identical_embed` (cosine sim = 1.0) and `_orthogonal_embed` (cosine sim = 0.0) are the two standard stubs used across test files.
