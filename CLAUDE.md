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

# Run demos
python examples/demo_mock.py                        # no API keys needed
python examples/demo_mock.py --snapshot-dir /tmp/s  # custom snapshot dir
python examples/demo_real.py                        # needs API keys in .env
```

## Architecture

The package name is `agentsnap`. The project root directory is named `agenttest` (legacy) — don't confuse the two.

### Data flow

```
AgentRecorder / AgentAsserter
        |
        | sets ContextVar
        v
  TraceAccumulator          (recorder.py — thread-safe, contextvar-based)
        ^
        | .push(event)
        |
  Adapter wrappers          (adapters/*.py — one per provider)
        |
        | forward call to real SDK
        v
  Real LLM / tool
```

Every adapter checks `TraceAccumulator.current()`, forwards the real call, pushes a `{"type": "llm_call"|"tool_call", ...}` event, and returns the response unchanged. Outside a recorder/asserter context the adapter is a transparent pass-through.

### Key design points

**ContextVar isolation** — `_accumulator_var` in `recorder.py` is a module-level `ContextVar`. Each `AgentRecorder`/`AgentAsserter` entry sets a new token and resets on exit. Nested agents and async code each get their own accumulator automatically.

**Diff engine** (`core/diff.py`) — three layers run in order:

1. **Structural** — Levenshtein edit distance on the tool name sequence. Fails fast; skips layer 2 if mismatch. Reports edit distance so a one-tool rename is distinguishable from a full rewrite.

2. **Arguments** — uses `deepdiff` when installed (path-based, type-aware, order-ignoring) and falls back to plain dict diff. Configurable `ignored_fields` list.

3. **Semantic** — two backends:
   - Default: cosine similarity via `all-MiniLM-L6-v2` (offline, lazy-loaded).
   - Optional: `LLMJudge` — calls an LLM to score equivalence and returns a reason string. Enabled by passing `judge=LLMJudge(...)` to `AgentAsserter`, or automatically via `AGENTSNAP_JUDGE_API_KEY` env var in the pytest fixture.

   Two separate thresholds: `semantic_threshold=0.92` for the final `output`, `llm_threshold=0.75` for intermediate `llm_call[n]` responses (which vary naturally between runs).

   `semantic_scores()` returns a `(scores: dict, reasons: dict)` tuple. `compute_diff()` accepts both `embed_fn` (for testing stubs) and `judge` parameters.

**Model tool decisions** — non-streaming Anthropic/OpenAI `llm_call` events also record `tool_requests` (the model's own `tool_use` blocks, `{"name", "args"}`); `model_tool_diffs()` in `core/diff.py` fails `model_tools`/`model_tool_args` when it drifts, gated on both sides of the diff carrying the key so pre-existing snapshots are unaffected.

**Configuration** (`config.py`) — `config.load()` merges: built-in defaults < `[tool.agentsnap]` in the nearest `pyproject.toml` < environment variables. `judge_from_env()` returns a configured `LLMJudge` or `None`. `LLMJudge.from_env()` is the public alias.

**pytest plugin** — registered via `pytest11` entry point. `pytest_addoption()` exposes `agentsnap_*` ini keys. The `snapshot` fixture reads config, builds an `LLMJudge` if `AGENTSNAP_JUDGE_API_KEY` is set, and passes it as the default to `assert_agent()`. Per-test overrides always win. Pass `judge=False` to force embeddings even when a key is set.

**Snapshot files** — `__agent_snapshots__/*.json` are committed (source of truth). `__agent_snapshots__/.last_run/*.json` are written on every assert run; gitignored; used by `agentsnap update` to approve regressions.

**Replay mode** (`core/replay.py`) — `AgentAsserter(mode="replay")` replays recorded `raw_response` payloads through the adapters instead of calling live APIs; the diff flips to request-side comparison (`compare_llm_requests` in `DiffConfig`). Anthropic/OpenAI (plus Groq/OpenRouter via OpenAIAdapter subclass); other adapters raise `ReplayError` in replay mode.

**Adapters** — OpenAI and Anthropic adapters tee `stream=True` calls (`OpenAIRecordingStream` / `AnthropicRecordingStream`): chunks pass through to the caller unmodified while the assembled response is recorded as `raw_response={"__stream__": True, "chunks": [...]}`. Replay rebuilds those chunks into real SDK objects via `replay_stream()`; a streaming recording replayed as a non-streaming request (or vice versa) raises `ReplayError`. Mistral still forces `stream=False` to prevent partial streaming responses from being recorded. Groq and OpenRouter subclass `OpenAIAdapter` directly (OpenAI-compatible interface), so they inherit the tee for free. Not yet supported: the `client.messages.stream()` context-manager helper, and async streams. All new adapters must either force `stream=False` if the SDK supports streaming, or implement an equivalent recording tee. A stream that is never iterated and never closed is finalized automatically at recorder/asserter exit (`TraceAccumulator.finalize_streams()`), but consuming or closing it promptly is still recommended so events appear in call order.

### Adding a new provider adapter

1. Create `agentsnap/adapters/<provider>.py`
2. Check `TraceAccumulator.current()`, force `stream=False` (or implement a recording tee if the SDK supports streaming), forward the call, push `{"type": "llm_call", "messages": [...], "response": str, "tokens": int}`
3. Add optional dep to `pyproject.toml`
4. Add mock client + demo function to `examples/demo_mock.py`

`adapters/groq.py` is the minimal example (one-liner subclass of `OpenAIAdapter`).

### Test conventions

Unit tests (`tests/unit/`) never load `sentence-transformers` or call any API. Pass deterministic stubs to `compute_diff()` / `semantic_scores()`:

- `_identical_embed` — returns same unit vector for all inputs (cosine sim = 1.0)
- `_orthogonal_embed` — returns orthogonal vectors (cosine sim = 0.0)

`semantic_scores()` returns `(scores, reasons)` — unpack accordingly in tests.

Integration tests (`tests/integration/`) use `MockAnthropicClient` / `MockAnthropicResponse` from `tests/fixtures/mock_agents.py`. No real API calls. Also pass `embed_fn` stubs and set `llm_threshold=0.0` when the test explicitly expects any similarity to pass.

### Environment variables

| Variable | Purpose |
|----------|---------|
| `AGENTSNAP_JUDGE_API_KEY` | Explicit key override — always wins |
| `AGENTSNAP_JUDGE_MODEL` | Model override (default: `openai/gpt-4o-mini`) |
| `AGENTSNAP_JUDGE_BASE_URL` | Base URL override (default: OpenRouter) |
| `OPENROUTER_API_KEY` | Auto-used when `judge_base_url` contains `openrouter.ai` |
| `OPENAI_API_KEY` | Auto-used when `judge_base_url` contains `api.openai.com` |
| `ANTHROPIC_API_KEY` | Auto-used when `judge_base_url` contains `anthropic.com` |

Key resolution is in `config._resolve_api_key()`. Add new entries to `_PROVIDER_KEY_MAP` in `config.py` to support additional providers.
