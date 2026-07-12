# Configuration

## `pyproject.toml`

```toml
[tool.agentsnap]
judge_model        = "openai/gpt-4o-mini"
judge_base_url     = "https://openrouter.ai/api/v1"
semantic_threshold = 0.92   # final agent output (strict)
llm_threshold      = 0.75   # intermediate LLM responses (tolerant)
mode               = "live"   # "live" (default) or "replay"
```

## pytest ini options

These can also be set as pytest ini options in `pyproject.toml` or `pytest.ini`:

```toml
[tool.pytest.ini_options]
agentsnap_judge_model        = "openai/gpt-4o-mini"
agentsnap_judge_base_url     = "https://openrouter.ai/api/v1"
agentsnap_semantic_threshold = "0.92"
agentsnap_llm_threshold      = "0.75"
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

## Thresholds

Two independent thresholds control the semantic layer:

| Threshold | Default | Applies to |
|-----------|---------|-----------|
| `semantic_threshold` | `0.92` | Final `output` — the agent's actual answer |
| `llm_threshold` | `0.75` | Intermediate `llm_call[n]` responses — tolerates natural phrasing variance |

The lower `llm_threshold` is intentional: LLM phrasing varies even for identical prompts, so intermediate responses are held loosely. The final output is held tightly.

Tune per-test:

```python
# Critical factual agent — hold output tightly
with AgentAsserter("rag_agent", semantic_threshold=0.97) as a: ...

# Creative agent — allow more paraphrasing
with AgentAsserter("writer_agent", semantic_threshold=0.75) as a: ...
```

Tune globally in `pyproject.toml`:

```toml
[tool.agentsnap]
semantic_threshold = 0.95
llm_threshold      = 0.80
```

## `structural_tolerance`

Configurable via `pyproject.toml`, pytest ini options, or a per-test override on `AgentAsserter(structural_tolerance=...)`, instead of being hardcoded.

`structural_tolerance` is an edit-distance budget: the structural check (and the model-tools check) only fails once the Levenshtein distance between the old and new tool-name sequences exceeds this value.

**Dual-role note:** `structural_tolerance` applies to BOTH the executed-tool sequence (the `structural` check) and the model-requested tool sequence (the `model_tools` check described in [Model tools](model-tools.md)). Relaxing it to tolerate flaky tool ordering in your own code also relaxes how much the model itself is allowed to drift in which tool it asks for — there is no separate knob for the two.

## Environment variables

| Variable | Purpose |
|----------|---------|
| `AGENTSNAP_JUDGE_API_KEY` | Explicit key override — always wins |
| `AGENTSNAP_JUDGE_MODEL` | Model override (default: `openai/gpt-4o-mini`) |
| `AGENTSNAP_JUDGE_BASE_URL` | Base URL override (default: OpenRouter) |
| `OPENROUTER_API_KEY` | Auto-used when `judge_base_url` contains `openrouter.ai` |
| `OPENAI_API_KEY` | Auto-used when `judge_base_url` contains `api.openai.com` |
| `ANTHROPIC_API_KEY` | Auto-used when `judge_base_url` contains `anthropic.com` |

## Judge key resolution

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

Key resolution is implemented in `config._resolve_api_key()`; the `_PROVIDER_KEY_MAP` in `agentsnap/config.py` is where new provider entries are added.

See the [API reference](../reference.md) for `LLMJudge` and `DiffConfig`.
