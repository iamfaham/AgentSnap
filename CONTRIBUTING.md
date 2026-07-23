# Contributing to agentsnap

Thanks for considering a contribution. This project is small and the workflow is intentionally simple.

## Development setup

```bash
pip install -e ".[dev]"
```

For working on a specific provider adapter, also install its SDK, e.g.:

```bash
pip install -e ".[google,cohere,mistral,groq]"
```

## Running tests

```bash
python -m pytest tests/
```

- `tests/unit/` — no API keys needed, no network calls, no `sentence-transformers` load. Unit tests pass deterministic embedding stubs into `compute_diff()` / `semantic_scores()`.
- `tests/integration/` — also no real API calls; they use mock clients from `tests/fixtures/mock_agents.py`.

Run a single test:

```bash
python -m pytest tests/unit/test_diff.py::test_structural_catches_reordering
```

### Snapshot files

`__agent_snapshots__/*.json` are the source of truth for snapshot tests and are committed to the repo. `__agent_snapshots__/.last_run/*.json` are written on every assert run and are gitignored — they exist so `agentsnap update` can show you a diff and let you approve a regression as the new golden snapshot.

## Lint

```bash
ruff check agentsnap/
```

Please run this before opening a PR — CI enforces it.

## Adding a provider adapter

1. Create `agentsnap/adapters/<provider>.py`. It should check `TraceAccumulator.current()`, force `stream=False` if the underlying SDK supports streaming, forward the call unchanged to the real SDK, and push an event shaped like:

   ```python
   {
       "type": "llm_call",
       "messages": [...],
       "response": str,
       "tokens": int,
       "raw_response": dump,  # full provider response, needed for replay mode
   }
   ```

   Outside a recorder/asserter context the adapter must be a transparent pass-through.

2. Add the optional dependency to `pyproject.toml` under `[project.optional-dependencies]`.

3. Add a mock client and an adapter story to `examples/providers.py` so the adapter is exercised without real API calls (that's where Gemini/Cohere/Mistral/Groq already live).

4. `agentsnap/adapters/groq.py` is the minimal example — a one-line subclass of `OpenAIAdapter` for OpenAI-compatible providers. Use it as a template when the new provider's SDK shares OpenAI's interface.

## Commit style

This repo uses [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` a new feature
- `fix:` a bug fix
- `docs:` documentation only
- `chore:` maintenance, deps, tooling
- `ci:` CI/CD configuration

Keep the subject line short and in the imperative mood (e.g. `fix: guard trailing newline in _write_env_key`).

## Dogfooding: live API validation

agentsnap validates itself against real provider APIs, not just mocks. Two GitHub Actions workflows do this in CI:

- **`.github/workflows/live-validation.yml`** — runs `python examples/run_all.py --real` (the same command described in `examples/README.md`) against whichever provider secrets are configured. Absent secrets degrade gracefully: each example prints a skip hint for that provider and exits 0, so a partial (or empty) key set never fails the job.
- **`.github/workflows/sdk-drift.yml`** — installs the latest, unpinned provider SDKs and runs the hermetic test suite (`python -m pytest tests/ -q`, no keys, no network) to catch upstream SDK changes that break our interception/reconstruction code, independent of live-API behavior.

**Running it manually:** Actions tab → `live-validation` → *Run workflow*. Optionally set the `only` input to a comma-separated subset of example names (e.g. `quickstart,replay`) to forward to `run_all.py --only`; leave it blank to run every example. Manual dispatch always runs, regardless of the kill-switch below.

**Arming/disarming the monthly scheduled runs:** both workflows also run on a monthly cron (`live-validation` at 06:00 UTC and `sdk-drift` at 07:00 UTC on the 1st), but the cron is a no-op unless a repo variable is set. Go to Settings → Secrets and variables → Actions → Variables and set `RUN_SCHEDULED_LIVE_TESTS` to `true` to arm the scheduled runs, or delete/unset it (or set anything other than `true`) to disarm them. This one variable gates both scheduled jobs.

**Secrets to configure** (Settings → Secrets and variables → Actions → Secrets) — set only the ones you have; the rest cause that provider's examples to skip:

| Secret | Used for |
|--------|----------|
| `ANTHROPIC_API_KEY` | Anthropic examples |
| `OPENAI_API_KEY` | OpenAI examples |
| `OPENROUTER_API_KEY` | OpenRouter-routed examples |
| `GEMINI_API_KEY` / `GOOGLE_API_KEY` | Gemini provider example |
| `COHERE_API_KEY` | Cohere provider example |
| `MISTRAL_API_KEY` | Mistral provider example |
| `GROQ_API_KEY` | Groq provider example |
| `AGENTSNAP_JUDGE_API_KEY` | LLM-judge segment of `tuning.py` |

Neither workflow blocks PR merges or the main `test`/`frameworks` CI jobs — both are manual/scheduled only, never on `push`/`pull_request`.

### SDK drift job

A red `sdk-drift` run means a provider SDK shipped a change that broke our interception or response-reconstruction code — this is informational, not a merge gate. To triage: compare the versions printed in the "Print resolved SDK versions" step against the pinned versions in `uv.lock`, then update the relevant adapter (`agentsnap/adapters/*.py`), patch (`agentsnap/patches.py`), or reconstruction (`extract_responses_text`/`reconstruct_response` in `agentsnap/adapters/openai.py`, etc.) code to match the new SDK shape.

## Release process

1. Bump the version in `pyproject.toml`.
2. Add a new entry at the top of `CHANGELOG.md` describing the user-facing changes.
3. Commit those changes.
4. Tag the commit: `git tag vX.Y.Z`.
5. Push the tag: `git push origin vX.Y.Z`.
6. The `release.yml` workflow builds the sdist/wheel and publishes to PyPI via Trusted Publishing — no manual `twine upload` needed.

### One-time PyPI setup

Before the first tagged release, a maintainer needs to register a Trusted Publisher on pypi.org for this project:

- Owner: `iamfaham`
- Repository: `AgentSnap`
- Workflow: `release.yml`
- Environment: `pypi`

This only needs to be done once; after that, every `vX.Y.Z` tag push publishes automatically.
