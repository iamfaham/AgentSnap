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

3. Add a mock client and a demo function to `examples/demo_mock.py` so the adapter is exercised without real API calls.

4. `agentsnap/adapters/groq.py` is the minimal example — a one-line subclass of `OpenAIAdapter` for OpenAI-compatible providers. Use it as a template when the new provider's SDK shares OpenAI's interface.

## Commit style

This repo uses [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` a new feature
- `fix:` a bug fix
- `docs:` documentation only
- `chore:` maintenance, deps, tooling
- `ci:` CI/CD configuration

Keep the subject line short and in the imperative mood (e.g. `fix: guard trailing newline in _write_env_key`).

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
