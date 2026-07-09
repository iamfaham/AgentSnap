# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.2.0] - 2026-07-09

### Added

- **Replay mode** — deterministic asserts that replay recorded LLM responses instead of calling a live provider, so CI can run without API keys. Enable per-test with `AgentAsserter(mode="replay")`, per-suite via `[tool.agentsnap] mode = "replay"`, or per-run with the `--agentsnap-replay` / `--agentsnap-live` pytest flags.
- Snapshot format bumped to **v1.1**: every `llm_call` event now stores a `raw_response` alongside the parsed fields, which replay reconstructs into a provider-shaped response object.
- `replay_tools=True` lets `ToolAdapter` return recorded tool results under replay instead of re-invoking the real tool.
- Request-side diffing for replay mode, plus an exact-match short-circuit so identical requests skip semantic scoring entirely.
- New `SnapshotFormatError` and `ReplayError` exception types with clear, actionable messages (e.g. when a snapshot predates v1.1 and can't be replayed).
- `agentsnap diff` now runs the full semantic comparison pipeline (previously only did a raw JSON diff); new `agentsnap show` command replaces the old ad-hoc diff pretty-printer.
- `structural_tolerance` is now configurable via `pyproject.toml`, pytest ini options, or a per-test override, instead of being hardcoded.
- Regression errors print before/after text excerpts for every failing semantic step, not just the final output.
- Scenario namespacing: multiple snapshots per test function, with an automatic input hash (`input_sha8`) when no explicit scenario name is given; `agentsnap update` promotes all scenario variants at once.
- Async support: `async with` on `AgentRecorder`/`AgentAsserter`, and `ainvoke` on the LangGraph adapter.
- Volatile-field normalization (timestamps, token counts, request IDs) before trace comparison, to cut down on noisy diffs.
- MIT `LICENSE` file and a `py.typed` marker for downstream type checkers.
- CI workflow (test matrix across Python 3.10-3.13 on Linux and Windows, plus lint) and a tag-triggered release workflow that publishes to PyPI via Trusted Publishing.

### Fixed

- `ReplayError` is raised cleanly (instead of leaking a lower-level exception) when a recorded response fails to reconstruct.
- LangGraph streaming calls are guarded so they don't break trace capture.
- Replay mode now errors clearly instead of silently falling back when a scenario has no matching recorded snapshot.
- `structural: ok` is reported in `agentsnap diff` output even when the configured tolerance absorbs a change, instead of being omitted.

## [0.1.2] - 2026-07-01

### Added

- `DiffConfig` for structural tolerance and LLM-judge configuration in one place; fixed `LLMJudge` client reuse and reason-key naming.
- Scenario parameter on snapshot paths, plus an `input_sha8` helper for auto-naming scenarios from input content.
- Scenario namespacing, input-based auto-hashing, and a warning when a test's captured input looks unbound.
- `agentsnap update` promotes all scenario variants for a test (wildcard), not just one at a time.

### Changed

- Removed an unused `shutil` import from `test_cli.py`.

## [0.1.1] - 2026-06-29

### Added

- `agentsnap init` and `agentsnap check` CLI commands, backed by a new interactive setup wizard (`setup_wizard` module) that validates the model/provider connection and writes config safely via `tomlkit`.
- `write_config()` for safe, idempotent `pyproject.toml` updates.
- `PatchSet` for zero-instrumentation LLM capture (patch the SDK directly, no adapter wrapping required), plus an `agentsnap_instrument` pytest fixture and `--agentsnap-instrument` flag.
- `--agentsnap-record` pytest flag to force re-recording all snapshots in a run.
- LangGraph adapter capturing node-level LLM/tool events via callbacks, with async (`ainvoke`) support.
- Async context managers on `AgentRecorder`/`AgentAsserter`.
- Before/after text shown directly in regression error output, so failures are readable without a separate diff step.
- Volatile-field normalization (timestamps, token counts) before trace comparison.
- `agentsnap.wrap()` and `snapshot.run()` for a lower-friction, zero-boilerplate end-user API; `AgentAsserter` now auto-records a golden snapshot on first miss instead of raising.
- Author and license metadata added to `pyproject.toml`.

### Fixed

- API keys are auto-resolved from existing provider environment variables for the LLM judge.
- `.env` files are written with mode `0600` to avoid exposing API keys to other users on shared machines.
- `llm_threshold` is auto-lowered when an LLM judge is configured, and is tracked separately from `semantic_threshold`.
- Guarded against an unconfigured embeddings/judge backend, and offline (`sentence-transformers`) dependencies were made optional so a bare install stays lightweight.
- Replaced em dashes with hyphens in terminal output for Windows ASCII-console compatibility.

## [0.1.0] - 2026-05-30

### Added

- Initial release of `agentsnap` (originally prototyped as `agenttest`): a deterministic snapshot-testing harness for AI agents built on a thread-safe, `ContextVar`-based trace accumulator.
- Provider adapters for OpenAI, Anthropic, Google Gemini, Cohere, Mistral, Groq, and OpenRouter (OpenAI-compatible).
- Diff engine combining structural (edit-distance) tool-sequence comparison, argument diffing (`deepdiff`), and semantic comparison via sentence embeddings or an LLM-as-judge.
- pytest plugin with a `snapshot` fixture and `agentsnap_*` ini options.
- `__agent_snapshots__/` as the committed snapshot store, with a gitignored `.last_run/` directory used by `agentsnap update` to approve regressions.
- Example demo scripts (`demo_mock.py`, `demo_real.py`) covering every supported provider.
