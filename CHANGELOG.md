# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.4.0] - 2026-07-11

### Added

- **Async client interception** — `PatchSet` now patches `AsyncMessages.create` (Anthropic) and `AsyncCompletions.create` (OpenAI) the same way it patches the sync classes: recording, stream teeing (`AsyncAnthropicRecordingStream` / `AsyncOpenAIRecordingStream`), and full replay (including async streams reconstructed as real SDK chunk objects). Replay's no-network guarantee now covers async clients, not just sync.
- **OpenAI Responses API support** — non-streaming `responses.create()` / `AsyncResponses.create()` calls are recorded and replayed (`tool_requests`, `raw_response`, token counts included). Streamed Responses-API calls remain the one documented capture hole.
- **Real-framework verification** — `tests/frameworks/` exercises Pydantic AI, the OpenAI Agents SDK, and LangChain against agentsnap's `PatchSet` interception through an offline mock HTTP transport, run by a dedicated `frameworks` CI job (`pip install -e ".[dev,frameworks]"`); the main test matrix stays hermetic via `pytest.importorskip`.
- `examples/demo_async.py`: a runnable walkthrough of recording an async agent, replaying it with zero live calls, and catching a prompt edit.
- README/USAGE "Works with your framework" compatibility matrix.

### Fixed

- LangChain's raw-response wrapper was recording empty responses; the OpenAI adapter now unwraps it before extracting text.

## [0.3.0] - 2026-07-11

### Added

- **Model tool-decision capture** — non-streaming Anthropic and OpenAI `llm_call` events now record `tool_requests`: the `tool_use`/`tool_calls` blocks the model itself returned (`{"name": ..., "args": {...}}`), independent of whichever tools your code actually executed. Groq and OpenRouter get this for free via `OpenAIAdapter` inheritance.
- The diff engine compares `tool_requests` across runs and fails `model_tools` (Levenshtein edit distance on the model's requested tool sequence) or `model_tool_args` (per-request argument drift), reported as `[MODEL TOOLS] ...` in `AgentRegressionError`. The check is gated on both the old and new trace carrying `tool_requests`, so snapshots recorded before this feature — and streamed events, which don't assemble it yet — are unaffected.
- `examples/demo_tool_use.py`: a runnable walkthrough of a model swapping its requested tool (`search` -> `delete_file`) getting caught even though the code's own tool sequence and final output are unchanged.
- `snapshot.record_agent()` (explicit record mode) now feeds the same "agentsnap snapshots" terminal summary as `run()`/`assert_agent()`, instead of being invisible to it.

### Changed

- Argument diffs render per-path when `deepdiff` produces a `values_changed`-style mapping, instead of dumping the raw deepdiff object.

### Known limitations

- The pytest terminal summary is per-worker under `pytest-xdist` and is not aggregated across workers — run without `-n` if you need the full picture in one place.

## [0.2.1] - 2026-07-10

### Added

- Passing/failing/recorded snapshot results now surface in an `agentsnap snapshots` section of the pytest terminal summary, so `print()`-based feedback isn't swallowed by pytest's output capture.
- `agentsnap init` wizard now offers to install the offline embedding backend (`sentence-transformers`) during setup.

### Changed

- Snapshots and last-run files omit the `input` key entirely when no input was set, instead of writing a noisy `"input": null`.

## [0.2.0] - 2026-07-10

### Added

- **Replay mode** — deterministic asserts that replay recorded LLM responses instead of calling a live provider, so CI can run without API keys. Enable per-test with `AgentAsserter(mode="replay")`, per-suite via `[tool.agentsnap] mode = "replay"`, or per-run with the `--agentsnap-replay` / `--agentsnap-live` pytest flags.
- Snapshot format bumped to **v1.1**: every `llm_call` event now stores a `raw_response` alongside the parsed fields, which replay reconstructs into a provider-shaped response object.
- `replay_tools=True` lets `ToolAdapter` return recorded tool results under replay instead of re-invoking the real tool.
- Request-side diffing for replay mode, plus an exact-match short-circuit so identical requests skip semantic scoring entirely.
- New `SnapshotFormatError` and `ReplayError` exception types with clear, actionable messages (e.g. when a snapshot predates v1.1 and can't be replayed).
- `agentsnap diff` now runs the full semantic comparison pipeline (previously only did a raw JSON diff); new `agentsnap show` command replaces the old ad-hoc diff pretty-printer.
- OpenAI and Anthropic adapters now record `stream=True` calls by teeing the stream — chunks reach the caller unmodified while the assembled response is captured for the snapshot, instead of forcing non-streaming. Groq and OpenRouter inherit this via `OpenAIAdapter`.
- Replay reconstructs recorded streams deterministically: chunks are rebuilt into real SDK chunk/event objects and yielded back incrementally, with a clear `ReplayError` on streaming/non-streaming shape mismatches. Mistral still forces `stream=False`; the `client.messages.stream()` helper and async streams are not yet supported.
- `structural_tolerance` is now configurable via `pyproject.toml`, pytest ini options, or a per-test override, instead of being hardcoded.
- `agentsnap status` shows pass/fail/stale state for every snapshot in one CI-friendly table, exiting 1 if any snapshot is failing.
- `agentsnap update --all` batch-approves every failing or new snapshot in one pass, instead of requiring one `agentsnap update <test_name>` call per test.
- `agentsnap init` now scaffolds the project after the wizard finishes: it idempotently adds `__agent_snapshots__/.last_run/` to `.gitignore`, and offers to write an example snapshot test to `tests/test_agentsnap_example.py`.
- Regression errors print before/after text excerpts for every failing semantic step, not just the final output.
- Scenario namespacing: multiple snapshots per test function, with an automatic input hash (`input_sha8`) when no explicit scenario name is given; `agentsnap update` promotes all scenario variants at once.
- Async support: `async with` on `AgentRecorder`/`AgentAsserter`, and `ainvoke` on the LangGraph adapter.
- Volatile-field normalization (timestamps, token counts, request IDs) before trace comparison, to cut down on noisy diffs.
- MIT `LICENSE` file and a `py.typed` marker for downstream type checkers.
- CI workflow (test matrix across Python 3.10-3.13 on Linux and Windows, plus lint) and a tag-triggered release workflow that publishes to PyPI via Trusted Publishing.

### Changed

- `agentsnap status` now exits 1 when a golden or last-run snapshot file is unreadable (corrupt JSON), not just on FAIL — a corrupt committed golden should not pass CI.

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
