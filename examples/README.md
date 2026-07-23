# agentsnap examples

Every example follows the same shape:

- `mock_demo()` -- runs entirely offline. No API keys, no network. Always exits 0.
- `real_demo()` -- the same story against a real LLM. Runs automatically when you
  pass `--real`; if no usable API key is found it prints a one-line hint and
  exits 0 (a skip, never a failure).
- `main()` -- wires the two together and owns `--real` / `--keep`.

```bash
python examples/quickstart.py            # mock only
python examples/quickstart.py --real      # mock, then the real-LLM version
python examples/quickstart.py --keep      # keep the temp snapshot dir, print its path
```

## Key setup for `--real`

`detect_real_client()` (in `_common.py`) checks these env vars in order and uses
whichever is set first:

| Env var              | Provider used             | Model               |
|----------------------|----------------------------|---------------------|
| `ANTHROPIC_API_KEY`  | `anthropic`                | `claude-haiku-4-5`  |
| `OPENAI_API_KEY`     | `openai`                   | `gpt-4o-mini`       |
| `OPENROUTER_API_KEY` | `openai` (OpenRouter base) | `openai/gpt-4o-mini`|

Real calls use `temperature=0` and `max_tokens<=150` to stay cheap and close to
deterministic. Set any one of the three in a `.env` file at the repo root (or
in your shell) and `--real` will pick it up automatically -- set
`AGENTSNAP_SKIP_DOTENV=1` to skip `.env` loading. Mock mode never touches
`.env` or the network. Every `--real` demo scores similarity with the
built-in lightweight comparator in `_common.demo_embed()`, so a single
provider key is all you need -- no judge, no downloaded model, no extra
setup; real projects should run `agentsnap init` to configure a proper
comparison backend instead. Most `--real` paths make exactly ONE live call
to record a golden, then use `mode="replay"` for every regression check
after that -- deterministic and free, and the same pattern real projects
should use instead of asserting two live calls against each other.

## `--keep`

By default every example writes snapshots to a temp directory that's deleted
when the script exits. Pass `--keep` to keep it and print its path, useful for
poking at the recorded JSON by hand.

## Examples

| File              | Feature                              | What `--real` does                                                        |
|-------------------|---------------------------------------|----------------------------------------------------------------------------|
| `quickstart.py`   | The golden flow (record/pass/regress/approve/re-pass) via zero-instrumentation `PatchSet` | One real call records the golden (Anthropic, OpenAI, or OpenRouter); every pass/regression/re-pass check after that runs in `mode="replay"` against a deliberately changed prompt -- deterministic, no second live call |
| `replay.py`       | Replay mode (`mode="replay"`) -- deterministic, zero-network asserts, plus `replay_tools=True` to stub tool results too | Records once against the real API, then proves replay needs zero network afterwards (identical replay, a prompt-change catch, all against the real golden) |
| `streaming.py`    | Streaming: teeing a `stream=True` call while recording, replaying recorded chunks, and finalizing an abandoned stream | A real streamed call with chunks printed as they arrive, then replayed with the network off |
| `model_tools.py`  | Model tool-decision capture (`tool_requests`) -- catches the model requesting a different tool than the golden, even if your code's own tool call is unchanged | A real model given a trivial `get_weather` tool schema; the golden captures whatever it decides to call, then a replay-assert reproduces it with zero network |
| `async_agents.py` | Async client interception -- `PatchSet` patches `AsyncAnthropic`/`AsyncOpenAI` exactly like their sync counterparts | A real async round trip (`asyncio.run`), then replayed with the network off |
| `scenarios.py`    | Scenario namespacing -- explicit `scenario=` and input auto-hash give one `test_name` many independent goldens, plus the one-time input-binding warning | Two real inputs recorded as two scenario goldens, then listed |
| `tuning.py`       | Comparison tuning -- `semantic_threshold` (loose vs strict) on a paraphrase, and `structural_tolerance` absorbing a model tool-choice swap | LLM judge scoring a real paraphrase for equivalence (needs `OPENAI_API_KEY`/`OPENROUTER_API_KEY`; an Anthropic-only key skips just the judge segment) |
| `cli_workflow.py` | The CLI approval loop -- `agentsnap status`/`update --all --yes` driven via subprocess exactly as a developer types them | One real call records the golden; the "drifted run" is a changed prompt caught via `mode="replay"` (no second live call), then the same status/update/status loop |
| `pytest_fixture.py` | The pytest plugin as users actually run it -- a mini test file run via `python -m pytest`, twice | A real call on the first pytest run, replayed (zero network) on the second via `pytest --agentsnap-replay` |
| `providers.py`    | The non-core provider adapters -- Gemini, Cohere, Mistral, Groq, each doing the same record/pass/regression story. Gemini/Cohere/Mistral are live-mode only today (`mode="replay"` raises `ReplayError`); Groq subclasses `OpenAIAdapter` and gets replay/streaming for free | One tiny real call per provider key present (`GEMINI_API_KEY`/`GOOGLE_API_KEY`, `COHERE_API_KEY`, `MISTRAL_API_KEY`, `GROQ_API_KEY`); providers without a key print a skip hint |
| `run_all.py`      | The matrix runner -- runs every example above as its own subprocess and prints a PASS/FAIL/time table | Forwards `--real` to every example so the whole suite runs against real APIs in one command |

## Validating a release against real APIs

`run_all.py --real` is the one-command way to exercise every example against
whatever real provider keys you have set (in `.env` or the shell). Providers
without a key present skip gracefully rather than failing, so this degrades
fine with a partial key set -- useful before cutting a release:

```bash
python examples/run_all.py --real
```

Costs real API usage (small, cheap calls -- see "Key setup" above), so it's
never run in CI on `push`/`pull_request`. Run it locally, or in a scratch
environment, before shipping.

## Validating against real APIs in CI

`run_all.py --real` also runs in CI via the `live-validation` workflow
(`.github/workflows/live-validation.yml`), manually or on a monthly schedule
gated by the `RUN_SCHEDULED_LIVE_TESTS` repo variable. See "Dogfooding: live
API validation" in `CONTRIBUTING.md` for how to trigger a manual run and how
to arm/disarm the scheduled one.
