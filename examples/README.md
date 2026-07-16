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
`.env` or the network.

## `--keep`

By default every example writes snapshots to a temp directory that's deleted
when the script exits. Pass `--keep` to keep it and print its path, useful for
poking at the recorded JSON by hand.

## Examples

| File              | Feature                              | What `--real` does                                                        |
|-------------------|---------------------------------------|----------------------------------------------------------------------------|
| `quickstart.py`   | The golden flow (record/pass/regress/approve/re-pass) via zero-instrumentation `PatchSet` | Same journey against a real LLM (Anthropic, OpenAI, or OpenRouter); the "regression" is a deliberately changed prompt |
| `replay.py`       | Replay mode (`mode="replay"`) -- deterministic, zero-network asserts, plus `replay_tools=True` to stub tool results too | Records once against the real API, then proves replay needs zero network afterwards (identical replay, a prompt-change catch, all against the real golden) |
| `streaming.py`    | Streaming: teeing a `stream=True` call while recording, replaying recorded chunks, and finalizing an abandoned stream | A real streamed call with chunks printed as they arrive, then replayed with the network off |
| `model_tools.py`  | Model tool-decision capture (`tool_requests`) -- catches the model requesting a different tool than the golden, even if your code's own tool call is unchanged | A real model given a trivial `get_weather` tool schema; the golden captures whatever it decides to call, then a replay-assert reproduces it with zero network |
| `async_agents.py` | Async client interception -- `PatchSet` patches `AsyncAnthropic`/`AsyncOpenAI` exactly like their sync counterparts | A real async round trip (`asyncio.run`), then replayed with the network off |
