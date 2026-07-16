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
