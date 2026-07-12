# Replay

Every assert can run in one of two modes:

| Mode | LLM calls | Catches | Best for |
|------|-----------|---------|----------|
| `live` (default) | Real API | Model/behavior drift | Nightly runs, pre-release |
| `replay` | None — recorded responses are replayed | Code regressions: prompt edits, tool wiring, loop logic | Every PR / CI |

In replay mode the recorded response for each LLM call is fed back to your agent — no API key, no cost, no flakes. The comparison flips to the **request side**: agentsnap fails the test if your code sends different prompts, makes a different number of LLM calls, or changes the tool sequence.

```python
from agentsnap import AgentAsserter

# per test
with AgentAsserter("my_agent", mode="replay") as a:
    a.output = my_agent(client, search_tool, "What is Python?")
```

```bash
# whole suite
pytest --agentsnap-replay        # force replay
pytest --agentsnap-live          # force live
```

```toml
[tool.agentsnap]
mode = "replay"   # make replay the project default
```

Tool calls still execute for real in replay mode. Pass `replay_tools=True` to stub them from the recording too (no side effects at all):

```python
with AgentAsserter("my_agent", mode="replay", replay_tools=True) as a:
    a.output = my_agent(client, search_tool, "What is Python?")
```

See `examples/demo_replay.py` for a full runnable walkthrough: record a golden run, replay it with the network disabled to prove zero live calls, then watch replay catch a prompt edit instantly.

## Caveats

- Replay needs snapshots recorded with agentsnap >= 0.2.0 (they include `raw_response`). Older snapshots raise `SnapshotFormatError` — re-record with `pytest --agentsnap-record`.
- Replay currently supports Anthropic, OpenAI, Groq, and OpenRouter. Other providers raise `ReplayError` — use live mode for those tests.
- With scenarios, pass `scenario=` explicitly in replay mode (input auto-hash is not available because the snapshot is read before the test body runs).
- If the replayed final output isn't byte-identical to the golden, scoring it needs a semantic backend — install and configure the embeddings backend (`pip install agentsnap[offline]`, then `agentsnap init` option 2) or configure a judge (`AGENTSNAP_JUDGE_API_KEY`).
- Async clients (`AsyncAnthropic`, `AsyncOpenAI`) are intercepted too — replay's no-network guarantee covers both sync and async clients, including async streams. The one remaining hole is the streamed OpenAI Responses API (`responses.create(stream=True)`), which passes through unrecorded. See `examples/demo_async.py`.

For streaming-specific replay behavior, see [Streaming](streaming.md).
