# Streaming

`AnthropicAdapter` and `OpenAIAdapter` tee `stream=True` calls instead of forcing non-streaming (Groq and OpenRouter inherit this since they subclass `OpenAIAdapter`). Chunks flow through to your agent unmodified while the assembled text/tokens are recorded, with `raw_response={"__stream__": True, "chunks": [...]}`.

## Replayed streams

Replay rebuilds the recorded chunks into real SDK chunk/event objects and yields them back incrementally — the agent consumes them exactly like a live stream, with zero API calls. Replaying a streaming recording against a non-streaming request (or vice versa) raises `ReplayError` with a "shape mismatch" message.

## Unconsumed-stream finalization

A stream that is never iterated and never closed is finalized automatically at recorder/asserter exit, but consuming or closing it promptly is still recommended so events appear in call order.

## Limitations

Not yet supported:

- The `client.messages.stream()` context-manager helper.
- Streamed OpenAI Responses-API calls.

Mistral still forces `stream=False` on every call.

See `examples/streaming.py` for a full runnable walkthrough of recording and replaying a streaming agent, and `examples/async_agents.py` for the async-client version.
