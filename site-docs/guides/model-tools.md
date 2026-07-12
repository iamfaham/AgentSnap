# Model tool decisions

Beyond the tools your code actually executes, agentsnap also captures which tool the **model** decided to call. Every non-streaming Anthropic/OpenAI `llm_call` event records a `tool_requests` list — the `tool_use` blocks the model returned, each as `{"name": ..., "args": {...}}`. Groq and OpenRouter get this for free since they subclass `OpenAIAdapter`.

On assert, agentsnap compares the model's requested tool sequence (not just what your code executed) and fails `model_tools` if it changed, or `model_tool_args` if the same tool was requested with different arguments — surfaced in the report as `[MODEL TOOLS] ...`. This catches a model quietly choosing a different tool than the golden run even when your code's own tool-calling logic is untouched (a model update, a prompt injection, a provider-side regression).

```
[MODEL TOOLS] Model-requested tool sequence changed (edit distance 1): ['search'] -> ['delete_file']

[ARGS] model_tool:search[0]:
  args: {'query': 'capital of France'} -> {'path': '/etc/passwd'}

Failed checks: ['model_tools', 'model_tool_args']
```

## Backward compatibility

Backward compatible: the comparison only engages when **every** `llm_call` event on **both** sides of the diff carries `tool_requests`. Note this gate is trace-wide, not per-event: a single streamed call or non-Anthropic/OpenAI call anywhere in the trace disables the model-tools check for the whole run. Old goldens (recorded before this feature) never fail from the new surface.

`structural_tolerance` applies to BOTH the executed-tool sequence and the model-requested tool sequence — relaxing it for flaky tool ordering also relaxes the model-tools check. See [Configuration](configuration.md#structural_tolerance) for the dual-role note.

Scope today: non-streaming Anthropic and OpenAI calls, plus Groq/OpenRouter via inheritance. Streamed `tool_use` assembly is not captured yet.

## Demo

See `examples/demo_tool_use.py` for a full runnable walkthrough: record a golden run where the model requests `search`, re-run it unchanged, then watch agentsnap catch the model requesting `delete_file` instead.
