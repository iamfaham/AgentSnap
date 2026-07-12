# CLI

```bash
agentsnap init                                     # interactive setup wizard — choose backend and save config
agentsnap check                                    # verify current backend is working (exits 0/1)
agentsnap list                                     # list all snapshots
agentsnap status                                   # pass/fail/stale status for every snapshot (CI-friendly, exits 0/1)
agentsnap diff __agent_snapshots__/my_agent.json   # pretty-print a snapshot
agentsnap update my_agent                          # show diff and approve last run as new golden
agentsnap update my_agent --yes                    # approve without confirmation prompt
agentsnap update --all                             # batch-approve every failing or new snapshot
```

## `agentsnap init`

Interactive setup wizard. Presents three semantic-comparison backend options:

**[1] LLM judge — API (recommended, default)**
Calls a small LLM to score whether two responses are semantically equivalent. More accurate for factual agents. Requires an API key from one of:

- OpenRouter (recommended — one key gives access to many models)
- OpenAI, Anthropic, or any OpenAI-compatible provider

The wizard saves your key to `.env` — never to `pyproject.toml` (which gets committed to git).

**[2] Offline embeddings — all-MiniLM-L6-v2**
Uses cosine similarity between sentence embeddings. No API key, no internet after first use. The 22 MB model downloads once and is cached permanently. Runs on any machine including budget cloud VMs (CPU only, ~500 MB RAM). Requires `sentence-transformers` — the wizard offers to install it during setup.

**[3] Local LLM judge — coming soon**
Run the judge on your own machine using a locally hosted model (Ollama, llama.cpp, or any OpenAI-compatible local server). Visible in the menu but not yet selectable.

### Scaffolding

After you make your backend choice, `agentsnap init` also scaffolds your project:

- Idempotently adds `__agent_snapshots__/.last_run/` to `.gitignore` (creating the file if it doesn't exist).
- Offers to write an example snapshot test to `tests/test_agentsnap_example.py` (opt-in, skipped by default so pytest stays green until you replace the fake agent).

## `agentsnap check`

Verifies your currently configured backend is working, and exits 0 on success / 1 on failure — safe to use in CI health checks.

Output example (LLM judge):

```
Backend : LLM judge
Provider: https://openrouter.ai/api/v1
Model   : openai/gpt-4o-mini
API key : found
Status  : ok (0.42s)
```

Output example (offline embeddings):

```
Backend : offline embeddings (all-MiniLM-L6-v2)
Model   : cached at ~/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2
Status  : ok
```

## Approval workflow

When you intentionally change agent behavior (new prompt, model upgrade, new tool), use the CLI to review and approve it:

```bash
# 1. Run tests — they fail, new trace saved to .last_run/
pytest tests/test_my_agent.py

# 2. Approve — shows a diff and prompts for confirmation
agentsnap update my_agent
# -> Shows a diff: output changes, tool sequence changes, model changes
# -> Prompts: "Accept this as the new golden? [y/N]"

# 3. Or approve immediately without the prompt
agentsnap update my_agent --yes

# 4. Commit the new baseline
git add __agent_snapshots__/my_agent.json
git commit -m "approve: updated golden after Sonnet upgrade"
```

For multiple failures at once, run `agentsnap status` first to see what changed across every snapshot:

```bash
agentsnap status
```

Then batch-approve every failing or new snapshot in one pass with `agentsnap update --all` — it shows a diff per file, then asks for one confirmation for the whole batch, unless `--yes` is passed:

```bash
agentsnap update --all
agentsnap update --all --yes
```

## Other commands

```bash
agentsnap list                                     # list all snapshot files
agentsnap diff __agent_snapshots__/my_agent.json   # pretty-print a snapshot (full semantic comparison pipeline)
```

`agentsnap update <test_name>` promotes all scenario variants for that test at once (wildcard) — see [Scenarios](recording.md#scenarios-and-input-binding).
