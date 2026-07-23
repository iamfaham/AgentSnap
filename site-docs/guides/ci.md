# CI

Snapshots are committed to the repo. CI only runs the asserter — no real agent API calls needed unless your tests explicitly make them.

```yaml
name: Agent regression tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: pip

      - name: Install
        run: pip install -e ".[dev]"

      - name: Run agent snapshot tests
        run: pytest tests/ -v
        env:
          # Optional: enables LLM judge for higher-accuracy semantic comparison
          AGENTSNAP_JUDGE_API_KEY: ${{ secrets.AGENTSNAP_JUDGE_API_KEY }}
```

If `AGENTSNAP_JUDGE_API_KEY` is not set, agentsnap uses offline embedding comparison — provided you ran `agentsnap init` with option [2] (offline embeddings) and committed the resulting `pyproject.toml`. CI works without any secrets once that setup is done.

## Replay on every PR, live nightly

The two modes described in [Replay](replay.md) map naturally onto two CI jobs with different jobs:

- **Replay, on every push/PR** — deterministic, no API key, no cost, no flakes. Catches code regressions: prompt edits, broken tool wiring, changed call counts. Force it with `pytest --agentsnap-replay` or `mode = "replay"` in `[tool.agentsnap]`.
- **Live, nightly** — real API calls against the current model, catching drift that only shows up when the model itself changes (a provider update, a model swap). Force it with `pytest --agentsnap-live`.

```yaml
# nightly.yml
on:
  schedule:
    - cron: "0 6 * * *"
jobs:
  live:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install -e ".[dev]"
      - run: pytest tests/ --agentsnap-live -v
        env:
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
          AGENTSNAP_JUDGE_API_KEY: ${{ secrets.AGENTSNAP_JUDGE_API_KEY }}
```

## `pytest-xdist` limitation

Every `snapshot.run()` / `record_agent()` / `assert_agent()` use feeds a terminal summary section ("agentsnap snapshots") printed at the end of the pytest run. **Known limitation:** under `pytest-xdist`, this summary is per-worker and is not aggregated across workers — run without `-n` if you need the full picture.

---

## Validating agentsnap itself against live APIs (maintainers)

Everything above is what **your** CI needs. This section is about agentsnap's own repository CI, not something you need to replicate downstream.

agentsnap dogfoods its own "replay on PRs, live nightly" story: two workflows in [`.github/workflows/`](https://github.com/iamfaham/AgentSnap/tree/main/.github/workflows) continuously exercise the project against live provider APIs and the latest, unpinned provider SDKs.

- **`live-validation.yml`** — runs `python examples/run_all.py --real` against whichever provider secrets are configured in the repo. A missing secret degrades gracefully (that example prints a skip hint and exits 0) rather than failing the job.
- **`sdk-drift.yml`** — installs the latest unpinned versions of every provider SDK (bypassing the lockfile) and runs the hermetic test suite, to catch upstream SDK changes that break agentsnap's interception/reconstruction code before a user hits them.

Both are `workflow_dispatch` (manual, always runs) plus a monthly `schedule`, and the monthly runs are gated behind a single repo variable, `RUN_SCHEDULED_LIVE_TESTS` — unset (or anything but `'true'`) keeps the cron a no-op. Neither workflow can block a PR or the main test/frameworks jobs; both are informational only.

See [CONTRIBUTING.md](https://github.com/iamfaham/AgentSnap/blob/main/CONTRIBUTING.md#dogfooding-live-api-validation) for the maintainer's guide to arming, running, and triaging these.
