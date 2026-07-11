from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentsnap.adapters.anthropic import AnthropicAdapter
from agentsnap.adapters.tool import ToolAdapter
from agentsnap.core.snapshot import snapshot_path
from tests.fixtures.mock_agents import MockAnthropicClient, MockAnthropicResponse, SimpleToolAgent

import numpy as np

pytest_plugins = ["pytester"]


@pytest.fixture(autouse=True)
def _clean_judge_env(monkeypatch):
    # Earlier tests' config.load() can leak a real .env judge key into
    # os.environ; pin the no-judge code path regardless of the dev machine.
    monkeypatch.delenv("AGENTSNAP_JUDGE_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)


_DIM = 8

def _identical_embed(texts):
    v = np.ones(_DIM, dtype=float)
    v /= np.linalg.norm(v)
    return [v.copy() for _ in texts]


def _make_client():
    return AnthropicAdapter(MockAnthropicClient([MockAnthropicResponse("I'll search for that.")]))


def _search(query: str) -> str:
    return f"result_for_{query}"


def test_snapshot_run_auto_records_on_first_use(tmp_path, snapshot):
    """snapshot.run() records when no file exists (default behavior)."""
    snapshot._snapshot_dir_override = str(tmp_path / "snaps")
    snap_dir = str(tmp_path / "snaps")

    # Reach into the fixture to point it at tmp_path
    from agentsnap.pytest_plugin import SnapshotFixture
    from agentsnap.core.diff import LLMJudge
    sf = SnapshotFixture(
        snapshot_dir=snap_dir,
        semantic_threshold=0.92,
        llm_threshold=0.75,
        judge=None,
        force_record=False,
    )

    with sf.run("first_use", model="test") as s:
        client = _make_client()
        tool = ToolAdapter(_search, name="search")
        s.output = SimpleToolAgent(client, tool, "hello")

    assert snapshot_path("first_use", snap_dir).exists()


def test_force_record_overwrites_existing_snapshot(tmp_path):
    """force_record=True re-records even when a golden already exists."""
    from agentsnap.pytest_plugin import SnapshotFixture

    snap_dir = str(tmp_path / "snaps")

    # First: record a golden with output "v1"
    sf = SnapshotFixture(snap_dir, 0.92, 0.75, None, force_record=False)
    with sf.run("overwrite_test", model="test") as s:
        client = _make_client()
        tool = ToolAdapter(_search, name="search")
        s.output = "v1"
        SimpleToolAgent(client, tool, "hello")

    golden = json.loads(snapshot_path("overwrite_test", snap_dir).read_text())
    assert golden["output"] == "v1"

    # Second: force_record=True — should overwrite with "v2"
    sf2 = SnapshotFixture(snap_dir, 0.92, 0.75, None, force_record=True)
    with sf2.run("overwrite_test", model="test") as s:
        client = _make_client()
        tool = ToolAdapter(_search, name="search")
        s.output = "v2"
        SimpleToolAgent(client, tool, "hello")

    golden2 = json.loads(snapshot_path("overwrite_test", snap_dir).read_text())
    assert golden2["output"] == "v2"


def test_force_record_false_asserts_when_snapshot_exists(tmp_path):
    """force_record=False uses assert mode when snapshot exists — identical run passes."""
    from agentsnap.pytest_plugin import SnapshotFixture

    snap_dir = str(tmp_path / "snaps")

    sf = SnapshotFixture(snap_dir, 0.92, 0.75, None, force_record=False)
    with sf.run("assert_test", model="test") as s:
        client = _make_client()
        tool = ToolAdapter(_search, name="search")
        s.output = SimpleToolAgent(client, tool, "hello")

    # Second run: same inputs → should assert and pass
    sf2 = SnapshotFixture(snap_dir, 0.0, 0.0, None, force_record=False)
    with sf2.run("assert_test", model="test") as s:
        client = _make_client()
        tool = ToolAdapter(_search, name="search")
        s.output = SimpleToolAgent(client, tool, "hello")


def test_config_default_mode_is_live():
    from agentsnap.config import DEFAULTS
    assert DEFAULTS["mode"] == "live"


def test_snapshot_fixture_passes_mode_to_asserter(tmp_path):
    from agentsnap.pytest_plugin import SnapshotFixture

    fixture = SnapshotFixture(
        snapshot_dir=str(tmp_path),
        semantic_threshold=0.9,
        llm_threshold=0.7,
        judge=None,
        mode="replay",
    )
    asserter = fixture.assert_agent("t")
    assert asserter.mode == "replay"


def test_per_test_mode_override_wins(tmp_path):
    from agentsnap.pytest_plugin import SnapshotFixture

    fixture = SnapshotFixture(
        snapshot_dir=str(tmp_path),
        semantic_threshold=0.9,
        llm_threshold=0.7,
        judge=None,
        mode="replay",
    )
    asserter = fixture.assert_agent("t", mode="live")
    assert asserter.mode == "live"


def test_replay_tools_passthrough(tmp_path):
    from agentsnap.pytest_plugin import SnapshotFixture

    fixture = SnapshotFixture(
        snapshot_dir=str(tmp_path),
        semantic_threshold=0.9,
        llm_threshold=0.7,
        judge=None,
    )
    asserter = fixture.assert_agent("t", mode="replay", replay_tools=True)
    assert asserter.replay_tools is True


# -- Pytester end-to-end: replay flag / mode resolution -----------------------
#
# These drive the `snapshot` fixture through a real (in-process) pytest run so
# the flag/ini resolution chain in pytest_plugin.py is exercised end to end,
# not just unit-tested against SnapshotFixture directly. The mini agent under
# test makes zero LLM calls, so no SDKs or embedding backends are required.

_STUB_TEST_SOURCE = """
    def test_agent(snapshot):
        with snapshot.run("flag_e2e") as s:
            s.output = "constant output"
    """


def _last_run_mode(pytester: pytest.Pytester) -> str:
    last_run_file = pytester.path / "__agent_snapshots__" / ".last_run" / "flag_e2e.json"
    data = json.loads(last_run_file.read_text(encoding="utf-8"))
    return data["result"]["mode"]


def test_pytester_record_then_replay_flag(pytester: pytest.Pytester) -> None:
    """Record on first run, then --agentsnap-replay asserts and last_run records mode='replay'."""
    pytester.makeconftest("")  # pins snapshot dir discovery to this pytester tmp dir
    pytester.makepyfile(test_flag_e2e=_STUB_TEST_SOURCE)

    record_result = pytester.runpytest()
    record_result.assert_outcomes(passed=1)

    last_run_file = pytester.path / "__agent_snapshots__" / ".last_run" / "flag_e2e.json"
    assert not last_run_file.exists(), "record mode should not write a .last_run file"

    replay_result = pytester.runpytest("--agentsnap-replay")
    replay_result.assert_outcomes(passed=1)
    assert _last_run_mode(pytester) == "replay"


def test_pytester_replay_and_live_flags_live_wins(pytester: pytest.Pytester) -> None:
    """--agentsnap-replay together with --agentsnap-live resolves to live (live always wins)."""
    pytester.makeconftest("")
    pytester.makepyfile(test_flag_e2e=_STUB_TEST_SOURCE)

    pytester.runpytest().assert_outcomes(passed=1)  # record

    result = pytester.runpytest("--agentsnap-replay", "--agentsnap-live")
    result.assert_outcomes(passed=1)
    assert _last_run_mode(pytester) == "live"


def test_pytester_terminal_summary_shows_recorded_then_passed(pytester: pytest.Pytester) -> None:
    """agentsnap terminal summary section appears with RECORDED then PASSED lines, no -s needed."""
    pytester.makeconftest("")
    pytester.makepyfile(test_flag_e2e=_STUB_TEST_SOURCE)

    record_result = pytester.runpytest()
    record_result.assert_outcomes(passed=1)
    record_result.stdout.fnmatch_lines(["*agentsnap snapshots*", "*RECORDED*"])
    # Exactly one RECORDED entry: the recorder itself emits it now (no double
    # append from _AutoContext, which used to append its own entry too).
    recorded_lines = [
        line for line in record_result.outlines if "RECORDED" in line and "flag_e2e" in line
    ]
    assert len(recorded_lines) == 1

    assert_result = pytester.runpytest()
    assert_result.assert_outcomes(passed=1)
    assert_result.stdout.fnmatch_lines(["*agentsnap snapshots*", "*PASSED*flag_e2e*"])


def test_pytester_record_agent_shows_recorded_in_summary(pytester: pytest.Pytester) -> None:
    """snapshot.record_agent() (explicit record mode) also feeds the terminal summary."""
    pytester.makeconftest("")
    pytester.makepyfile(
        test_explicit_rec="""
        def test_agent(snapshot):
            with snapshot.record_agent("explicit_rec") as rec:
                rec.output = "constant output"
        """
    )

    result = pytester.runpytest()
    result.assert_outcomes(passed=1)
    result.stdout.fnmatch_lines(["*agentsnap snapshots*", "*RECORDED*explicit_rec*"])
    recorded_lines = [
        line for line in result.outlines if "RECORDED" in line and "explicit_rec" in line
    ]
    assert len(recorded_lines) == 1


def test_pytester_terminal_summary_shows_failed(pytester: pytest.Pytester) -> None:
    """A structural regression shows a FAILED line in the summary section, no -s needed.

    Uses a tool rename (not an output change) so the failure is structural and
    never touches a semantic backend inside the pytester environment.
    """
    pytester.makeconftest("")
    tool_test = """
    from agentsnap.adapters.tool import ToolAdapter

    def test_agent(snapshot):
        with snapshot.run("flag_e2e") as s:
            tool = ToolAdapter(lambda **kw: "r", name="{name}")
            tool(query="x")
            s.output = "constant output"
    """
    pytester.makepyfile(test_flag_e2e=tool_test.replace("{name}", "search"))
    pytester.runpytest().assert_outcomes(passed=1)  # record golden with tool 'search'

    pytester.makepyfile(test_flag_e2e=tool_test.replace("{name}", "fetch"))
    result = pytester.runpytest()  # renamed tool -> structural failure
    result.assert_outcomes(failed=1)
    result.stdout.fnmatch_lines(["*agentsnap snapshots*", "*FAILED*flag_e2e*"])


def test_pytester_ini_mode_replay_resolves_without_flags(pytester: pytest.Pytester) -> None:
    """agentsnap_mode = replay in ini resolves to replay mode with no CLI flags."""
    pytester.makeconftest("")
    pytester.makeini(
        """
        [pytest]
        agentsnap_mode = replay
        """
    )
    pytester.makepyfile(test_flag_e2e=_STUB_TEST_SOURCE)

    pytester.runpytest().assert_outcomes(passed=1)  # record

    result = pytester.runpytest()  # no flags; ini alone should resolve to replay
    result.assert_outcomes(passed=1)
    assert _last_run_mode(pytester) == "replay"
