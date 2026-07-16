"""
scenarios.py -- Scenario namespacing: one agent, many golden snapshots.

One aspect only: `scenario=` (and its auto-hash sibling) let a single
`test_name` own several independent goldens -- one per input -- instead of
overwriting a single snapshot every time the input changes.

  - Explicit `scenario="..."` writes `{test_name}__{scenario}.json`.
  - Leaving `scenario=None` but setting `.input` derives the suffix from an
    8-char sha256 hash of the input: `{test_name}__{sha8}.json`. Handy when
    you don't want to invent scenario names for every input in a parametrized
    test.
  - Whichever golden gets picked, agentsnap remembers the input it was
    recorded with (when written via `AgentRecorder.input_data`) and warns --
    once, non-fatally -- if you assert against it with a different input.

Usage:
    python examples/scenarios.py             # mock only, no keys/network needed
    python examples/scenarios.py --real      # mock, then two real inputs recorded
                                               # as two scenario goldens (needs
                                               # ANTHROPIC_API_KEY, OPENAI_API_KEY,
                                               # or OPENROUTER_API_KEY; prints a skip
                                               # hint and exits 0 if none set)
    python examples/scenarios.py --keep      # keep the temp snapshot dir, print its path

The journey (mock_demo):
  1. Explicit scenario -- record `weather__us_west.json`.
  2. Auto-hash scenario -- record `weather__<sha8>.json` from `.input`, no
     scenario name needed.
  3. Input-binding warning -- assert the explicit-scenario golden again with a
     different `.input`; agentsnap warns the baseline may no longer match.
"""

from __future__ import annotations

import sys
import unittest.mock as mock
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import _common as ex
from agentsnap import PatchSet
from agentsnap.core.asserter import AgentAsserter
from agentsnap.core.recorder import AgentRecorder
from agentsnap.core.snapshot import input_sha8, list_snapshots

NAME = "weather"


def _agent(region: str) -> str:
    """A tiny agent -- raw anthropic client, zero agentsnap imports."""
    import anthropic

    client = anthropic.Anthropic(api_key="demo-key-no-real-call")
    client.messages.create(
        model="claude-haiku-4-5",
        messages=[{"role": "user", "content": f"weather in {region}"}],
        max_tokens=50,
    )
    return f"Sunny in {region}"


def mock_demo(snapshot_dir: str) -> None:
    ex.header("SCENARIOS (mock)  --  one agent, many goldens")
    print("  Same test_name, different inputs -- each gets its own scenario file.\n")

    from anthropic.resources.messages.messages import Messages as _AnthMessages

    response_west = ex.make_anthropic_message("Looking that up.")
    response_eu = ex.make_anthropic_message("Checking now.")

    ex.subheader("Step 1  Explicit scenario -- scenario='us_west'")
    input_west = {"region": "us-west"}
    with mock.patch.object(_AnthMessages, "create", return_value=response_west):
        with PatchSet():
            with AgentRecorder(NAME, snapshot_dir=snapshot_dir, scenario="us_west") as rec:
                rec.input_data = input_west
                rec.output = _agent("us-west")
    print(f"  Golden written: {NAME}__us_west.json")

    ex.subheader("Step 2  Auto-hash scenario -- no scenario= given, derived from .input")
    input_eu = {"region": "eu-central"}
    with mock.patch.object(_AnthMessages, "create", return_value=response_eu):
        with PatchSet():
            with AgentRecorder(NAME, snapshot_dir=snapshot_dir) as rec:
                rec.input_data = input_eu
                rec.output = _agent("eu-central")
    eu_hash = input_sha8(input_eu)
    print(f"  Golden written: {NAME}__{eu_hash}.json  (sha8 of {input_eu!r})")

    print("\n  Snapshots on disk:")
    for p in list_snapshots(snapshot_dir):
        print(f"    {p.name}")

    ex.subheader("Step 3  Reassert 'us_west' with a changed .input -- one-time warning")
    with mock.patch.object(_AnthMessages, "create", return_value=response_west):
        with PatchSet():
            with AgentAsserter(
                NAME, snapshot_dir=snapshot_dir, scenario="us_west", embed_fn=ex.demo_embed
            ) as a:
                a.input = {"region": "us-east"}  # different from what was recorded
                a.output = _agent("us-west")
    print("  (warning above; the run still compared cleanly against the 'us_west' golden)")

    ex.header("Done -- scenario= for named variants, .input for ad-hoc auto-hashed ones.")


def real_demo(snapshot_dir: str) -> None:
    ex.maybe_load_dotenv()
    detected = ex.detect_real_client()
    if detected.client is None:
        ex.header("SCENARIOS (real)  --  skipped")
        print(f"  {detected.hint}")
        return

    ex.header(f"SCENARIOS (real)  --  provider: {detected.provider}, model: {detected.model}")
    print("  Two real inputs, two scenario goldens.\n")

    name = f"{NAME}_real"

    def call(question: str) -> str:
        if detected.provider == "anthropic":
            response = detected.client.messages.create(
                model=detected.model,
                messages=[{"role": "user", "content": question}],
                max_tokens=ex.REAL_MAX_TOKENS,
                temperature=ex.REAL_TEMPERATURE,
            )
            text = response.content[0].text
        else:
            response = detected.client.chat.completions.create(
                model=detected.model,
                messages=[{"role": "user", "content": question}],
                max_tokens=ex.REAL_MAX_TOKENS,
                temperature=ex.REAL_TEMPERATURE,
            )
            text = response.choices[0].message.content
        return f"Answer: {text}"

    ex.subheader("Step 1  Record scenario='paris'")
    with PatchSet():
        with AgentAsserter(name, snapshot_dir=snapshot_dir, scenario="paris", embed_fn=ex.demo_embed) as a:
            a.output = call("What is the capital of France?")
    print(f"  Golden written: {name}__paris.json")

    ex.subheader("Step 2  Record scenario='tokyo'")
    with PatchSet():
        with AgentAsserter(name, snapshot_dir=snapshot_dir, scenario="tokyo", embed_fn=ex.demo_embed) as a:
            a.output = call("What is the capital of Japan?")
    print(f"  Golden written: {name}__tokyo.json")

    print("\n  Snapshots on disk:")
    for p in list_snapshots(snapshot_dir):
        if p.stem.startswith(name):
            print(f"    {p.name}")


def main() -> None:
    args = ex.parse_args(__doc__)
    with ex.temp_snapshot_dir(keep=args.keep) as snapshot_dir:
        if args.keep:
            print(f"Snapshot dir: {snapshot_dir}")
        mock_demo(snapshot_dir)
        if args.real:
            real_demo(snapshot_dir)
    ex.header("Scenarios complete")


if __name__ == "__main__":
    main()
