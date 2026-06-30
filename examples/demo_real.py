"""
demo_real.py -- Run agentsnap with real API calls.

Covers the full user journey with real LLMs:
  1. Zero instrumentation (PatchSet) -- raw SDK client, no adapter needed
  2. Provider demos -- adapter-based, one per available key
  3. Scenario 1 -- catching an unintended structural change
  4. Scenario 2 -- intentional model upgrade with approval
  5. Scenario 3 -- CI simulation via pytest

Set whichever API keys you have in .env -- providers without a key are skipped.
OpenRouter is used for PatchSet demo and scenarios (only key needed for those).

Required env vars:
    OPENROUTER_API_KEY   -- used for PatchSet demo, scenarios 1-3
    ANTHROPIC_API_KEY    -- optional (adapter demo)
    OPENAI_API_KEY       -- optional (adapter demo)
    GEMINI_API_KEY       -- optional (adapter demo)
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
if not os.getenv("AGENTSNAP_SKIP_DOTENV"):
    load_dotenv(Path(__file__).parent.parent / ".env")

import openai

from agentsnap import PatchSet
from agentsnap.adapters.openrouter import OpenRouterAdapter, OPENROUTER_BASE_URL
from agentsnap.adapters.tool import ToolAdapter
from agentsnap.core.asserter import AgentAsserter
from agentsnap.core.recorder import AgentRecorder
from agentsnap.core.snapshot import last_run_path, snapshot_path
from agentsnap.exceptions import AgentRegressionError

SNAPSHOT_DIR = "__agent_snapshots__"
SEPARATOR = "=" * 60
THIN = "-" * 60

MODEL_A = "anthropic/claude-haiku-4-5"
MODEL_B = "openai/gpt-4o-mini"


# -- Shared helpers ------------------------------------------------------------

def lookup(query: str) -> str:
    data = {
        "agentsnap": "A deterministic snapshot testing harness for AI agents.",
        "python":    "A high-level, interpreted programming language.",
    }
    return data.get(query.lower(), f"No entry found for '{query}'.")


def make_client(key: str) -> OpenRouterAdapter:
    return OpenRouterAdapter(
        openai.OpenAI(api_key=key, base_url=OPENROUTER_BASE_URL)
    )


def call_agent(client, tool, query: str, model: str = MODEL_A) -> str:
    client.chat.completions.create(
        model=model,
        max_tokens=256,
        messages=[{"role": "user", "content": f"Look up: {query}"}],
    )
    result = tool(query=query)
    return f"Answer: {result}"


def header(title: str) -> None:
    print(f"\n{SEPARATOR}\n  {title}\n{SEPARATOR}")

def step(msg: str) -> None:
    print(f"\n{THIN}\n  {msg}\n{THIN}")


# -- Zero instrumentation (PatchSet) ------------------------------------------

def zero_instrumentation_demo(key: str) -> None:
    """
    PatchSet captures any raw SDK client -- no adapter needed.

    The agent function below has zero agentsnap imports. The test wraps
    it with PatchSet and AgentAsserter; that's the only change.
    """
    header("ZERO INSTRUMENTATION (PatchSet)")
    print("  Raw openai.OpenAI client. No adapter. No changes to agent code.\n")

    # This is what a user's agent looks like -- no agentsnap code
    def my_agent(query: str) -> str:
        client = openai.OpenAI(api_key=key, base_url=OPENROUTER_BASE_URL)
        client.chat.completions.create(
            model=MODEL_A,
            max_tokens=256,
            messages=[{"role": "user", "content": f"Look up: {query}"}],
        )
        result = lookup(query)
        return f"Answer: {result}"

    name = "real_zero_instrument"

    step("Step 1  First run -- no snapshot yet, golden recorded automatically")
    with PatchSet():
        with AgentAsserter(name, snapshot_dir=SNAPSHOT_DIR) as a:
            a.output = my_agent("agentsnap")

    step("Step 2  Identical run -- expect PASS")
    try:
        with PatchSet():
            with AgentAsserter(name, snapshot_dir=SNAPSHOT_DIR) as a:
                a.output = my_agent("agentsnap")
    except AgentRegressionError as e:
        print(str(e))

    step("Step 3  Regression -- changed final output")
    try:
        with PatchSet():
            with AgentAsserter(name, snapshot_dir=SNAPSHOT_DIR) as a:
                # LLM call is the same, but agent produces a different answer
                client = openai.OpenAI(api_key=key, base_url=OPENROUTER_BASE_URL)
                client.chat.completions.create(
                    model=MODEL_A, max_tokens=256,
                    messages=[{"role": "user", "content": "Look up: agentsnap"}],
                )
                a.output = "This is a completely different answer from the agent."
    except AgentRegressionError as e:
        print(str(e))


# -- Provider demos (record on first run, assert on subsequent) ----------------

def run_provider(name: str, make_client_fn, call_agent_fn) -> None:
    snap_name = f"real_{name}"
    try:
        with AgentAsserter(snap_name, snapshot_dir=SNAPSHOT_DIR) as a:
            a.output = call_agent_fn(make_client_fn(), ToolAdapter(lookup, name="lookup"), "agentsnap")
        print(f"[{name}] OK")
    except AgentRegressionError as e:
        print(f"[{name}] FAIL  regression: {e.diff_report.failed_checks}")
    except Exception as e:
        print(f"[{name}] ERROR: {type(e).__name__}: {e}")


def openrouter_demo(key: str) -> None:
    run_provider(
        "openrouter",
        lambda: make_client(key),
        lambda client, tool, q: call_agent(client, tool, q, MODEL_A),
    )


def anthropic_demo() -> None:
    k = os.getenv("ANTHROPIC_API_KEY")
    if not k:
        print("[anthropic] skipped -- ANTHROPIC_API_KEY not set"); return
    try:
        import anthropic
        from agentsnap.adapters.anthropic import AnthropicAdapter
    except ImportError:
        print("[anthropic] skipped -- pip install anthropic"); return
    def _call(client, tool, q):
        client.messages.create(model="claude-haiku-4-5-20251001", max_tokens=256,
                                messages=[{"role": "user", "content": f"Look up: {q}"}])
        return f"Answer: {tool(query=q)}"
    run_provider("anthropic", lambda: AnthropicAdapter(anthropic.Anthropic(api_key=k)), _call)


def openai_demo() -> None:
    k = os.getenv("OPENAI_API_KEY")
    if not k:
        print("[openai] skipped -- OPENAI_API_KEY not set"); return
    try:
        import openai as _openai
        from agentsnap.adapters.openai import OpenAIAdapter
    except ImportError:
        print("[openai] skipped -- pip install openai"); return
    def _call(client, tool, q):
        client.chat.completions.create(model="gpt-4o-mini", max_tokens=256,
                                       messages=[{"role": "user", "content": f"Look up: {q}"}])
        return f"Answer: {tool(query=q)}"
    run_provider("openai", lambda: OpenAIAdapter(_openai.OpenAI(api_key=k)), _call)


def gemini_demo() -> None:
    k = os.getenv("GEMINI_API_KEY")
    if not k:
        print("[gemini] skipped -- GEMINI_API_KEY not set"); return
    try:
        from google import genai
        from agentsnap.adapters.google import GeminiAdapter
    except ImportError:
        print("[gemini] skipped -- pip install google-genai"); return
    def _call(client, tool, q):
        client.models.generate_content(model="gemini-2.0-flash", contents=f"Look up: {q}")
        return f"Answer: {tool(query=q)}"
    run_provider("gemini", lambda: GeminiAdapter(genai.Client(api_key=k)), _call)


# -- Scenario demos (requires OpenRouter) -------------------------------------

def scenario_1_unintended_change(key: str) -> None:
    """
    SCENARIO 1: Catch an unintended change.

    The golden agent makes one tool call: lookup(query=...).
    A 'bad' version adds a second tool call: summarize(content=...).
    agentsnap catches the structural drift immediately.
    """
    header("SCENARIO 1: Catching an unintended change")
    snap_name = "scenario_golden"

    def summarize(content: str) -> str:
        return f"Summary of: {content}"

    # -- Record the golden (one tool call) ------------------------------------
    step("Recording golden run  (1 LLM call + 1 tool call: lookup)")
    with AgentRecorder(snap_name, snapshot_dir=SNAPSHOT_DIR) as rec:
        client = make_client(key)
        tool = ToolAdapter(lookup, name="lookup")
        rec.input_data = {"query": "python"}
        rec.output = call_agent(client, tool, "python", MODEL_A)
    print(f"  Recorded: {rec.output}")

    # -- Run mutated agent (two tool calls) -----------------------------------
    step("Running mutated agent  (adds an extra summarize tool call)")
    print("  Expected: AgentRegressionError with structural diff")
    print()

    try:
        with AgentAsserter(snap_name, snapshot_dir=SNAPSHOT_DIR) as a:
            client = make_client(key)
            tool = ToolAdapter(lookup, name="lookup")
            summarize_tool = ToolAdapter(summarize, name="summarize")

            # Same LLM call
            client.chat.completions.create(
                model=MODEL_A, max_tokens=256,
                messages=[{"role": "user", "content": "Look up: python"}],
            )
            # Same first tool call
            result = tool(query="python")
            # Extra tool call -- this is the unintended change
            summary = summarize_tool(content=result)
            a.output = f"Answer: {summary}"

    except AgentRegressionError as e:
        print(str(e))
        print()
        print("  -> You introduced a new tool call. If unintended, revert your code.")
        print("     If intended, run: agentsnap update scenario_golden")

    # Cleanup
    for p in [snapshot_path(snap_name, SNAPSHOT_DIR),
              last_run_path(snap_name, SNAPSHOT_DIR)]:
        p.unlink(missing_ok=True)


def scenario_2_model_upgrade(key: str) -> None:
    """
    SCENARIO 2: Intentional model upgrade with approval.

    Record golden with MODEL_A. Upgrade to MODEL_B.
    Semantic drift is detected. We inspect and approve.
    Second run passes with the new model as baseline.
    """
    header("SCENARIO 2: Intentional model upgrade + approval")
    snap_name = "scenario_upgrade"

    # -- Record golden with MODEL_A -------------------------------------------
    step(f"Recording golden run with MODEL_A  ({MODEL_A})")
    with AgentRecorder(snap_name, snapshot_dir=SNAPSHOT_DIR, model=MODEL_A) as rec:
        client = make_client(key)
        tool = ToolAdapter(lookup, name="lookup")
        rec.input_data = {"query": "agentsnap"}
        rec.output = call_agent(client, tool, "agentsnap", MODEL_A)
    print(f"  Recorded: {rec.output}")

    # -- Assert with MODEL_B (drift expected) ---------------------------------
    step(f"Asserting with MODEL_B  ({MODEL_B})  -- expect semantic drift")
    try:
        with AgentAsserter(snap_name, snapshot_dir=SNAPSHOT_DIR,
                           semantic_threshold=0.92, llm_threshold=0.75) as a:
            client = make_client(key)
            tool = ToolAdapter(lookup, name="lookup")
            a.output = call_agent(client, tool, "agentsnap", MODEL_B)

        # May pass if models are similar enough
        print("  Passed -- models are semantically equivalent at this threshold.")

    except AgentRegressionError as e:
        print(str(e))
        print()

        # -- Approve: promote last run to golden ------------------------------
        step("Approving the new model output  (agentsnap update)")
        src = last_run_path(snap_name, SNAPSHOT_DIR)
        dst = snapshot_path(snap_name, SNAPSHOT_DIR)
        shutil.copy2(src, dst)
        print(f"  Copied .last_run/{snap_name}.json -> {snap_name}.json")
        print(f"  (In real use: python -m agentsnap.cli update {snap_name})")

        # -- Re-assert with MODEL_B (should now pass) -------------------------
        step("Re-asserting with MODEL_B against updated golden -- expect PASS")
        try:
            with AgentAsserter(snap_name, snapshot_dir=SNAPSHOT_DIR,
                               semantic_threshold=0.92, llm_threshold=0.75) as a:
                client = make_client(key)
                tool = ToolAdapter(lookup, name="lookup")
                a.output = call_agent(client, tool, "agentsnap", MODEL_B)
            print("  PASSED -- MODEL_B is now the baseline.")
        except AgentRegressionError as e2:
            print("  Still drifting (LLM responses are non-deterministic):")
            print(str(e2))

    # Cleanup
    for p in [snapshot_path(snap_name, SNAPSHOT_DIR),
              last_run_path(snap_name, SNAPSHOT_DIR)]:
        p.unlink(missing_ok=True)


def scenario_3_ci(key: str) -> None:
    """
    SCENARIO 3: CI simulation via pytest.

    Writes a real pytest test file, runs it with pytest, shows the output.
    This is exactly what CI does on every PR.
    """
    header("SCENARIO 3: CI simulation -- pytest integration")

    snap_name = "scenario_ci"

    # -- Record the golden so the test has something to assert against ---------
    step("Recording golden snapshot for CI test")
    with AgentRecorder(snap_name, snapshot_dir=SNAPSHOT_DIR) as rec:
        client = make_client(key)
        tool = ToolAdapter(lookup, name="lookup")
        rec.input_data = {"query": "agentsnap"}
        rec.output = call_agent(client, tool, "agentsnap", MODEL_A)
    print(f"  Golden written: {rec.output}")

    snap_dir_abs = str(Path(SNAPSHOT_DIR).resolve())

    # -- Write a real pytest test file ----------------------------------------
    test_code = f'''\
"""Auto-generated CI test -- produced by demo_real.py scenario 3."""
import openai
from agentsnap.adapters.openrouter import OpenRouterAdapter, OPENROUTER_BASE_URL
from agentsnap.adapters.tool import ToolAdapter
from agentsnap.core.asserter import AgentAsserter
import os

SNAPSHOT_DIR = {snap_dir_abs!r}
MODEL = {MODEL_A!r}

def lookup(query: str) -> str:
    data = {{"agentsnap": "A deterministic snapshot testing harness for AI agents."}}
    return data.get(query.lower(), f"No entry found for {{query!r}}.")

def test_agent_no_regression():
    key = os.environ["OPENROUTER_API_KEY"]
    client = OpenRouterAdapter(openai.OpenAI(api_key=key, base_url=OPENROUTER_BASE_URL))
    tool = ToolAdapter(lookup, name="lookup")

    with AgentAsserter("scenario_ci", snapshot_dir=SNAPSHOT_DIR, semantic_threshold=0.92, llm_threshold=0.75) as a:
        client.chat.completions.create(
            model=MODEL, max_tokens=256,
            messages=[{{"role": "user", "content": "Look up: agentsnap"}}],
        )
        result = tool(query="agentsnap")
        a.output = f"Answer: {{result}}"
'''

    with tempfile.NamedTemporaryFile(
        mode="w", suffix="_test_ci.py", delete=False, dir=".", prefix="tmp_agentsnap_"
    ) as f:
        test_path = f.name
        f.write(test_code)

    step(f"Running: pytest {Path(test_path).name} -v")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", test_path, "-v", "--tb=short"],
            capture_output=True, text=True,
            env={**os.environ, "OPENROUTER_API_KEY": key},
        )
        print(result.stdout)
        if result.returncode == 0:
            print("  CI would pass on this PR.")
        else:
            print("  CI would FAIL on this PR.")
            if result.stderr:
                print(result.stderr[:500])
    finally:
        Path(test_path).unlink(missing_ok=True)
        # Cleanup snapshots
        for p in [snapshot_path(snap_name, SNAPSHOT_DIR),
                  last_run_path(snap_name, SNAPSHOT_DIR)]:
            p.unlink(missing_ok=True)


# -- Main ----------------------------------------------------------------------

if __name__ == "__main__":
    header("agentsnap real-API demo")

    # -- Zero instrumentation demo --------------------------------------------
    key = os.getenv("OPENROUTER_API_KEY")
    if key:
        zero_instrumentation_demo(key)
    else:
        header("ZERO INSTRUMENTATION (PatchSet)")
        print("  Skipped -- OPENROUTER_API_KEY not set")

    # -- Provider record/assert demos -----------------------------------------
    if key:
        openrouter_demo(key)
    else:
        print("[openrouter] skipped -- OPENROUTER_API_KEY not set")

    anthropic_demo()
    openai_demo()
    gemini_demo()

    # -- Three scenario demos (OpenRouter only) --------------------------------
    if not key:
        header("Scenarios skipped -- OPENROUTER_API_KEY not set")
        sys.exit(0)

    scenario_1_unintended_change(key)
    scenario_2_model_upgrade(key)
    scenario_3_ci(key)

    header("All scenarios complete")
