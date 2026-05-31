"""
demo_real.py -- Run agentsnap with real API calls.

Set whichever API keys you have. Providers without a key are skipped.

Required env vars (set whichever you have):
    ANTHROPIC_API_KEY
    OPENAI_API_KEY
    GEMINI_API_KEY          (Google AI Studio -> https://aistudio.google.com/apikey)
    COHERE_API_KEY
    MISTRAL_API_KEY
    GROQ_API_KEY

Run:
    python examples/demo_real.py

Snapshots are written to __agent_snapshots__/ and committed to git.
Run a second time to assert against them.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Add project root to path when running as a script
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load .env from the project root if present
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    pass  # python-dotenv not installed, rely on shell env vars

from agentsnap.adapters.tool import ToolAdapter
from agentsnap.core.asserter import AgentAsserter
from agentsnap.core.recorder import AgentRecorder
from agentsnap.exceptions import AgentRegressionError, SnapshotNotFoundError

SNAPSHOT_DIR = "__agent_snapshots__"
SEPARATOR = "-" * 60


# -- Shared tool ---------------------------------------------------------------


def lookup(query: str) -> str:
    """Simulates a knowledge-base lookup. Replace with a real tool."""
    data = {
        "agentsnap": "A deterministic snapshot testing harness for AI agents.",
        "python": "A high-level, interpreted programming language.",
        "default": f"No entry found for '{query}'.",
    }
    return data.get(query.lower(), data["default"])


def header(title: str) -> None:
    print(f"\n{SEPARATOR}\n  {title}\n{SEPARATOR}")


def run_provider(name: str, make_client, call_agent) -> None:
    """Record on first run, assert on subsequent runs."""
    tool = ToolAdapter(lookup, name="lookup")
    snap_name = f"real_{name}"

    try:
        # Try asserting first (snapshot already exists)
        print(f"[{name}] snapshot found -- asserting...")
        with AgentAsserter(snap_name, snapshot_dir=SNAPSHOT_DIR) as a:
            client = make_client()
            a.output = call_agent(
                client, ToolAdapter(lookup, name="lookup"), "agentsnap"
            )
        print(f"[{name}] OK no regression")

    except SnapshotNotFoundError:
        # First run -- record the golden snapshot
        print(f"[{name}] no snapshot yet -- recording golden run...")
        try:
            with AgentRecorder(snap_name, snapshot_dir=SNAPSHOT_DIR, model=name) as rec:
                client = make_client()
                rec.input_data = {"query": "agentsnap"}
                rec.output = call_agent(client, tool, "agentsnap")
            print(f"[{name}] OK snapshot written -> {snap_name}.json")
            print(f"[{name}]   commit it: git add {SNAPSHOT_DIR}/{snap_name}.json")
        except Exception as e:
            print(f"[{name}] ERROR during record: {type(e).__name__}: {e}")

    except AgentRegressionError as e:
        print(f"[{name}] FAIL REGRESSION DETECTED")
        print(f"         Failed checks: {e.diff_report.failed_checks}")
        if e.diff_report.structural_diff:
            print(f"         Structural: {e.diff_report.structural_diff}")
        for k, v in e.diff_report.semantic_scores.items():
            print(f"         Semantic [{k}]: {v:.4f}")
        print(f"         To approve: python -m agentsnap.cli update {snap_name}")

    except Exception as e:
        print(f"[{name}] ERROR: {type(e).__name__}: {e}")


# -- Anthropic -----------------------------------------------------------------


def anthropic_demo() -> None:
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        print("[anthropic] skipped -- ANTHROPIC_API_KEY not set")
        return
    try:
        import anthropic
        from agentsnap.adapters.anthropic import AnthropicAdapter
    except ImportError:
        print("[anthropic] skipped -- pip install anthropic")
        return

    def make_client():
        return AnthropicAdapter(anthropic.Anthropic(api_key=key))

    def call_agent(client, tool, query: str) -> str:
        client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
            messages=[{"role": "user", "content": f"Look up: {query}"}],
        )
        result = tool(query=query)
        return f"Answer: {result}"

    run_provider("anthropic", make_client, call_agent)


# -- OpenAI --------------------------------------------------------------------


def openai_demo() -> None:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        print("[openai] skipped -- OPENAI_API_KEY not set")
        return
    try:
        import openai
        from agentsnap.adapters.openai import OpenAIAdapter
    except ImportError:
        print("[openai] skipped -- pip install openai")
        return

    def make_client():
        return OpenAIAdapter(openai.OpenAI(api_key=key))

    def call_agent(client, tool, query: str) -> str:
        client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=1024,
            messages=[{"role": "user", "content": f"Look up: {query}"}],
        )
        result = tool(query=query)
        return f"Answer: {result}"

    run_provider("openai", make_client, call_agent)


# -- Google Gemini -------------------------------------------------------------


def gemini_demo() -> None:
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        print("[gemini] skipped -- GEMINI_API_KEY not set")
        return
    try:
        from google import genai
        from agentsnap.adapters.google import GeminiAdapter
    except ImportError:
        print("[gemini] skipped -- pip install google-genai")
        return

    def make_client():
        return GeminiAdapter(genai.Client(api_key=key))

    def call_agent(client, tool, query: str) -> str:
        client.models.generate_content(
            model="gemini-2.0-flash",
            contents=f"Look up: {query}",
        )
        result = tool(query=query)
        return f"Answer: {result}"

    run_provider("gemini", make_client, call_agent)


# -- Cohere --------------------------------------------------------------------


def cohere_demo() -> None:
    key = os.getenv("COHERE_API_KEY")
    if not key:
        print("[cohere] skipped -- COHERE_API_KEY not set")
        return
    try:
        import cohere
        from agentsnap.adapters.cohere import CohereAdapter
    except ImportError:
        print("[cohere] skipped -- pip install cohere")
        return

    def make_client():
        return CohereAdapter(cohere.ClientV2(api_key=key))

    def call_agent(client, tool, query: str) -> str:
        client.chat(
            model="command-r-plus",
            messages=[{"role": "user", "content": f"Look up: {query}"}],
        )
        result = tool(query=query)
        return f"Answer: {result}"

    run_provider("cohere", make_client, call_agent)


# -- Mistral -------------------------------------------------------------------


def mistral_demo() -> None:
    key = os.getenv("MISTRAL_API_KEY")
    if not key:
        print("[mistral] skipped -- MISTRAL_API_KEY not set")
        return
    try:
        from mistralai import Mistral
        from agentsnap.adapters.mistral import MistralAdapter
    except ImportError:
        print("[mistral] skipped -- pip install mistralai")
        return

    def make_client():
        return MistralAdapter(Mistral(api_key=key))

    def call_agent(client, tool, query: str) -> str:
        client.chat.complete(
            model="mistral-large-latest",
            messages=[{"role": "user", "content": f"Look up: {query}"}],
        )
        result = tool(query=query)
        return f"Answer: {result}"

    run_provider("mistral", make_client, call_agent)


# -- Groq ----------------------------------------------------------------------


def groq_demo() -> None:
    key = os.getenv("GROQ_API_KEY")
    if not key:
        print("[groq] skipped -- GROQ_API_KEY not set")
        return
    try:
        from groq import Groq
        from agentsnap.adapters.groq import GroqAdapter
    except ImportError:
        print("[groq] skipped -- pip install groq")
        return

    def make_client():
        return GroqAdapter(Groq(api_key=key))

    def call_agent(client, tool, query: str) -> str:
        client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            max_tokens=1024,
            messages=[{"role": "user", "content": f"Look up: {query}"}],
        )
        result = tool(query=query)
        return f"Answer: {result}"

    run_provider("groq", make_client, call_agent)


# -- OpenRouter ----------------------------------------------------------------


def openrouter_demo() -> None:
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        print("[openrouter] skipped -- OPENROUTER_API_KEY not set")
        return
    try:
        import openai
        from agentsnap.adapters.openrouter import OpenRouterAdapter, OPENROUTER_BASE_URL
    except ImportError:
        print("[openrouter] skipped -- pip install openai")
        return

    def make_client():
        return OpenRouterAdapter(
            openai.OpenAI(api_key=key, base_url=OPENROUTER_BASE_URL)
        )

    def call_agent(client, tool, query: str) -> str:
        client.chat.completions.create(
            model="google/gemini-3.5-flash",
            max_tokens=1024,
            messages=[{"role": "user", "content": f"Look up: {query}"}],
        )
        result = tool(query=query)
        return f"Answer: {result}"

    run_provider("openrouter", make_client, call_agent)


# -- Main ----------------------------------------------------------------------

if __name__ == "__main__":
    header("agentsnap real-API demo")
    print("Snapshots dir:", SNAPSHOT_DIR)
    print("Run once to record, run again to assert.\n")

    anthropic_demo()
    openai_demo()
    gemini_demo()
    cohere_demo()
    mistral_demo()
    groq_demo()
    openrouter_demo()

    header("Done")
    snaps = list(Path(SNAPSHOT_DIR).glob("real_*.json"))
    if snaps:
        print(f"Recorded snapshots ({len(snaps)}):")
        for p in sorted(snaps):
            print(f"  {p.name}")
        print("\nCommit them:")
        print(
            f"  git add {SNAPSHOT_DIR}/real_*.json && git commit -m 'feat: add golden snapshots'"
        )
    else:
        print("No snapshots written (set at least one API key env var).")
