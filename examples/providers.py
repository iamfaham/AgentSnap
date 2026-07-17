"""
providers.py -- The non-core provider adapters: Gemini, Cohere, Mistral, Groq.

One aspect only: each of these four adapters wraps the same record/assert/
regression story as the core Anthropic/OpenAI adapters, using each SDK's own
response shape.

Gemini, Cohere, and Mistral are **live-mode only** today: passing
``mode="replay"`` raises ``ReplayError`` (see ``agentsnap/adapters/{google,cohere,mistral}.py``).
Groq subclasses ``OpenAIAdapter`` (it's an OpenAI-compatible interface), so it
inherits full streaming and replay support for free -- it's grouped here
because it's still a "less-core" provider, not because it shares that
limitation.

Usage:
    python examples/providers.py             # mock only, no keys/network needed
    python examples/providers.py --real      # mock, then one tiny real call per
                                               # provider key present (GEMINI_API_KEY
                                               # or GOOGLE_API_KEY, COHERE_API_KEY,
                                               # MISTRAL_API_KEY, GROQ_API_KEY);
                                               # absent keys print a skip hint and
                                               # exit 0 rather than fail
    python examples/providers.py --keep      # keep the temp snapshot dir, print its path

The journey (mock_demo), once per provider:
  1. Record a golden run through the adapter.
  2. An identical run -- passes.
  3. One regression, shown once (on the last provider) so the story doesn't
     repeat itself four times: the agent's output drifts and is caught.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import _common as ex
from agentsnap.adapters.cohere import CohereAdapter
from agentsnap.adapters.google import GeminiAdapter
from agentsnap.adapters.groq import GroqAdapter
from agentsnap.adapters.mistral import MistralAdapter
from agentsnap.adapters.tool import ToolAdapter
from agentsnap.core.asserter import AgentAsserter
from agentsnap.exceptions import AgentRegressionError

QUERY = "What is agentsnap?"


def lookup(query: str) -> str:
    return f"mock result for '{query}'"


# ---------------------------------------------------------------------------
# Mock response shapes (ported from the retired demo_mock.py -- these
# providers' real SDKs aren't installed by default, so there's no pydantic
# schema to validate a mock against; these are the minimal duck-typed shapes
# each adapter actually reads off the response.)
# ---------------------------------------------------------------------------

class _GeminiResponse:
    def __init__(self, text: str) -> None:
        self.text = text
        self.usage_metadata = type("U", (), {"total_token_count": 30})()


class _CohereBlock:
    def __init__(self, text: str) -> None:
        self.text = text


class _CohereResponse:
    def __init__(self, text: str) -> None:
        self.message = type("M", (), {"content": [_CohereBlock(text)]})()
        self.usage = type("U", (), {"input_tokens": 10, "output_tokens": 20})()


class _MistralResponse:
    def __init__(self, text: str) -> None:
        message = type("M", (), {"content": text, "role": "assistant"})()
        self.choices = [type("C", (), {"message": message})()]
        self.usage = type("U", (), {"total_tokens": 30})()


class MockGeminiClient:
    def __init__(self, response) -> None:
        self._response = response
        self.models = self

    def generate_content(self, model, contents, **kwargs):
        return self._response


class MockCohereClient:
    def __init__(self, response) -> None:
        self._response = response

    def chat(self, **kwargs):
        return self._response


class MockMistralClient:
    def __init__(self, response) -> None:
        self._response = response
        self.chat = self

    def complete(self, **kwargs):
        return self._response


# ---------------------------------------------------------------------------
# Per-provider adapter + call wiring
# ---------------------------------------------------------------------------

def _gemini_call(client, tool):
    client.models.generate_content(model="gemini-2.0-flash", contents=QUERY)
    return f"Result: {tool(query=QUERY)}"


def _cohere_call(client, tool):
    client.chat(model="command-r-plus", messages=[{"role": "user", "content": QUERY}])
    return f"Result: {tool(query=QUERY)}"


def _mistral_call(client, tool):
    client.chat.complete(model="mistral-large-latest", messages=[{"role": "user", "content": QUERY}])
    return f"Result: {tool(query=QUERY)}"


def _groq_call(client, tool):
    client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": QUERY}],
        max_tokens=50,
    )
    return f"Result: {tool(query=QUERY)}"


PROVIDERS = [
    ("gemini", lambda text: GeminiAdapter(MockGeminiClient(_GeminiResponse(text))), _gemini_call),
    ("cohere", lambda text: CohereAdapter(MockCohereClient(_CohereResponse(text))), _cohere_call),
    ("mistral", lambda text: MistralAdapter(MockMistralClient(_MistralResponse(text))), _mistral_call),
    (
        "groq",
        lambda text: GroqAdapter(ex.MockSequence([ex.make_openai_chat_completion(text)], provider="openai")),
        _groq_call,
    ),
]


def mock_demo(snapshot_dir: str) -> None:
    ex.header("PROVIDERS (mock)  --  Gemini / Cohere / Mistral / Groq adapters")
    print("  Same record -> assert -> regression story, one per non-core adapter.\n")

    for provider, make_client, call_fn in PROVIDERS:
        name = f"providers_{provider}"
        tool = ToolAdapter(lookup, name="lookup")

        ex.subheader(f"{provider}  --  record the golden run")
        with AgentAsserter(name, snapshot_dir=snapshot_dir, embed_fn=ex.demo_embed) as a:
            a.output = call_fn(make_client("I'll look that up."), tool)
        print(f"  [{provider}] golden written -> {name}.json")

        ex.subheader(f"{provider}  --  identical run")
        with AgentAsserter(name, snapshot_dir=snapshot_dir, embed_fn=ex.demo_embed) as a:
            a.output = call_fn(make_client("I'll look that up."), tool)
        print(f"  [{provider}] PASSED")

    ex.subheader("groq  --  regression: the agent's output drifts")
    try:
        with AgentAsserter("providers_groq", snapshot_dir=snapshot_dir, embed_fn=ex.demo_embed) as a:
            a.output = "drifted output -- agent changed its answer"
        print("  ERROR: should have failed!")
    except AgentRegressionError as e:
        print(f"  [groq] caught: {e.diff_report.failed_checks}")

    ex.header("Done -- all four provider adapters record, assert, and catch drift.")


# ---------------------------------------------------------------------------
# Real calls: one tiny call per provider key present
# ---------------------------------------------------------------------------

def _real_gemini(snapshot_dir: str) -> bool:
    ex.subheader("gemini")
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        print("  skipped -- set GEMINI_API_KEY or GOOGLE_API_KEY to run this provider")
        return False
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        print("  skipped -- pip install google-genai")
        return False

    client = GeminiAdapter(genai.Client(api_key=key))
    tool = ToolAdapter(lookup, name="lookup")
    name = "providers_gemini_real"
    try:
        with AgentAsserter(name, snapshot_dir=snapshot_dir, embed_fn=ex.demo_embed) as a:
            client.models.generate_content(
                model="gemini-2.0-flash",
                contents="Look up: agentsnap",
                config=types.GenerateContentConfig(
                    temperature=ex.REAL_TEMPERATURE, max_output_tokens=ex.REAL_MAX_TOKENS
                ),
            )
            a.output = f"Result: {tool(query='agentsnap')}"
        print(f"  recorded + asserted -> {name}.json")
    except AgentRegressionError as e:
        print(f"  regression: {e.diff_report.failed_checks}")
    return True


def _real_cohere(snapshot_dir: str) -> bool:
    ex.subheader("cohere")
    key = os.getenv("COHERE_API_KEY")
    if not key:
        print("  skipped -- set COHERE_API_KEY to run this provider")
        return False
    try:
        import cohere
    except ImportError:
        print("  skipped -- pip install cohere")
        return False

    client = CohereAdapter(cohere.ClientV2(api_key=key))
    tool = ToolAdapter(lookup, name="lookup")
    name = "providers_cohere_real"
    try:
        with AgentAsserter(name, snapshot_dir=snapshot_dir, embed_fn=ex.demo_embed) as a:
            client.chat(
                model="command-r-plus",
                messages=[{"role": "user", "content": "Look up: agentsnap"}],
                temperature=ex.REAL_TEMPERATURE,
                max_tokens=ex.REAL_MAX_TOKENS,
            )
            a.output = f"Result: {tool(query='agentsnap')}"
        print(f"  recorded + asserted -> {name}.json")
    except AgentRegressionError as e:
        print(f"  regression: {e.diff_report.failed_checks}")
    return True


def _real_mistral(snapshot_dir: str) -> bool:
    ex.subheader("mistral")
    key = os.getenv("MISTRAL_API_KEY")
    if not key:
        print("  skipped -- set MISTRAL_API_KEY to run this provider")
        return False
    try:
        from mistralai import Mistral
    except ImportError:
        print("  skipped -- pip install mistralai")
        return False

    client = MistralAdapter(Mistral(api_key=key))
    tool = ToolAdapter(lookup, name="lookup")
    name = "providers_mistral_real"
    try:
        with AgentAsserter(name, snapshot_dir=snapshot_dir, embed_fn=ex.demo_embed) as a:
            client.chat.complete(
                model="mistral-large-latest",
                messages=[{"role": "user", "content": "Look up: agentsnap"}],
                temperature=ex.REAL_TEMPERATURE,
                max_tokens=ex.REAL_MAX_TOKENS,
            )
            a.output = f"Result: {tool(query='agentsnap')}"
        print(f"  recorded + asserted -> {name}.json")
    except AgentRegressionError as e:
        print(f"  regression: {e.diff_report.failed_checks}")
    return True


def _real_groq(snapshot_dir: str) -> bool:
    ex.subheader("groq")
    key = os.getenv("GROQ_API_KEY")
    if not key:
        print("  skipped -- set GROQ_API_KEY to run this provider")
        return False
    try:
        from groq import Groq
    except ImportError:
        print("  skipped -- pip install groq")
        return False

    client = GroqAdapter(Groq(api_key=key))
    tool = ToolAdapter(lookup, name="lookup")
    name = "providers_groq_real"
    try:
        with AgentAsserter(name, snapshot_dir=snapshot_dir, embed_fn=ex.demo_embed) as a:
            client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": "Look up: agentsnap"}],
                temperature=ex.REAL_TEMPERATURE,
                max_tokens=ex.REAL_MAX_TOKENS,
            )
            a.output = f"Result: {tool(query='agentsnap')}"
        print(f"  recorded + asserted -> {name}.json")
    except AgentRegressionError as e:
        print(f"  regression: {e.diff_report.failed_checks}")
    return True


def real_demo(snapshot_dir: str) -> None:
    ex.maybe_load_dotenv()
    ex.header("PROVIDERS (real)  --  one tiny call per provider key present")
    print("  Gemini/Cohere/Mistral are live-mode only (mode=\"replay\" raises ReplayError).")
    print("  Groq subclasses OpenAIAdapter, so it also gets replay/streaming for free.\n")

    ran_any = False
    for fn in (_real_gemini, _real_cohere, _real_mistral, _real_groq):
        ran_any = fn(snapshot_dir) or ran_any

    if not ran_any:
        print(
            "\n  No provider keys found -- set any of GEMINI_API_KEY/GOOGLE_API_KEY, "
            "COHERE_API_KEY, MISTRAL_API_KEY, or GROQ_API_KEY to exercise --real."
        )


def main() -> None:
    args = ex.parse_args(__doc__)
    with ex.temp_snapshot_dir(keep=args.keep) as snapshot_dir:
        if args.keep:
            print(f"Snapshot dir: {snapshot_dir}")
        mock_demo(snapshot_dir)
        if args.real:
            real_demo(snapshot_dir)
    ex.header("Providers complete")


if __name__ == "__main__":
    main()
