from __future__ import annotations

from typing import Callable


# ── Mock Anthropic primitives ──────────────────────────────────────────────────

class MockTextBlock:
    def __init__(self, text: str) -> None:
        self.text = text
        self.type = "text"


class MockUsage:
    def __init__(self, input_tokens: int = 10, output_tokens: int = 20) -> None:
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class MockToolUseBlock:
    def __init__(self, name: str, input: dict, id: str) -> None:
        self.type = "tool_use"
        self.name = name
        self.input = input
        self.id = id


class MockAnthropicResponse:
    def __init__(
        self,
        text: str,
        model: str = "claude-mock",
        tool_uses: list[tuple[str, dict]] | None = None,
    ) -> None:
        self.content = [MockTextBlock(text)]
        self.model = model
        self.usage = MockUsage()
        self._tool_uses = tool_uses or []
        for i, (name, input_dict) in enumerate(self._tool_uses):
            self.content.append(
                MockToolUseBlock(name, input_dict, id=f"toolu_mock{i}")
            )

    def model_dump(self, mode: str = "python") -> dict:
        """Anthropic-Message-shaped dict so recorded snapshots are replayable."""
        content = [{"type": "text", "text": self.content[0].text}]
        for i, (name, input_dict) in enumerate(self._tool_uses):
            content.append(
                {
                    "type": "tool_use",
                    "id": f"toolu_mock{i}",
                    "name": name,
                    "input": input_dict,
                }
            )
        return {
            "id": "msg_mock",
            "type": "message",
            "role": "assistant",
            "model": self.model,
            "content": content,
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {
                "input_tokens": self.usage.input_tokens,
                "output_tokens": self.usage.output_tokens,
            },
        }


class MockMessages:
    """Simulates client.messages with a pre-configured sequence of responses."""

    def __init__(self, responses: list[MockAnthropicResponse]) -> None:
        self._responses = list(responses)
        self._index = 0

    def create(self, **kwargs) -> MockAnthropicResponse:
        if self._index >= len(self._responses):
            raise ValueError(
                f"MockMessages exhausted: got {self._index + 1} calls, "
                f"only {len(self._responses)} responses configured."
            )
        resp = self._responses[self._index]
        self._index += 1
        return resp


class MockAnthropicClient:
    """Drop-in replacement for anthropic.Anthropic() with deterministic responses."""

    def __init__(self, responses: list[MockAnthropicResponse]) -> None:
        self.messages = MockMessages(responses)


# ── Mock Agents ────────────────────────────────────────────────────────────────

def SimpleToolAgent(client, search_tool: Callable, input_text: str) -> str:
    """
    Makes exactly one LLM call then one tool call ("search" with {"query": input_text}).
    Returns "Result: <tool_result>".

    client      — An AnthropicAdapter-wrapped MockAnthropicClient (or real client).
    search_tool — A ToolAdapter-wrapped callable that accepts query=str.
    """
    client.messages.create(
        model="claude-mock",
        messages=[{"role": "user", "content": input_text}],
        max_tokens=100,
    )
    tool_result = search_tool(query=input_text)
    return f"Result: {tool_result}"


def MultiStepAgent(
    client,
    fetch_tool: Callable,
    summarize_tool: Callable,
    input_text: str,
) -> str:
    """
    Makes two LLM calls and two tool calls in order: fetch → summarize.
    Returns "<fetch_result> | <summarize_result>".

    client         — An AnthropicAdapter-wrapped MockAnthropicClient (or real client).
    fetch_tool     — A ToolAdapter-wrapped callable that accepts query=str.
    summarize_tool — A ToolAdapter-wrapped callable that accepts content=str.
    """
    client.messages.create(
        model="claude-mock",
        messages=[{"role": "user", "content": input_text}],
        max_tokens=100,
    )
    fetch_result = fetch_tool(query=input_text)

    client.messages.create(
        model="claude-mock",
        messages=[
            {"role": "user", "content": input_text},
            {"role": "assistant", "content": f"Fetched: {fetch_result}"},
        ],
        max_tokens=100,
    )
    summarize_result = summarize_tool(content=fetch_result)

    return f"{fetch_result} | {summarize_result}"
