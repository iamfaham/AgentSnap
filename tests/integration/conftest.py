"""Shared fixtures for integration tests."""
from __future__ import annotations

import unittest.mock as mock

import pytest
from anthropic.resources.messages.messages import Messages as _AnthMessages


class _AnthContent:
    text = "zero-instrument anthropic"


class _AnthResp:
    content = [_AnthContent()]

    class usage:
        input_tokens = 5
        output_tokens = 10


@pytest.fixture
def mock_anthropic_messages():
    """Patch Messages.create with a fake response BEFORE PatchSet wraps it.

    Requesting this fixture before agentsnap_instrument in the test signature
    ensures that PatchSet captures the mock as its `original`, so calls flow
    through PatchSet's interceptor -> mock -> fake response.
    """
    with mock.patch.object(_AnthMessages, "create", return_value=_AnthResp()):
        yield
