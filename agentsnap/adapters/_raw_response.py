from __future__ import annotations

import warnings


def unwrap_legacy_response(response):
    """Unwrap an SDK ``LegacyAPIResponse`` into the parsed model it wraps.

    Some callers (e.g. langchain-openai, langchain-anthropic — both sync and
    async) request the raw HTTP response via the SDK's
    ``with_raw_response``/``X-Stainless-Raw-Response`` mechanism so they can
    read rate-limit headers, then call ``.parse()`` themselves to get the
    real parsed object (``ChatCompletion``/``Response``/``Message``). Our
    interceptor still needs the parsed object to extract text/tokens/
    tool_requests, but must return the original (possibly-legacy) object to
    the caller unchanged so their own ``.parse()``/header access keeps
    working.

    This check is provider-agnostic: a real (already-parsed) response never
    exposes a callable ``.parse`` of its own, so identity is returned unless
    one is present.
    """
    parse = getattr(response, "parse", None)
    if not callable(parse):
        return response
    try:
        return parse()
    except Exception:
        warnings.warn(
            "agentsnap: failed to unwrap a raw-response wrapper; "
            "this call may record an empty response",
            stacklevel=2,
        )
        return response


def wants_raw_response(kwargs: dict) -> bool:
    """True if the caller requested the SDK's raw-response wrapper.

    langchain-openai and langchain-anthropic (both sync and async) call
    ``with_raw_response.create()``/``.parse()`` instead of ``create()``
    directly, which the SDK implements by stamping a special header on the
    request and having the transport return a ``LegacyAPIResponse`` wrapper
    whose ``.parse()`` yields the real object.
    """
    extra_headers = kwargs.get("extra_headers") or {}
    return extra_headers.get("X-Stainless-Raw-Response") == "true"


class ReplayLegacyResponse:
    """Minimal stand-in for an SDK ``LegacyAPIResponse`` during replay.

    In replay mode no HTTP call is made, so there's no real LegacyAPIResponse
    for a raw-response caller (e.g. langchain-openai's/langchain-anthropic's
    ``with_raw_response``) to call ``.parse()`` on. This thin wrapper
    supplies just enough of that surface: ``.parse()`` returns the
    reconstructed response, and any other attribute access forwards to it.
    """

    def __init__(self, parsed) -> None:
        self._parsed = parsed

    def parse(self):
        return self._parsed

    def __getattr__(self, name):
        return getattr(self._parsed, name)


class RawResponseStreamShim:
    """Mimics the SDK's raw-response wrapper around a (teed) stream.

    When a caller uses ``with_raw_response`` on a streaming call (e.g.
    langchain-openai, langchain-anthropic), the SDK returns a
    ``LegacyAPIResponse`` whose ``.parse()`` yields the real ``Stream``. Our
    recording tee wraps the real stream, so a raw-response caller needs this
    shim in front of it: ``.parse()`` returns the tee (so it still gets
    recorded), and every other attribute (headers, etc.) forwards to the
    original ``LegacyAPIResponse`` from the real HTTP call.
    """

    def __init__(self, tee, legacy) -> None:
        self._tee = tee
        self._legacy = legacy  # the original LegacyAPIResponse (real headers etc.)

    def parse(self):
        return self._tee

    def __getattr__(self, name):
        return getattr(self._legacy, name)
