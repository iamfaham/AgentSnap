from __future__ import annotations

DEFAULT_VOLATILE_FIELDS: frozenset[str] = frozenset({
    "tokens",
    "recorded_at",
    "timestamp",
    "request_id",
    "latency_ms",
    "raw_response",
})


def normalize_event(event: dict, ignore_fields: "frozenset[str] | set[str]") -> dict:
    return {k: v for k, v in event.items() if k not in ignore_fields}


def normalize_trace(
    trace: list[dict],
    ignore_fields: "frozenset[str] | set[str]",
) -> list[dict]:
    return [normalize_event(e, ignore_fields) for e in trace]
