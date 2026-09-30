"""Shared parsing helpers for user-supplied time bounds."""

from __future__ import annotations

import math
import re
from datetime import UTC, datetime


_RELATIVE_SECONDS = {
    "s": 1,
    "m": 60,
    "h": 60 * 60,
    "d": 24 * 60 * 60,
    "w": 7 * 24 * 60 * 60,
}
_RELATIVE_RE = re.compile(r"(\d+)\s*([smhdw])", re.IGNORECASE)


def parse_time(value: str, now: float) -> float | None:
    """Parse relative age, epoch seconds, or ISO 8601 into epoch seconds."""
    text = (value or "").strip()
    if not text:
        return None

    for parser in (_parse_relative, _parse_epoch, _parse_iso):
        result = parser(text, now)
        if result is not None:
            return result
    return None


def _parse_relative(text: str, now: float) -> float | None:
    match = _RELATIVE_RE.fullmatch(text)
    if not match:
        return None
    amount = int(match.group(1))
    unit = match.group(2).lower()
    return now - amount * _RELATIVE_SECONDS[unit]


def _parse_epoch(text: str, _now: float) -> float | None:
    try:
        parsed = float(text)
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


def _parse_iso(text: str, _now: float) -> float | None:
    normalized = text[:-1] + "+00:00" if text.endswith(("Z", "z")) else text
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.timestamp()
