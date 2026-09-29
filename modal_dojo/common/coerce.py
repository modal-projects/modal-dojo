"""Small value-coercion helpers shared across the package."""

from __future__ import annotations

from typing import Any


def safe_int(value: Any) -> int:
    """Best-effort int coercion; returns 0 for None / unparseable values."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def optional_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None
