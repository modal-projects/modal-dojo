"""Unit tests for the dashboard's ``get_run_logs`` helper functions.

These check the pure logic extracted from the historical-log route.
"""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

from modal_dojo._dashboard import (
    _log_entry_dict,
    _resolve_log_window,
)

NOW = 1_720_557_600.0


# ── _resolve_log_window ──────────────────────────────────────────────────


def test_window_defaults_when_bounds_unset() -> None:
    since_ts, until_ts = _resolve_log_window(
        None,
        None,
        default_since=100.0,
        default_until=900.0,
        now=NOW,
    )
    assert since_ts == 100.0
    assert until_ts == 900.0


def test_window_until_defaults_to_now_when_default_nonpositive() -> None:
    _, until_ts = _resolve_log_window(
        None,
        None,
        default_since=100.0,
        default_until=0.0,
        now=NOW,
    )
    assert until_ts == NOW


def test_window_nonpositive_explicit_until_becomes_now() -> None:
    _, until_ts = _resolve_log_window(
        1000.0,
        -5.0,
        default_since=100.0,
        default_until=900.0,
        now=NOW,
    )
    assert until_ts == NOW


def test_window_negative_since_is_floored_at_zero() -> None:
    since_ts, _ = _resolve_log_window(
        -30.0,
        1000.0,
        default_since=100.0,
        default_until=900.0,
        now=NOW,
    )
    assert since_ts == 0.0


def test_window_explicit_bounds_pass_through() -> None:
    since_ts, until_ts = _resolve_log_window(
        200.0,
        1000.0,
        default_since=100.0,
        default_until=900.0,
        now=NOW,
    )
    assert since_ts == 200.0
    assert until_ts == 1000.0


# ── _log_entry_dict ──────────────────────────────────────────────────────


def _entry(
    message: str,
    *,
    source: str = "stdout",
    ts: float = 12.5,
    context_ids: list[str] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        message=message,
        source=source,
        timestamp=datetime.fromtimestamp(ts, tz=timezone.utc),
        context_ids=context_ids
        if context_ids is not None
        else ["fu-x", "fc-y", "task-a"],
    )


def test_log_entry_maps_fields() -> None:
    entry = _log_entry_dict(_entry("hello"))
    assert entry == {
        "task_id": "task-a",
        "line": "hello",
        "fd": 1,
        "ts": 12.5,
        "ts_ns": 12_500_000_000,
    }


def test_log_entry_task_id_uses_last_context_id() -> None:
    assert _log_entry_dict(_entry("x", context_ids=["ta-1"]))["task_id"] == "ta-1"
    assert _log_entry_dict(_entry("x", context_ids=[]))["task_id"] == ""


def test_log_entry_source_to_fd() -> None:
    assert _log_entry_dict(_entry("x", source="stderr"))["fd"] == 2
    assert _log_entry_dict(_entry("x", source="system"))["fd"] == 3
    assert _log_entry_dict(_entry("x", source="other"))["fd"] == 0
