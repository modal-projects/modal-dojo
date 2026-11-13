"""Unit tests for the dashboard's ``get_run_logs`` helper functions.

These check the pure logic extracted from the historical-log route.
"""

from __future__ import annotations

from datetime import datetime, timezone

from modal.types import LogEntry

from modal_dojo._dashboard import _log_entry, _resolve_log_window

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


# ── _log_entry ───────────────────────────────────────────────────────


def _modal_entry(context_ids: list[str]) -> LogEntry:
    return LogEntry(
        message="hello\n",
        timestamp=datetime.fromtimestamp(12.5, timezone.utc),
        source="stderr",
        object_id="ap-1",
        context_ids=context_ids,
    )


def test_log_entry_uses_narrowest_context_id() -> None:
    entry = _log_entry(_modal_entry(["fu-1", "fc-1", "in-1", "ta-1"]))
    assert entry == {
        "task_id": "ta-1",
        "line": "hello\n",
        "source": "stderr",
        "ts": 12.5,
    }


def test_log_entry_without_context_ids() -> None:
    assert _log_entry(_modal_entry([]))["task_id"] == ""
