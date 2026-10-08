"""Unit tests for the dashboard's ``get_run_logs`` helper functions.

These check the pure logic extracted from the historical-log route.
"""

from __future__ import annotations

from types import SimpleNamespace

from modal_dojo._dashboard import (
    _compute_next_page,
    _parse_log_batches,
    _resolve_log_window,
    _to_timestamp,
)

NOW = 1_720_557_600.0


def _batch(task_id: str, items: list[SimpleNamespace]) -> SimpleNamespace:
    return SimpleNamespace(task_id=task_id, items=items)


def _item(
    data: str,
    *,
    file_descriptor: int = 0,
    timestamp: float = 0.0,
    timestamp_ns: int = 0,
) -> SimpleNamespace:
    return SimpleNamespace(
        data=data,
        file_descriptor=file_descriptor,
        timestamp=timestamp,
        timestamp_ns=timestamp_ns,
    )


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


# ── _to_timestamp ────────────────────────────────────────────────────────


def test_to_timestamp_whole_seconds() -> None:
    ts = _to_timestamp(1_720_557_600.0)
    assert ts.seconds == 1_720_557_600
    assert ts.nanos == 0


def test_to_timestamp_fractional_seconds() -> None:
    ts = _to_timestamp(1_720_557_600.5)
    assert ts.seconds == 1_720_557_600
    assert ts.nanos == 500_000_000


# ── _parse_log_batches ───────────────────────────────────────────────────


def test_parse_skips_empty_data_and_sets_core_fields() -> None:
    batches = [
        _batch(
            "task-a",
            [
                _item("hello", file_descriptor=1),
                _item(""),  # dropped
            ],
        )
    ]
    logs = _parse_log_batches(batches)
    assert logs == [
        {"task_id": "task-a", "line": "hello", "fd": 1, "ts": None, "ts_ns": None}
    ]


def test_parse_sets_timestamps_or_none_when_absent() -> None:
    batches = [
        _batch(
            "task-a",
            [
                _item("with-ts", timestamp=12.5, timestamp_ns=12_500_000_000),
                _item("no-ts"),
            ],
        )
    ]
    logs = _parse_log_batches(batches)
    assert logs[0] == {
        "task_id": "task-a",
        "line": "with-ts",
        "fd": 0,
        "ts": 12.5,
        "ts_ns": 12_500_000_000,
    }
    assert logs[1] == {
        "task_id": "task-a",
        "line": "no-ts",
        "fd": 0,
        "ts": None,
        "ts_ns": None,
    }


def test_parse_flattens_multiple_batches_in_order() -> None:
    batches = [
        _batch("task-a", [_item("a1"), _item("a2")]),
        _batch("task-b", [_item("b1")]),
    ]
    logs = _parse_log_batches(batches)
    assert [e["line"] for e in logs] == ["a1", "a2", "b1"]
    assert [e["task_id"] for e in logs] == ["task-a", "task-a", "task-b"]


# ── _compute_next_page ───────────────────────────────────────────────────


def test_next_page_partial_page_has_no_more() -> None:
    logs = [{"task_id": "t", "line": "x", "fd": 0, "ts_ns": 5_000_000_000}]
    assert _compute_next_page(logs, limit=100) == (False, None)


def test_next_page_empty_logs() -> None:
    assert _compute_next_page([], limit=100) == (False, None)


def test_next_page_full_page_uses_ts_ns_of_oldest() -> None:
    logs = [
        {"task_id": "t", "line": "x", "fd": 0, "ts_ns": 5_000_000_000},
        {"task_id": "t", "line": "y", "fd": 0, "ts_ns": 6_000_000_000},
    ]
    has_more, next_until = _compute_next_page(logs, limit=2)
    assert has_more is True
    assert next_until == (5_000_000_000 - 1) / 1_000_000_000


def test_next_page_falls_back_to_scaled_ts() -> None:
    logs = [
        {"task_id": "t", "line": "x", "fd": 0, "ts": 5.0},
        {"task_id": "t", "line": "y", "fd": 0, "ts": 6.0},
    ]
    has_more, next_until = _compute_next_page(logs, limit=2)
    assert has_more is True
    assert next_until == (5_000_000_000 - 1) / 1_000_000_000
