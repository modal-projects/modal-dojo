from datetime import UTC, datetime

import pytest

from modal_dojo.common.time import parse_time


NOW = datetime(2026, 7, 9, 18, 0, 0, tzinfo=UTC).timestamp()


@pytest.mark.parametrize("value", ["", "   ", "\t\n"])
def test_parse_time_empty_or_blank_returns_none(value):
    assert parse_time(value, NOW) is None


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("45s", NOW - 45),
        ("30m", NOW - 30 * 60),
        ("2h", NOW - 2 * 60 * 60),
        ("1d", NOW - 24 * 60 * 60),
        ("1w", NOW - 7 * 24 * 60 * 60),
        ("0m", NOW),
        ("30 m", NOW - 30 * 60),
    ],
)
def test_parse_time_relative_age_subtracts_from_now(value, expected):
    assert parse_time(value, NOW) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("1720557600", 1720557600.0),
        ("1720557600.5", 1720557600.5),
        ("  1720557600  ", 1720557600.0),
    ],
)
def test_parse_time_epoch_seconds(value, expected):
    assert parse_time(value, NOW) == expected


def test_parse_time_iso_with_trailing_z_is_utc():
    assert parse_time("2026-07-09T18:00:00Z", NOW) == NOW


def test_parse_time_iso_naive_is_read_as_utc():
    assert parse_time("2026-07-09T18:00:00", NOW) == NOW


def test_parse_time_iso_with_explicit_offset_is_honored():
    expected = datetime(2026, 7, 9, 22, 0, 0, tzinfo=UTC).timestamp()
    assert parse_time("2026-07-09T18:00:00-04:00", NOW) == expected


@pytest.mark.parametrize(
    "value",
    [
        "not-a-time",
        "30x",
        "2026-13-01T00:00:00Z",
        "yesterday",
        "-5m",
        "inf",
        "nan",
        "Infinity",
        "1e999",
    ],
)
def test_parse_time_unparseable_returns_none(value):
    assert parse_time(value, NOW) is None
