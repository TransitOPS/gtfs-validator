"""Unit tests for trip_calendar_utils shared helpers."""

from __future__ import annotations

from datetime import date, timedelta

import polars as pl

from gtfs_validator.trip_calendar_utils import (
    compute_majority_service_coverage,
    count_trips_per_service_date,
)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _dur(hours: int, minutes: int = 0, seconds: int = 0) -> timedelta:
    """Create a timedelta representing a GTFS TIME value."""
    return timedelta(hours=hours, minutes=minutes, seconds=seconds)


def _make_trips(rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(rows, schema={"trip_id": pl.Utf8, "service_id": pl.Utf8})


def _make_frequencies(rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(
        rows,
        schema={
            "trip_id": pl.Utf8,
            "start_time": pl.Duration("ms"),
            "end_time": pl.Duration("ms"),
            "headway_secs": pl.Int64,
        },
    )


# ------------------------------------------------------------------
# Tests for count_trips_per_service_date
# ------------------------------------------------------------------


def test_count_trips_basic():
    """H1: 2 trips on 1 service_id with 3 active dates -> each date has count=2."""
    service_date_map = {
        "s1": {date(2022, 12, 1), date(2022, 12, 2), date(2022, 12, 3)},
    }
    feed = {
        "trips": _make_trips([
            {"trip_id": "t1", "service_id": "s1"},
            {"trip_id": "t2", "service_id": "s1"},
        ]),
    }
    result = count_trips_per_service_date(feed, service_date_map)
    assert len(result) == 3
    for d in service_date_map["s1"]:
        assert result[d] == 2


def test_count_trips_with_frequencies():
    """H2: 1 trip with frequency (06:00-08:00, headway=1800) -> 4 effective trips."""
    service_date_map = {
        "s1": {date(2022, 12, 1), date(2022, 12, 2)},
    }
    feed = {
        "trips": _make_trips([{"trip_id": "t1", "service_id": "s1"}]),
        "frequencies": _make_frequencies([
            {
                "trip_id": "t1",
                "start_time": _dur(6),
                "end_time": _dur(8),
                "headway_secs": 1800,
            },
        ]),
    }
    result = count_trips_per_service_date(feed, service_date_map)
    # 1 + floor((7200 - 1) / 1800) = 1 + 3 = 4
    assert result[date(2022, 12, 1)] == 4
    assert result[date(2022, 12, 2)] == 4


def test_count_trips_empty_map():
    """H3: Empty service_date_map -> empty result even with trips."""
    feed = {
        "trips": _make_trips([{"trip_id": "t1", "service_id": "s1"}]),
    }
    result = count_trips_per_service_date(feed, {})
    assert result == {}


# ------------------------------------------------------------------
# Tests for compute_majority_service_coverage
# ------------------------------------------------------------------


def test_majority_coverage_basic():
    """H4: 4 high-count dates + 1 low-count -> window excludes low-count date."""
    d1 = date(2022, 12, 1)
    d2 = date(2022, 12, 2)
    d3 = date(2022, 12, 3)
    d4 = date(2022, 12, 4)
    d5 = date(2022, 12, 5)
    counts = {d1: 100, d2: 100, d3: 100, d4: 100, d5: 10}
    result = compute_majority_service_coverage(counts)
    assert result is not None
    # threshold = int(0.75 * 100) = 75. d5 has 10 < 75 so excluded.
    assert result == (d1, d4)


def test_majority_coverage_empty():
    """H5: Empty dict -> None."""
    assert compute_majority_service_coverage({}) is None


def test_majority_coverage_single_date():
    """H6: Single date -> (d1, d1)."""
    d1 = date(2022, 12, 1)
    result = compute_majority_service_coverage({d1: 50})
    assert result == (d1, d1)


def test_majority_coverage_all_equal():
    """H7: 10 dates all with count=100 -> window spans all 10."""
    base = date(2022, 12, 1)
    counts = {base + timedelta(days=i): 100 for i in range(10)}
    result = compute_majority_service_coverage(counts)
    assert result is not None
    assert result == (base, base + timedelta(days=9))
