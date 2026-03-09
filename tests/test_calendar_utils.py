"""Tests for calendar_utils module."""

from __future__ import annotations

from datetime import date

import polars as pl

from gtfs_validator.calendar_utils import (
    ServiceIdIntersectionCache,
    build_service_date_map,
)


def _make_calendar(
    service_id: str,
    start_date: date,
    end_date: date,
    days: str = "1111100",
) -> pl.DataFrame:
    """Create a 1-row calendar DataFrame.

    days is a 7-char string for mon-sun (1=active, 0=inactive).
    """
    day_cols = [
        "monday", "tuesday", "wednesday", "thursday",
        "friday", "saturday", "sunday",
    ]
    data: dict[str, list] = {  # type: ignore[type-arg]
        "service_id": [service_id],
        "start_date": [start_date],
        "end_date": [end_date],
    }
    for i, col in enumerate(day_cols):
        data[col] = [int(days[i])]
    return pl.DataFrame(data)


def _make_calendar_dates(
    entries: list[tuple[str, date, int]],
) -> pl.DataFrame:
    """Create a calendar_dates DataFrame from (service_id, date, exc_type) tuples."""
    return pl.DataFrame(
        {
            "service_id": [e[0] for e in entries],
            "date": [e[1] for e in entries],
            "exception_type": [e[2] for e in entries],
        },
    )


# ------------------------------------------------------------------
# Test C1: weekday-only calendar
# ------------------------------------------------------------------


def test_build_service_date_map_weekday_only() -> None:
    feed = {
        "calendar": _make_calendar("WEEK", date(2021, 1, 4), date(2021, 1, 10)),
    }
    result = build_service_date_map(feed)
    assert "WEEK" in result
    assert len(result["WEEK"]) == 5
    expected = {date(2021, 1, d) for d in range(4, 9)}  # Mon-Fri
    assert result["WEEK"] == expected


# ------------------------------------------------------------------
# Test C2: calendar with exception_type=1 (add)
# ------------------------------------------------------------------


def test_build_service_date_map_with_exception_add() -> None:
    feed = {
        "calendar": _make_calendar("WEEK", date(2021, 1, 4), date(2021, 1, 10)),
        "calendar_dates": _make_calendar_dates([
            ("WEEK", date(2021, 1, 9), 1),  # Saturday added
        ]),
    }
    result = build_service_date_map(feed)
    assert len(result["WEEK"]) == 6
    assert date(2021, 1, 9) in result["WEEK"]


# ------------------------------------------------------------------
# Test C3: calendar with exception_type=2 (remove)
# ------------------------------------------------------------------


def test_build_service_date_map_with_exception_remove() -> None:
    feed = {
        "calendar": _make_calendar("WEEK", date(2021, 1, 4), date(2021, 1, 10)),
        "calendar_dates": _make_calendar_dates([
            ("WEEK", date(2021, 1, 5), 2),  # Tuesday removed
        ]),
    }
    result = build_service_date_map(feed)
    assert len(result["WEEK"]) == 4
    assert date(2021, 1, 5) not in result["WEEK"]
    expected = {date(2021, 1, 4), date(2021, 1, 6), date(2021, 1, 7), date(2021, 1, 8)}
    assert result["WEEK"] == expected


# ------------------------------------------------------------------
# Test C4: calendar_dates only (no calendar table)
# ------------------------------------------------------------------


def test_build_service_date_map_calendar_dates_only() -> None:
    feed = {
        "calendar_dates": _make_calendar_dates([
            ("SPECIAL", date(2021, 3, 1), 1),
            ("SPECIAL", date(2021, 3, 15), 1),
        ]),
    }
    result = build_service_date_map(feed)
    assert "SPECIAL" in result
    assert result["SPECIAL"] == {date(2021, 3, 1), date(2021, 3, 15)}


# ------------------------------------------------------------------
# Test C5: intersection cache with shared dates
# ------------------------------------------------------------------


def test_intersection_cache_shared_dates() -> None:
    service_dates = {
        "A": {date(2021, 1, 4), date(2021, 1, 5), date(2021, 1, 6)},
        "B": {date(2021, 1, 5), date(2021, 1, 6), date(2021, 1, 7)},
    }
    cache = ServiceIdIntersectionCache(service_dates)
    result = cache.first_intersecting_date("A", "B")
    assert result == date(2021, 1, 5)
    # Second call should use cache
    result2 = cache.first_intersecting_date("B", "A")
    assert result2 == date(2021, 1, 5)


# ------------------------------------------------------------------
# Test C6: intersection cache with no shared dates
# ------------------------------------------------------------------


def test_intersection_cache_no_shared_dates() -> None:
    service_dates = {
        "A": {date(2021, 1, 4), date(2021, 1, 5)},
        "B": {date(2021, 1, 6), date(2021, 1, 7)},
    }
    cache = ServiceIdIntersectionCache(service_dates)
    result = cache.first_intersecting_date("A", "B")
    assert result is None


# ------------------------------------------------------------------
# Test C7: intersection cache with unknown service_id
# ------------------------------------------------------------------


def test_intersection_cache_unknown_service_id() -> None:
    service_dates = {
        "A": {date(2021, 1, 4)},
    }
    cache = ServiceIdIntersectionCache(service_dates)
    result = cache.first_intersecting_date("A", "UNKNOWN")
    assert result is None
