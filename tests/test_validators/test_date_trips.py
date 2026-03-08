"""Tests for DateTripsValidator (validate_date_trips)."""

from __future__ import annotations

from datetime import date, timedelta

import polars as pl

from gtfs_validator.context import ValidationContext
from gtfs_validator.notices import Severity
from gtfs_validator.validators.date_trips import validate_date_trips

VALIDATION_DATE = date(2022, 12, 1)
CTX = ValidationContext(country_code="US", date_for_validation=VALIDATION_DATE)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _make_calendar(
    service_id: str,
    start_date: date,
    end_date: date,
    days: str = "1111111",
) -> pl.DataFrame:
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


def _make_calendar_dates(rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(
        rows,
        schema={
            "service_id": pl.Utf8,
            "date": pl.Date,
            "exception_type": pl.Int64,
        },
    )


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


def _dur(hours: int, minutes: int = 0, seconds: int = 0) -> timedelta:
    return timedelta(hours=hours, minutes=minutes, seconds=seconds)


def _n_trips(service_id: str, n: int) -> list[dict]:
    return [{"trip_id": f"t{i}", "service_id": service_id} for i in range(n)]


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------


def test_service_window_ending_before_7_days():
    """Test 1: Service ends on Dec 7 (before validation_date + 7 = Dec 8)."""
    feed = {
        "calendar": _make_calendar("s1", date(2022, 12, 1), date(2022, 12, 7)),
        "trips": _make_trips(_n_trips("s1", 6)),
    }
    notices = validate_date_trips(feed, CTX)
    assert len(notices) == 1
    n = notices[0]
    assert n.code == "trip_coverage_not_active_for_next_7_days"
    assert n.severity == Severity.WARNING
    assert n.fields["current_date"] == "20221201"
    assert n.fields["service_window_start_date"] == "20221201"
    assert n.fields["service_window_end_date"] == "20221207"


def test_service_window_starting_after_now():
    """Test 2: Service starts Dec 2 (after validation_date Dec 1)."""
    feed = {
        "calendar": _make_calendar("s1", date(2022, 12, 2), date(2022, 12, 8)),
        "trips": _make_trips(_n_trips("s1", 6)),
    }
    notices = validate_date_trips(feed, CTX)
    assert len(notices) == 1
    n = notices[0]
    assert n.fields["current_date"] == "20221201"
    assert n.fields["service_window_start_date"] == "20221202"
    assert n.fields["service_window_end_date"] == "20221208"


def test_service_window_exactly_7_days_no_notice():
    """Test 3: Service Dec 1-8 exactly covers validation_date + 7."""
    feed = {
        "calendar": _make_calendar("s1", date(2022, 12, 1), date(2022, 12, 8)),
        "trips": _make_trips(_n_trips("s1", 6)),
    }
    notices = validate_date_trips(feed, CTX)
    assert notices == []


def test_service_window_wider_than_7_days_no_notice():
    """Test 4: Service Nov 30 - Dec 9 comfortably covers 7 days."""
    feed = {
        "calendar": _make_calendar("s1", date(2022, 11, 30), date(2022, 12, 9)),
        "trips": _make_trips(_n_trips("s1", 6)),
    }
    notices = validate_date_trips(feed, CTX)
    assert notices == []


def test_no_trips_no_notice():
    """Test 5: Calendar exists but no trips -> no notices."""
    feed = {
        "calendar": _make_calendar("s1", date(2022, 12, 1), date(2022, 12, 8)),
        "trips": _make_trips([]),
    }
    notices = validate_date_trips(feed, CTX)
    assert notices == []


def test_no_calendar_data_no_notice():
    """Test 6: No calendar data, trips have no matching service dates."""
    feed = {
        "trips": _make_trips([{"trip_id": "t1", "service_id": "s1"}]),
    }
    notices = validate_date_trips(feed, CTX)
    assert notices == []


def test_frequency_based_trips():
    """Test 7: Frequency-based trip (06:00-10:00, headway=1800) -> 8 effective trips."""
    feed = {
        "calendar": _make_calendar("s1", date(2022, 12, 1), date(2022, 12, 8)),
        "trips": _make_trips([{"trip_id": "t1", "service_id": "s1"}]),
        "frequencies": _make_frequencies([
            {
                "trip_id": "t1",
                "start_time": _dur(6),
                "end_time": _dur(10),
                "headway_secs": 1800,
            },
        ]),
    }
    notices = validate_date_trips(feed, CTX)
    assert notices == []


def test_frequency_based_headway_zero():
    """Test 8: Frequency with headway_secs=0 counts as 1 trip."""
    feed = {
        "calendar": _make_calendar("s1", date(2022, 12, 1), date(2022, 12, 8)),
        "trips": _make_trips([{"trip_id": "t1", "service_id": "s1"}]),
        "frequencies": _make_frequencies([
            {
                "trip_id": "t1",
                "start_time": _dur(6),
                "end_time": _dur(10),
                "headway_secs": 0,
            },
        ]),
    }
    notices = validate_date_trips(feed, CTX)
    # 1 trip with headway=0 -> effective count 1, still covers 7 days
    assert notices == []


def test_single_day_service():
    """Test 9: Service only on Jan 15 2023, validation Dec 1 2022."""
    feed = {
        "calendar": _make_calendar("s1", date(2023, 1, 15), date(2023, 1, 15)),
        "trips": _make_trips(_n_trips("s1", 5)),
    }
    notices = validate_date_trips(feed, CTX)
    assert len(notices) == 1
    assert notices[0].code == "trip_coverage_not_active_for_next_7_days"


def test_orphan_service_id_trips_ignored():
    """Test 10: Trips with orphan service_id (not in calendar) -> no notices."""
    feed = {
        "calendar": _make_calendar("s_other", date(2022, 12, 1), date(2022, 12, 8)),
        "trips": _make_trips([{"trip_id": "t1", "service_id": "orphan"}]),
    }
    notices = validate_date_trips(feed, CTX)
    assert notices == []


def test_calendar_dates_only_service():
    """Test 11: Service defined only in calendar_dates (no calendar.txt)."""
    dates_rows = [
        {"service_id": "s1", "date": VALIDATION_DATE + timedelta(days=i), "exception_type": 1}
        for i in range(9)  # Dec 1 through Dec 9
    ]
    feed = {
        "calendar_dates": _make_calendar_dates(dates_rows),
        "trips": _make_trips(_n_trips("s1", 6)),
    }
    notices = validate_date_trips(feed, CTX)
    assert notices == []


def test_majority_threshold_with_outlier_dates():
    """Test 12: Main service covers validation+7, outlier date in 2024 doesn't shift window."""
    feed = {
        "calendar": pl.concat([
            _make_calendar("s1", date(2022, 11, 1), date(2023, 1, 31)),
            _make_calendar("s2", date(2024, 6, 15), date(2024, 6, 15)),
        ]),
        "trips": _make_trips(
            _n_trips("s1", 100) + [{"trip_id": "outlier", "service_id": "s2"}]
        ),
    }
    notices = validate_date_trips(feed, CTX)
    assert notices == []


def test_missing_trips_table():
    """Test 13: Feed has calendar but no trips key."""
    feed = {
        "calendar": _make_calendar("s1", date(2022, 12, 1), date(2022, 12, 8)),
    }
    notices = validate_date_trips(feed, CTX)
    assert notices == []


def test_empty_feed():
    """Test 14: Empty feed dict."""
    notices = validate_date_trips({}, CTX)
    assert notices == []
