"""Tests for validate_service_gap."""

from __future__ import annotations

from datetime import date, timedelta

import polars as pl
import pytest

from gtfs_validator.context import ValidationContext
from gtfs_validator.notices import Severity
from gtfs_validator.validators.service_gap import validate_service_gap

CTX = ValidationContext(country_code="US", date_for_validation=date(2024, 1, 1))

ALL_DAYS = {
    "monday": 1,
    "tuesday": 1,
    "wednesday": 1,
    "thursday": 1,
    "friday": 1,
    "saturday": 1,
    "sunday": 1,
}


def make_calendar(rows: list[dict]) -> pl.DataFrame:
    """Build a calendar DataFrame from a list of row dicts."""
    if not rows:
        return pl.DataFrame(
            schema={
                "service_id": pl.Utf8,
                "start_date": pl.Date,
                "end_date": pl.Date,
                "monday": pl.Int64,
                "tuesday": pl.Int64,
                "wednesday": pl.Int64,
                "thursday": pl.Int64,
                "friday": pl.Int64,
                "saturday": pl.Int64,
                "sunday": pl.Int64,
            }
        )
    return pl.DataFrame(rows)


def make_calendar_dates(rows: list[dict]) -> pl.DataFrame:
    """Build a calendar_dates DataFrame from a list of row dicts."""
    if not rows:
        return pl.DataFrame(
            schema={
                "service_id": pl.Utf8,
                "date": pl.Date,
                "exception_type": pl.Int64,
            }
        )
    return pl.DataFrame(rows)


def make_feed(calendar_rows: list[dict], calendar_dates_rows: list[dict] | None = None) -> dict[str, pl.DataFrame]:
    feed: dict[str, pl.DataFrame] = {"calendar": make_calendar(calendar_rows)}
    if calendar_dates_rows is not None:
        feed["calendar_dates"] = make_calendar_dates(calendar_dates_rows)
    return feed


# Helper: build SERVICE_REMOVED entries for a range of dates
def removed_range(service_id: str, start: date, end: date) -> list[dict]:
    rows = []
    current = start
    while current <= end:
        rows.append({"service_id": service_id, "date": current, "exception_type": 2})
        current += timedelta(days=1)
    return rows


# ---------- Test 1 ----------
def test_continuous_service_no_notice():
    """Continuous coverage yields a single merged interval — no gaps."""
    feed = make_feed([
        {"service_id": "service_1", "start_date": date(2024, 1, 1), "end_date": date(2024, 1, 31), **ALL_DAYS}
    ])
    assert validate_service_gap(feed, CTX) == []


# ---------- Test 2 ----------
def test_gap_exactly_at_limit_no_notice():
    """Gap == MAX_GAP_DAYS (13 days) — threshold is strictly greater-than, no notice."""
    calendar_dates_rows = removed_range("service_1", date(2024, 1, 11), date(2024, 1, 23))  # 13 days
    feed = make_feed(
        [{"service_id": "service_1", "start_date": date(2024, 1, 1), "end_date": date(2024, 1, 31), **ALL_DAYS}],
        calendar_dates_rows,
    )
    assert validate_service_gap(feed, CTX) == []


# ---------- Test 3 ----------
def test_gap_just_above_limit_one_notice():
    """Gap == 14 days — exactly 1 notice emitted."""
    calendar_dates_rows = removed_range("service_1", date(2024, 1, 11), date(2024, 1, 24))  # 14 days
    feed = make_feed(
        [{"service_id": "service_1", "start_date": date(2024, 1, 1), "end_date": date(2024, 1, 31), **ALL_DAYS}],
        calendar_dates_rows,
    )
    notices = validate_service_gap(feed, CTX)
    assert len(notices) == 1
    n = notices[0]
    assert n.code == "big_gap_in_service"
    assert n.severity == Severity.INFO
    assert n.fields["service_id"] == "service_1"
    assert n.fields["gap_start_date"] == "2024-01-10"
    assert n.fields["gap_end_date"] == "2024-01-25"
    assert n.fields["gap_duration_days"] == 14


# ---------- Test 4 ----------
def test_large_gap_correct_notice_fields():
    """Same 14-day gap — all four notice fields verified."""
    calendar_dates_rows = removed_range("service_1", date(2024, 1, 11), date(2024, 1, 24))
    feed = make_feed(
        [{"service_id": "service_1", "start_date": date(2024, 1, 1), "end_date": date(2024, 1, 31), **ALL_DAYS}],
        calendar_dates_rows,
    )
    notices = validate_service_gap(feed, CTX)
    assert len(notices) == 1
    fields = notices[0].fields
    assert fields["service_id"] == "service_1"
    assert fields["gap_start_date"] == "2024-01-10"
    assert fields["gap_end_date"] == "2024-01-25"
    assert fields["gap_duration_days"] == 14


# ---------- Test 5 ----------
def test_multiple_services_only_gappy_service_reports_notice():
    """Only the service with a large gap emits a notice."""
    # service_1: Jan 1 – Feb 29; remove Jan 11–31 = 21-day gap
    # service_2: Jan 1 – Feb 29; no removals
    calendar_rows = [
        {"service_id": "service_1", "start_date": date(2024, 1, 1), "end_date": date(2024, 2, 29), **ALL_DAYS},
        {"service_id": "service_2", "start_date": date(2024, 1, 1), "end_date": date(2024, 2, 29), **ALL_DAYS},
    ]
    calendar_dates_rows = removed_range("service_1", date(2024, 1, 11), date(2024, 1, 31))  # 21 days
    feed = make_feed(calendar_rows, calendar_dates_rows)
    notices = validate_service_gap(feed, CTX)
    assert len(notices) == 1
    assert notices[0].fields["service_id"] == "service_1"
    service_ids = {n.fields["service_id"] for n in notices}
    assert "service_2" not in service_ids


# ---------- Test 6 ----------
def test_multiple_services_both_with_large_gaps_two_notices():
    """Both services have large gaps — two notices emitted."""
    calendar_rows = [
        {"service_id": "service_1", "start_date": date(2024, 1, 1), "end_date": date(2024, 2, 29), **ALL_DAYS},
        {"service_id": "service_2", "start_date": date(2024, 3, 1), "end_date": date(2024, 4, 30), **ALL_DAYS},
    ]
    calendar_dates_rows = (
        removed_range("service_1", date(2024, 1, 11), date(2024, 1, 31))  # 21-day gap
        + removed_range("service_2", date(2024, 3, 11), date(2024, 3, 31))  # 21-day gap
    )
    feed = make_feed(calendar_rows, calendar_dates_rows)
    notices = validate_service_gap(feed, CTX)
    assert len(notices) == 2
    service_ids = {n.fields["service_id"] for n in notices}
    assert "service_1" in service_ids
    assert "service_2" in service_ids


# ---------- Test 7 ----------
def test_small_gap_no_notice():
    """A 3-day gap (< 13) does not trigger a notice."""
    calendar_dates_rows = removed_range("service_1", date(2024, 1, 15), date(2024, 1, 17))  # 3 days
    feed = make_feed(
        [{"service_id": "service_1", "start_date": date(2024, 1, 1), "end_date": date(2024, 2, 28), **ALL_DAYS}],
        calendar_dates_rows,
    )
    assert validate_service_gap(feed, CTX) == []


# ---------- Test 8 ----------
def test_calendar_dates_only_service_not_checked():
    """calendar_dates-only services are not iterated when calendar is absent/empty."""
    feed = {
        "calendar": make_calendar([]),
        "calendar_dates": make_calendar_dates([
            {"service_id": "cd_service", "date": date(2024, 1, 1), "exception_type": 1},
            {"service_id": "cd_service", "date": date(2024, 1, 14), "exception_type": 1},
        ]),
    }
    assert validate_service_gap(feed, CTX) == []


# ---------- Test 9 ----------
def test_empty_calendar_no_notice():
    """Empty calendar DataFrame returns no notices."""
    feed = {"calendar": make_calendar([])}
    assert validate_service_gap(feed, CTX) == []


# ---------- Test 10 ----------
def test_single_day_service_no_notice():
    """Only one interval — no consecutive pair to compute a gap."""
    feed = make_feed([
        {"service_id": "service_1", "start_date": date(2024, 1, 1), "end_date": date(2024, 1, 1), **ALL_DAYS}
    ])
    assert validate_service_gap(feed, CTX) == []


# ---------- Test 11 ----------
def test_calendar_absent_no_notice():
    """No 'calendar' key in feed returns no notices."""
    feed: dict[str, pl.DataFrame] = {}
    assert validate_service_gap(feed, CTX) == []


# ---------- Test 12 ----------
def test_all_days_removed_no_notice():
    """All days removed — empty interval list, no gaps possible."""
    calendar_dates_rows = removed_range("service_1", date(2024, 1, 1), date(2024, 1, 5))
    feed = make_feed(
        [{"service_id": "service_1", "start_date": date(2024, 1, 1), "end_date": date(2024, 1, 5), **ALL_DAYS}],
        calendar_dates_rows,
    )
    assert validate_service_gap(feed, CTX) == []


# ---------- Test 13 ----------
def test_service_added_adjacent_merges_correctly():
    """SERVICE_ADDED adjacent to an existing interval merges correctly, no spurious gaps."""
    # Mon–Fri only calendar for Jan 1–10 creates breaks at weekends
    weekday_pattern = {
        "monday": 1, "tuesday": 1, "wednesday": 1, "thursday": 1, "friday": 1,
        "saturday": 0, "sunday": 0,
    }
    calendar_dates_rows = [
        # Add Jan 6 (Saturday) and Jan 7 (Sunday) to fill the first weekend
        {"service_id": "service_1", "date": date(2024, 1, 6), "exception_type": 1},
        {"service_id": "service_1", "date": date(2024, 1, 7), "exception_type": 1},
    ]
    feed = make_feed(
        [{"service_id": "service_1", "start_date": date(2024, 1, 1), "end_date": date(2024, 1, 10), **weekday_pattern}],
        calendar_dates_rows,
    )
    notices = validate_service_gap(feed, CTX)
    # No notice should be emitted since all remaining gaps are small (≤ 13 days)
    assert all(n.fields["gap_duration_days"] <= 13 for n in notices)


# ---------- Test 14 ----------
def test_weekly_pattern_zero_contributes_nothing():
    """All day flags = 0 means no active days — no intervals, no notices."""
    zero_days = {
        "monday": 0, "tuesday": 0, "wednesday": 0, "thursday": 0,
        "friday": 0, "saturday": 0, "sunday": 0,
    }
    feed = make_feed([
        {"service_id": "service_1", "start_date": date(2024, 1, 1), "end_date": date(2024, 1, 31), **zero_days}
    ])
    assert validate_service_gap(feed, CTX) == []


# ---------- Test 15 ----------
def test_service_removed_not_in_interval_is_noop():
    """SERVICE_REMOVED for a date outside the active range is a no-op."""
    calendar_dates_rows = [
        {"service_id": "service_1", "date": date(2024, 2, 15), "exception_type": 2}
    ]
    feed = make_feed(
        [{"service_id": "service_1", "start_date": date(2024, 1, 1), "end_date": date(2024, 1, 31), **ALL_DAYS}],
        calendar_dates_rows,
    )
    # Jan 1–31 is continuous, so no gap
    assert validate_service_gap(feed, CTX) == []


# ---------- Test 16 ----------
def test_multiple_gaps_single_service_multiple_notices():
    """Two large gaps for the same service produce two notices."""
    # Jan 1 – Apr 30, all days; remove Jan 15 – Feb 4 (21 days) and Mar 10 – Mar 30 (21 days)
    calendar_dates_rows = (
        removed_range("service_1", date(2024, 1, 15), date(2024, 2, 4))   # 21 days
        + removed_range("service_1", date(2024, 3, 10), date(2024, 3, 30))  # 21 days
    )
    feed = make_feed(
        [{"service_id": "service_1", "start_date": date(2024, 1, 1), "end_date": date(2024, 4, 30), **ALL_DAYS}],
        calendar_dates_rows,
    )
    notices = validate_service_gap(feed, CTX)
    assert len(notices) == 2
    assert all(n.fields["service_id"] == "service_1" for n in notices)
    assert all(n.fields["gap_duration_days"] == 21 for n in notices)


# ---------- Test 17 ----------
def test_notice_severity_is_info():
    """Gap notices have Severity.INFO."""
    calendar_dates_rows = removed_range("service_1", date(2024, 1, 11), date(2024, 1, 24))  # 14-day gap
    feed = make_feed(
        [{"service_id": "service_1", "start_date": date(2024, 1, 1), "end_date": date(2024, 1, 31), **ALL_DAYS}],
        calendar_dates_rows,
    )
    notices = validate_service_gap(feed, CTX)
    assert len(notices) == 1
    assert notices[0].severity == Severity.INFO


# ---------- Test 18 ----------
def test_calendar_dates_absent_uses_calendar_only():
    """Two calendar rows for the same service_id produce a merged interval set and detect a 21-day gap."""
    # Jan 1–10 and Feb 1–10: gap between Jan 10 and Feb 1 = 21 days
    calendar_rows = [
        {"service_id": "service_1", "start_date": date(2024, 1, 1), "end_date": date(2024, 1, 10), **ALL_DAYS},
        {"service_id": "service_1", "start_date": date(2024, 2, 1), "end_date": date(2024, 2, 10), **ALL_DAYS},
    ]
    feed = make_feed(calendar_rows)  # no calendar_dates
    notices = validate_service_gap(feed, CTX)
    assert len(notices) == 1
    assert notices[0].fields["gap_duration_days"] == 21
