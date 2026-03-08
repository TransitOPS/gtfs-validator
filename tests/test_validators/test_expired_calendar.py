"""Tests for validate_expired_calendar."""

from __future__ import annotations

from datetime import date, timedelta

import polars as pl

from gtfs_validator.context import ValidationContext
from gtfs_validator.notices import Severity
from gtfs_validator.validators.expired_calendar import validate_expired_calendar

NOW = date(2024, 4, 15)  # a Tuesday
CTX = ValidationContext(country_code="US", date_for_validation=NOW)


def make_calendar(rows: list[dict]) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame(
            schema={
                "service_id": pl.Utf8,
                "monday": pl.Int64,
                "tuesday": pl.Int64,
                "wednesday": pl.Int64,
                "thursday": pl.Int64,
                "friday": pl.Int64,
                "saturday": pl.Int64,
                "sunday": pl.Int64,
                "start_date": pl.Date,
                "end_date": pl.Date,
            }
        )
    return pl.DataFrame(rows)


def make_calendar_dates(rows: list[dict]) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame(
            schema={
                "service_id": pl.Utf8,
                "date": pl.Date,
                "exception_type": pl.Int64,
            }
        )
    return pl.DataFrame(rows)


# ---------- Test 1 ----------
def test_calendar_end_date_one_day_ago():
    feed = {
        "calendar": make_calendar([{
            "service_id": "WEEK", "monday": 1, "tuesday": 1, "wednesday": 1,
            "thursday": 1, "friday": 1, "saturday": 0, "sunday": 0,
            "start_date": NOW - timedelta(days=7), "end_date": NOW - timedelta(days=1),
        }]),
        "calendar_dates": make_calendar_dates([]),
    }
    notices = validate_expired_calendar(feed, CTX)
    assert len(notices) == 1
    assert notices[0].code == "expired_calendar"
    assert notices[0].fields["csv_row_number"] == 2
    assert notices[0].fields["service_id"] == "WEEK"


# ---------- Test 2 ----------
def test_calendar_end_date_today_no_notice():
    feed = {
        "calendar": make_calendar([{
            "service_id": "WEEK", "monday": 1, "tuesday": 1, "wednesday": 1,
            "thursday": 1, "friday": 1, "saturday": 0, "sunday": 0,
            "start_date": NOW - timedelta(days=7), "end_date": NOW,
        }]),
        "calendar_dates": make_calendar_dates([]),
    }
    notices = validate_expired_calendar(feed, CTX)
    assert len(notices) == 0


# ---------- Test 3 ----------
def test_calendar_end_date_tomorrow_no_notice():
    feed = {
        "calendar": make_calendar([{
            "service_id": "WEEK", "monday": 1, "tuesday": 1, "wednesday": 1,
            "thursday": 1, "friday": 1, "saturday": 0, "sunday": 0,
            "start_date": NOW - timedelta(days=7), "end_date": NOW + timedelta(days=1),
        }]),
        "calendar_dates": make_calendar_dates([]),
    }
    notices = validate_expired_calendar(feed, CTX)
    assert len(notices) == 0


# ---------- Test 4 ----------
def test_expired_calendar_extended_by_added_date():
    feed = {
        "calendar": make_calendar([{
            "service_id": "WEEK", "monday": 1, "tuesday": 1, "wednesday": 1,
            "thursday": 1, "friday": 1, "saturday": 0, "sunday": 0,
            "start_date": NOW - timedelta(days=7), "end_date": NOW - timedelta(days=1),
        }]),
        "calendar_dates": make_calendar_dates([{
            "service_id": "WEEK", "date": NOW, "exception_type": 1,
        }]),
    }
    notices = validate_expired_calendar(feed, CTX)
    assert len(notices) == 0


# ---------- Test 5 ----------
def test_active_calendar_shortened_by_removed_date():
    feed = {
        "calendar": make_calendar([{
            "service_id": "WEEK", "monday": 1, "tuesday": 1, "wednesday": 1,
            "thursday": 1, "friday": 1, "saturday": 0, "sunday": 0,
            "start_date": NOW - timedelta(days=7), "end_date": NOW,
        }]),
        "calendar_dates": make_calendar_dates([{
            "service_id": "WEEK", "date": NOW, "exception_type": 2,
        }]),
    }
    notices = validate_expired_calendar(feed, CTX)
    assert len(notices) == 1
    assert notices[0].fields["csv_row_number"] == 2
    assert notices[0].fields["service_id"] == "WEEK"


# ---------- Test 6 ----------
def test_calendar_no_active_days_all_zeros():
    feed = {
        "calendar": make_calendar([{
            "service_id": "WEEK", "monday": 0, "tuesday": 0, "wednesday": 0,
            "thursday": 0, "friday": 0, "saturday": 0, "sunday": 0,
            "start_date": NOW - timedelta(days=7), "end_date": NOW,
        }]),
        "calendar_dates": make_calendar_dates([]),
    }
    notices = validate_expired_calendar(feed, CTX)
    assert len(notices) == 0


# ---------- Test 7 ----------
def test_fk_violation_calendar_nonempty():
    feed = {
        "calendar": make_calendar([{
            "service_id": "SERVICE_ID", "monday": 0, "tuesday": 0, "wednesday": 0,
            "thursday": 0, "friday": 0, "saturday": 0, "sunday": 0,
            "start_date": NOW - timedelta(days=7), "end_date": NOW,
        }]),
        "calendar_dates": make_calendar_dates([
            {"service_id": "NOT_SERVICE_ID", "date": NOW - timedelta(days=3), "exception_type": 1},
            {"service_id": "NOT_SERVICE_ID", "date": NOW - timedelta(days=2), "exception_type": 1},
            {"service_id": "NOT_SERVICE_ID", "date": NOW - timedelta(days=1), "exception_type": 1},
        ]),
    }
    notices = validate_expired_calendar(feed, CTX)
    assert len(notices) == 0


# ---------- Test 8 ----------
def test_calendar_dates_only_one_not_expired():
    feed = {
        "calendar": make_calendar([]),
        "calendar_dates": make_calendar_dates([
            {"service_id": "s1", "date": NOW - timedelta(days=2), "exception_type": 1},
            {"service_id": "s2", "date": NOW - timedelta(days=3), "exception_type": 1},
            {"service_id": "s3", "date": NOW + timedelta(days=1), "exception_type": 1},
        ]),
    }
    notices = validate_expired_calendar(feed, CTX)
    assert len(notices) == 0


# ---------- Test 9 ----------
def test_calendar_dates_only_all_expired():
    feed = {
        "calendar": make_calendar([]),
        "calendar_dates": make_calendar_dates([
            {"service_id": "s1", "date": NOW - timedelta(days=4), "exception_type": 1},
            {"service_id": "s2", "date": NOW - timedelta(days=3), "exception_type": 1},
            {"service_id": "s3", "date": NOW - timedelta(days=2), "exception_type": 1},
            {"service_id": "s4", "date": NOW - timedelta(days=1), "exception_type": 1},
        ]),
    }
    notices = validate_expired_calendar(feed, CTX)
    assert len(notices) == 4
    row_numbers = [n.fields["csv_row_number"] for n in notices]
    assert row_numbers == sorted(row_numbers)


# ---------- Test 10 ----------
def test_both_tables_empty():
    feed = {
        "calendar": make_calendar([]),
        "calendar_dates": make_calendar_dates([]),
    }
    notices = validate_expired_calendar(feed, CTX)
    assert len(notices) == 0


# ---------- Test 11 ----------
def test_single_service_not_expired():
    feed = {
        "calendar": make_calendar([{
            "service_id": "A", "monday": 1, "tuesday": 1, "wednesday": 1,
            "thursday": 1, "friday": 1, "saturday": 0, "sunday": 0,
            "start_date": NOW - timedelta(days=7), "end_date": NOW + timedelta(days=30),
        }]),
        "calendar_dates": make_calendar_dates([]),
    }
    notices = validate_expired_calendar(feed, CTX)
    assert len(notices) == 0


# ---------- Test 12 ----------
def test_multiple_services_some_expired():
    feed = {
        "calendar": make_calendar([
            {
                "service_id": "A", "monday": 1, "tuesday": 1, "wednesday": 1,
                "thursday": 1, "friday": 1, "saturday": 0, "sunday": 0,
                "start_date": NOW - timedelta(days=14), "end_date": NOW - timedelta(days=1),
            },
            {
                "service_id": "B", "monday": 1, "tuesday": 1, "wednesday": 1,
                "thursday": 1, "friday": 1, "saturday": 0, "sunday": 0,
                "start_date": NOW - timedelta(days=7), "end_date": NOW + timedelta(days=30),
            },
        ]),
        "calendar_dates": make_calendar_dates([]),
    }
    notices = validate_expired_calendar(feed, CTX)
    assert len(notices) == 1
    assert notices[0].fields["csv_row_number"] == 2
    assert notices[0].fields["service_id"] == "A"


# ---------- Test 13 ----------
def test_notice_severity_is_warning():
    feed = {
        "calendar": make_calendar([{
            "service_id": "WEEK", "monday": 1, "tuesday": 1, "wednesday": 1,
            "thursday": 1, "friday": 1, "saturday": 0, "sunday": 0,
            "start_date": NOW - timedelta(days=7), "end_date": NOW - timedelta(days=1),
        }]),
        "calendar_dates": make_calendar_dates([]),
    }
    notices = validate_expired_calendar(feed, CTX)
    assert len(notices) == 1
    assert notices[0].severity == Severity.WARNING


# ---------- Test 14 ----------
def test_notice_code():
    feed = {
        "calendar": make_calendar([{
            "service_id": "WEEK", "monday": 1, "tuesday": 1, "wednesday": 1,
            "thursday": 1, "friday": 1, "saturday": 0, "sunday": 0,
            "start_date": NOW - timedelta(days=7), "end_date": NOW - timedelta(days=1),
        }]),
        "calendar_dates": make_calendar_dates([]),
    }
    notices = validate_expired_calendar(feed, CTX)
    assert len(notices) == 1
    assert notices[0].code == "expired_calendar"


# ---------- Test 15 ----------
def test_notices_sorted_by_csv_row_number():
    feed = {
        "calendar": make_calendar([
            {
                "service_id": "C", "monday": 1, "tuesday": 1, "wednesday": 1,
                "thursday": 1, "friday": 1, "saturday": 0, "sunday": 0,
                "start_date": NOW - timedelta(days=14), "end_date": NOW - timedelta(days=1),
            },
            {
                "service_id": "A", "monday": 1, "tuesday": 1, "wednesday": 1,
                "thursday": 1, "friday": 1, "saturday": 0, "sunday": 0,
                "start_date": NOW - timedelta(days=14), "end_date": NOW - timedelta(days=1),
            },
            {
                "service_id": "B", "monday": 1, "tuesday": 1, "wednesday": 1,
                "thursday": 1, "friday": 1, "saturday": 0, "sunday": 0,
                "start_date": NOW - timedelta(days=14), "end_date": NOW - timedelta(days=1),
            },
        ]),
        "calendar_dates": make_calendar_dates([]),
    }
    notices = validate_expired_calendar(feed, CTX)
    assert len(notices) == 3
    row_numbers = [n.fields["csv_row_number"] for n in notices]
    assert row_numbers == [2, 3, 4]


# ---------- Test 16 ----------
def test_missing_tables_no_notice():
    feed: dict[str, pl.DataFrame] = {}
    notices = validate_expired_calendar(feed, CTX)
    assert len(notices) == 0


# ---------- Test 17 ----------
def test_calendar_dates_only_single_expired_service():
    feed = {
        "calendar": make_calendar([]),
        "calendar_dates": make_calendar_dates([{
            "service_id": "solo", "date": NOW - timedelta(days=1), "exception_type": 1,
        }]),
    }
    notices = validate_expired_calendar(feed, CTX)
    assert len(notices) == 1
    assert notices[0].fields["service_id"] == "solo"
    assert notices[0].fields["csv_row_number"] == 2


# ---------- Test 18 ----------
def test_removed_dates_dont_create_service():
    feed = {
        "calendar": make_calendar([]),
        "calendar_dates": make_calendar_dates([{
            "service_id": "ghost", "date": NOW - timedelta(days=1), "exception_type": 2,
        }]),
    }
    notices = validate_expired_calendar(feed, CTX)
    assert len(notices) == 0
