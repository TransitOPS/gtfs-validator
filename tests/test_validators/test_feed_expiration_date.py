"""Tests for validate_feed_expiration_date."""

from __future__ import annotations

from datetime import date, timedelta

import polars as pl
import pytest

from gtfs_validator.context import ValidationContext
from gtfs_validator.notices import Severity
from gtfs_validator.validators.feed_expiration_date import validate_feed_expiration_date

TEST_NOW = date(2021, 1, 25)
CTX = ValidationContext(country_code="US", date_for_validation=TEST_NOW)


def make_feed_info(feed_end_date: date | None) -> dict[str, pl.DataFrame]:
    """Return a minimal feed dict with a single feed_info row."""
    row = {
        "feed_publisher_name": ["Test Publisher"],
        "feed_publisher_url": ["https://example.com"],
        "feed_lang": ["en"],
        "feed_end_date": [feed_end_date],
        "csv_row_number": [1],
    }
    return {"feed_info": pl.DataFrame(row)}


# ---------------------------------------------------------------------------
# Tests derived from Java contracts
# ---------------------------------------------------------------------------


def test_expiring_in_6_days_emits_7_day_notice() -> None:
    feed = make_feed_info(TEST_NOW + timedelta(days=6))  # 2021-01-31
    notices = validate_feed_expiration_date(feed, CTX)

    assert len(notices) == 1
    n = notices[0]
    assert n.code == "feed_expiration_date_7_days"
    assert n.severity == Severity.WARNING
    assert n.fields["csv_row_number"] == 1
    assert n.fields["current_date"] == "20210125"
    assert n.fields["feed_end_date"] == "20210131"
    assert n.fields["suggested_expiration_date"] == "20210201"  # Feb 1, month boundary


def test_expiring_in_7_days_emits_30_day_notice() -> None:
    # The 7-day boundary itself (current_date + 7) falls into the 30-day window.
    feed = make_feed_info(TEST_NOW + timedelta(days=7))  # 2021-02-01
    notices = validate_feed_expiration_date(feed, CTX)

    assert len(notices) == 1
    n = notices[0]
    assert n.code == "feed_expiration_date_30_days"
    assert n.severity == Severity.WARNING
    assert n.fields["csv_row_number"] == 1
    assert n.fields["current_date"] == "20210125"
    assert n.fields["feed_end_date"] == "20210201"
    assert n.fields["suggested_expiration_date"] == "20210224"  # Feb 24, month boundary


def test_expiring_in_29_days_emits_30_day_notice() -> None:
    feed = make_feed_info(TEST_NOW + timedelta(days=29))  # 2021-02-23
    notices = validate_feed_expiration_date(feed, CTX)

    assert len(notices) == 1
    n = notices[0]
    assert n.code == "feed_expiration_date_30_days"
    assert n.fields["suggested_expiration_date"] == "20210224"


def test_expiring_in_30_days_emits_no_notice() -> None:
    # The 30-day boundary itself is excluded by strict <.
    feed = make_feed_info(TEST_NOW + timedelta(days=30))  # 2021-02-24
    notices = validate_feed_expiration_date(feed, CTX)

    assert notices == []


def test_expiring_in_past_emits_7_day_notice() -> None:
    feed = make_feed_info(TEST_NOW - timedelta(days=1))  # 2021-01-24
    notices = validate_feed_expiration_date(feed, CTX)

    assert len(notices) == 1
    n = notices[0]
    assert n.code == "feed_expiration_date_7_days"
    assert n.fields["feed_end_date"] == "20210124"
    assert n.fields["suggested_expiration_date"] == "20210201"


# ---------------------------------------------------------------------------
# Python edge-case tests
# ---------------------------------------------------------------------------


def test_feed_end_date_null_emits_no_notice() -> None:
    feed = make_feed_info(None)
    notices = validate_feed_expiration_date(feed, CTX)
    assert notices == []


def test_feed_info_absent_emits_no_notice() -> None:
    notices = validate_feed_expiration_date({}, CTX)
    assert notices == []


def test_feed_info_empty_emits_no_notice() -> None:
    feed: dict[str, pl.DataFrame] = {
        "feed_info": pl.DataFrame(
            {"feed_end_date": pl.Series([], dtype=pl.Date), "csv_row_number": pl.Series([], dtype=pl.Int64)}
        )
    }
    notices = validate_feed_expiration_date(feed, CTX)
    assert notices == []


def test_expiring_exactly_today_emits_7_day_notice() -> None:
    # today < today + 7, so triggers the 7-day notice
    feed = make_feed_info(TEST_NOW)
    notices = validate_feed_expiration_date(feed, CTX)

    assert len(notices) == 1
    assert notices[0].code == "feed_expiration_date_7_days"


def test_far_future_emits_no_notice() -> None:
    feed = make_feed_info(TEST_NOW + timedelta(days=365))
    notices = validate_feed_expiration_date(feed, CTX)
    assert notices == []


def test_past_by_many_days_emits_7_day_notice() -> None:
    feed = make_feed_info(TEST_NOW - timedelta(days=365))
    notices = validate_feed_expiration_date(feed, CTX)

    assert len(notices) == 1
    assert notices[0].code == "feed_expiration_date_7_days"


def test_only_one_notice_per_row() -> None:
    # A row in the 7-day window must not also emit a 30-day notice.
    feed = make_feed_info(TEST_NOW + timedelta(days=6))
    notices = validate_feed_expiration_date(feed, CTX)

    assert len(notices) == 1
    codes = [n.code for n in notices]
    assert "feed_expiration_date_30_days" not in codes


def test_notice_fields_are_complete() -> None:
    feed = make_feed_info(TEST_NOW + timedelta(days=6))
    notices = validate_feed_expiration_date(feed, CTX)

    assert len(notices) == 1
    expected_keys = {"csv_row_number", "current_date", "feed_end_date", "suggested_expiration_date"}
    assert set(notices[0].fields.keys()) == expected_keys


def test_month_boundary_arithmetic_7_days() -> None:
    # feed_end_date exactly equals TEST_NOW + 7 (Feb 1) — falls in 30-day window.
    feed = make_feed_info(date(2021, 2, 1))
    notices = validate_feed_expiration_date(feed, CTX)

    assert len(notices) == 1
    n = notices[0]
    assert n.code == "feed_expiration_date_30_days"
    assert n.fields["suggested_expiration_date"] == "20210224"
