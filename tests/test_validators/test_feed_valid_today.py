"""Tests for validate_feed_valid_today."""

from datetime import date

import polars as pl
import pytest

from gtfs_validator.context import ValidationContext
from gtfs_validator.notices import Severity
from gtfs_validator.validators.feed_valid_today import validate_feed_valid_today

TEST_NOW = date(2026, 2, 25)
CTX = ValidationContext(country_code="US", date_for_validation=TEST_NOW)


def make_feed_info(
    feed_start_date: date | None,
    csv_row_number: int = 1,
) -> dict[str, pl.DataFrame]:
    return {
        "feed_info": pl.DataFrame(
            {
                "feed_publisher_name": ["Test Publisher"],
                "feed_publisher_url": ["https://example.com"],
                "feed_lang": ["en"],
                "feed_start_date": [feed_start_date],
                "feed_end_date": [None],
                "csv_row_number": [csv_row_number],
            },
            schema={
                "feed_publisher_name": pl.Utf8,
                "feed_publisher_url": pl.Utf8,
                "feed_lang": pl.Utf8,
                "feed_start_date": pl.Date,
                "feed_end_date": pl.Date,
                "csv_row_number": pl.Int64,
            },
        )
    }


def test_feed_start_date_today_no_notice() -> None:
    feed = make_feed_info(feed_start_date=date(2026, 2, 25))
    notices = validate_feed_valid_today(feed, CTX)
    assert notices == []


def test_feed_start_date_in_past_no_notice() -> None:
    feed = make_feed_info(feed_start_date=date(2026, 1, 26))
    notices = validate_feed_valid_today(feed, CTX)
    assert notices == []


def test_feed_start_date_in_future_emits_notice() -> None:
    feed = make_feed_info(feed_start_date=date(2026, 3, 4))
    notices = validate_feed_valid_today(feed, CTX)
    assert len(notices) == 1
    assert notices[0].code == "future_feed"
    assert notices[0].severity == Severity.INFO
    assert notices[0].fields["feed_start_date"] == "20260304"
    assert notices[0].fields["current_date"] == "20260225"


def test_feed_info_absent_no_notice() -> None:
    feed: dict[str, pl.DataFrame] = {}
    notices = validate_feed_valid_today(feed, CTX)
    assert notices == []


def test_feed_info_empty_no_notice() -> None:
    feed = {
        "feed_info": pl.DataFrame(
            {"feed_start_date": [], "feed_end_date": [], "csv_row_number": []},
            schema={
                "feed_start_date": pl.Date,
                "feed_end_date": pl.Date,
                "csv_row_number": pl.Int64,
            },
        )
    }
    notices = validate_feed_valid_today(feed, CTX)
    assert notices == []


def test_feed_start_date_null_no_notice() -> None:
    feed = make_feed_info(feed_start_date=None)
    notices = validate_feed_valid_today(feed, CTX)
    assert notices == []


def test_multiple_rows_one_past_one_future_no_notice() -> None:
    feed = {
        "feed_info": pl.DataFrame(
            {
                "feed_publisher_name": ["A", "B"],
                "feed_publisher_url": ["https://a.com", "https://b.com"],
                "feed_lang": ["en", "en"],
                "feed_start_date": [date(2026, 1, 26), date(2026, 3, 4)],
                "feed_end_date": [None, None],
                "csv_row_number": [1, 2],
            },
            schema={
                "feed_publisher_name": pl.Utf8,
                "feed_publisher_url": pl.Utf8,
                "feed_lang": pl.Utf8,
                "feed_start_date": pl.Date,
                "feed_end_date": pl.Date,
                "csv_row_number": pl.Int64,
            },
        )
    }
    notices = validate_feed_valid_today(feed, CTX)
    assert notices == []


def test_multiple_rows_all_future_emits_notice() -> None:
    feed = {
        "feed_info": pl.DataFrame(
            {
                "feed_publisher_name": ["A", "B"],
                "feed_publisher_url": ["https://a.com", "https://b.com"],
                "feed_lang": ["en", "en"],
                "feed_start_date": [date(2026, 3, 4), date(2026, 3, 11)],
                "feed_end_date": [None, None],
                "csv_row_number": [1, 2],
            },
            schema={
                "feed_publisher_name": pl.Utf8,
                "feed_publisher_url": pl.Utf8,
                "feed_lang": pl.Utf8,
                "feed_start_date": pl.Date,
                "feed_end_date": pl.Date,
                "csv_row_number": pl.Int64,
            },
        )
    }
    notices = validate_feed_valid_today(feed, CTX)
    assert len(notices) == 1
    assert notices[0].fields["feed_start_date"] == "20260304"


def test_feed_start_date_one_day_future_emits_notice() -> None:
    feed = make_feed_info(feed_start_date=date(2026, 2, 26))
    notices = validate_feed_valid_today(feed, CTX)
    assert len(notices) == 1
    assert notices[0].code == "future_feed"


def test_notice_fields_are_complete() -> None:
    feed = make_feed_info(feed_start_date=date(2026, 3, 4))
    notices = validate_feed_valid_today(feed, CTX)
    assert len(notices) == 1
    assert set(notices[0].fields.keys()) == {"feed_start_date", "current_date"}


def test_feed_start_date_column_missing_no_notice() -> None:
    feed = {
        "feed_info": pl.DataFrame(
            {
                "feed_publisher_name": ["Test"],
                "feed_publisher_url": ["https://example.com"],
                "feed_lang": ["en"],
                "csv_row_number": [1],
            }
        )
    }
    notices = validate_feed_valid_today(feed, CTX)
    assert notices == []
