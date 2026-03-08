"""Tests for validate_feed_service_date."""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from gtfs_validator.context import ValidationContext
from gtfs_validator.notices import Severity
from gtfs_validator.validators.feed_service_date import validate_feed_service_date

CTX = ValidationContext(country_code="US", date_for_validation=date(2024, 1, 1))

DATE_A = date(1971, 3, 27)
DATE_B = date(1971, 7, 9)


def make_feed_info(
    feed_start_date: date | None,
    feed_end_date: date | None,
    csv_row_number: int = 1,
) -> dict[str, pl.DataFrame]:
    return {
        "feed_info": pl.DataFrame(
            {
                "feed_publisher_name": ["Test Publisher"],
                "feed_publisher_url": ["https://example.com"],
                "feed_lang": ["en"],
                "feed_start_date": [feed_start_date],
                "feed_end_date": [feed_end_date],
                "csv_row_number": [csv_row_number],
            }
        )
    }


def test_no_start_date_emits_notice() -> None:
    """feed_end_date present but feed_start_date absent → notice for feed_start_date."""
    feed = make_feed_info(feed_start_date=None, feed_end_date=DATE_A)
    notices = validate_feed_service_date(feed, CTX)
    assert len(notices) == 1
    n = notices[0]
    assert n.code == "missing_feed_info_date"
    assert n.severity == Severity.WARNING
    assert n.fields["csv_row_number"] == 1
    assert n.fields["field_name"] == "feed_start_date"


def test_no_end_date_emits_notice() -> None:
    """feed_start_date present but feed_end_date absent → notice for feed_end_date."""
    feed = make_feed_info(feed_start_date=DATE_A, feed_end_date=None)
    notices = validate_feed_service_date(feed, CTX)
    assert len(notices) == 1
    n = notices[0]
    assert n.code == "missing_feed_info_date"
    assert n.severity == Severity.WARNING
    assert n.fields["csv_row_number"] == 1
    assert n.fields["field_name"] == "feed_end_date"


def test_both_dates_blank_emits_no_notice() -> None:
    """Both dates absent → no notice."""
    feed = make_feed_info(feed_start_date=None, feed_end_date=None)
    notices = validate_feed_service_date(feed, CTX)
    assert notices == []


def test_both_dates_present_emits_no_notice() -> None:
    """Both dates present → no notice."""
    feed = make_feed_info(feed_start_date=DATE_A, feed_end_date=DATE_B)
    notices = validate_feed_service_date(feed, CTX)
    assert notices == []


def test_feed_info_absent_emits_no_notice() -> None:
    """No feed_info key in feed → no notice."""
    feed: dict[str, pl.DataFrame] = {}
    notices = validate_feed_service_date(feed, CTX)
    assert notices == []


def test_feed_info_empty_emits_no_notice() -> None:
    """Empty feed_info DataFrame → no notice."""
    feed = {
        "feed_info": pl.DataFrame(
            {
                "feed_start_date": pl.Series([], dtype=pl.Date),
                "feed_end_date": pl.Series([], dtype=pl.Date),
                "csv_row_number": pl.Series([], dtype=pl.Int64),
            }
        )
    }
    notices = validate_feed_service_date(feed, CTX)
    assert notices == []


def test_both_dates_same_value_emits_no_notice() -> None:
    """Both dates present and equal → no notice (equality is not checked here)."""
    feed = make_feed_info(feed_start_date=DATE_A, feed_end_date=DATE_A)
    notices = validate_feed_service_date(feed, CTX)
    assert notices == []


def test_csv_row_number_preserved() -> None:
    """csv_row_number from the row is preserved in the emitted notice."""
    feed = make_feed_info(feed_start_date=DATE_A, feed_end_date=None, csv_row_number=3)
    notices = validate_feed_service_date(feed, CTX)
    assert len(notices) == 1
    assert notices[0].fields["csv_row_number"] == 3


def test_notice_fields_are_complete() -> None:
    """Notice fields contain exactly csv_row_number and field_name."""
    feed = make_feed_info(feed_start_date=None, feed_end_date=DATE_A)
    notices = validate_feed_service_date(feed, CTX)
    assert len(notices) == 1
    assert set(notices[0].fields.keys()) == {"csv_row_number", "field_name"}


def test_ctx_unused() -> None:
    """ctx value has no effect on the output."""
    wrong_ctx = ValidationContext(country_code="XX", date_for_validation=date(1900, 1, 1))
    feed = make_feed_info(feed_start_date=None, feed_end_date=DATE_A)
    notices = validate_feed_service_date(feed, wrong_ctx)
    assert len(notices) == 1
    assert notices[0].fields["field_name"] == "feed_start_date"


def test_multiple_rows_each_checked_independently() -> None:
    """Multiple rows are checked independently; only rows with exactly one date emit."""
    feed = {
        "feed_info": pl.DataFrame(
            {
                "feed_publisher_name": ["Publisher A", "Publisher B"],
                "feed_publisher_url": ["https://a.com", "https://b.com"],
                "feed_lang": ["en", "fr"],
                "feed_start_date": [DATE_A, None],
                "feed_end_date": [None, None],
                "csv_row_number": [1, 2],
            }
        )
    }
    notices = validate_feed_service_date(feed, CTX)
    assert len(notices) == 1
    assert notices[0].fields["csv_row_number"] == 1
    assert notices[0].fields["field_name"] == "feed_end_date"


def test_date_parse_failure_treated_as_absent() -> None:
    """A null feed_start_date (e.g. from parse failure) is treated the same as absent."""
    feed = make_feed_info(feed_start_date=None, feed_end_date=DATE_A)
    notices = validate_feed_service_date(feed, CTX)
    assert len(notices) == 1
    assert notices[0].fields["field_name"] == "feed_start_date"
