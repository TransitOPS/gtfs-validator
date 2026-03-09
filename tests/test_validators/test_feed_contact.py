"""Tests for validate_feed_contact."""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from gtfs_validator.context import ValidationContext
from gtfs_validator.notices import Severity
from gtfs_validator.validators.feed_contact import validate_feed_contact

CTX = ValidationContext(country_code="US", date_for_validation=date(2024, 1, 1))


def make_feed_info(
    feed_contact_email: str | None,
    feed_contact_url: str | None,
    csv_row_number: int = 2,
) -> pl.DataFrame:
    """Build a minimal feed_info DataFrame with the two contact fields."""
    return pl.DataFrame(
        {
            "feed_contact_email": pl.Series([feed_contact_email], dtype=pl.Utf8),
            "feed_contact_url": pl.Series([feed_contact_url], dtype=pl.Utf8),
            "csv_row_number": [csv_row_number],
        }
    )


def test_both_fields_set_no_notice() -> None:
    feed = {"feed_info": make_feed_info("email@gmail.com", "https://example.com", csv_row_number=2)}
    assert validate_feed_contact(feed, CTX) == []


def test_email_set_url_blank_no_notice() -> None:
    feed = {"feed_info": make_feed_info("email@gmail.com", "", csv_row_number=2)}
    assert validate_feed_contact(feed, CTX) == []


def test_email_set_url_null_no_notice() -> None:
    feed = {"feed_info": make_feed_info("email@gmail.com", None, csv_row_number=2)}
    assert validate_feed_contact(feed, CTX) == []


def test_url_set_email_blank_no_notice() -> None:
    feed = {"feed_info": make_feed_info("", "https://example.com", csv_row_number=2)}
    assert validate_feed_contact(feed, CTX) == []


def test_url_set_email_null_no_notice() -> None:
    feed = {"feed_info": make_feed_info(None, "https://example.com", csv_row_number=2)}
    assert validate_feed_contact(feed, CTX) == []


def test_both_fields_blank_emits_notice() -> None:
    feed = {"feed_info": make_feed_info("", "", csv_row_number=2)}
    notices = validate_feed_contact(feed, CTX)
    assert len(notices) == 1
    assert notices[0].code == "missing_feed_contact_email_and_url"
    assert notices[0].severity == Severity.WARNING
    assert notices[0].fields["csv_row_number"] == 2


def test_both_fields_null_emits_notice() -> None:
    feed = {"feed_info": make_feed_info(None, None, csv_row_number=2)}
    notices = validate_feed_contact(feed, CTX)
    assert len(notices) == 1
    assert notices[0].code == "missing_feed_contact_email_and_url"
    assert notices[0].severity == Severity.WARNING
    assert notices[0].fields["csv_row_number"] == 2


def test_email_null_url_blank_emits_notice() -> None:
    feed = {"feed_info": make_feed_info(None, "", csv_row_number=2)}
    notices = validate_feed_contact(feed, CTX)
    assert len(notices) == 1
    assert notices[0].code == "missing_feed_contact_email_and_url"
    assert notices[0].severity == Severity.WARNING
    assert notices[0].fields["csv_row_number"] == 2


def test_email_whitespace_url_null_emits_notice() -> None:
    feed = {"feed_info": make_feed_info("   ", None, csv_row_number=2)}
    notices = validate_feed_contact(feed, CTX)
    assert len(notices) == 1
    assert notices[0].code == "missing_feed_contact_email_and_url"
    assert notices[0].severity == Severity.WARNING
    assert notices[0].fields["csv_row_number"] == 2


def test_email_null_url_whitespace_emits_notice() -> None:
    feed = {"feed_info": make_feed_info(None, "  ", csv_row_number=2)}
    notices = validate_feed_contact(feed, CTX)
    assert len(notices) == 1
    assert notices[0].code == "missing_feed_contact_email_and_url"
    assert notices[0].severity == Severity.WARNING
    assert notices[0].fields["csv_row_number"] == 2


def test_feed_info_absent_skips() -> None:
    assert validate_feed_contact({}, CTX) == []


def test_feed_info_empty_skips() -> None:
    empty_df = pl.DataFrame(
        {
            "feed_contact_email": pl.Series([], dtype=pl.Utf8),
            "feed_contact_url": pl.Series([], dtype=pl.Utf8),
            "csv_row_number": pl.Series([], dtype=pl.Int64),
        }
    )
    assert validate_feed_contact({"feed_info": empty_df}, CTX) == []


def test_notice_fields_exact() -> None:
    feed = {"feed_info": make_feed_info(None, None, csv_row_number=7)}
    notices = validate_feed_contact(feed, CTX)
    assert len(notices) == 1
    assert notices[0].code == "missing_feed_contact_email_and_url"
    assert notices[0].severity == Severity.WARNING
    assert notices[0].fields == {"csv_row_number": 7}


def test_multiple_rows_each_checked_independently() -> None:
    df = pl.DataFrame(
        {
            "feed_contact_email": pl.Series([None, "contact@example.com"], dtype=pl.Utf8),
            "feed_contact_url": pl.Series([None, None], dtype=pl.Utf8),
            "csv_row_number": [2, 3],
        }
    )
    notices = validate_feed_contact({"feed_info": df}, CTX)
    assert len(notices) == 1
    assert notices[0].fields["csv_row_number"] == 2
