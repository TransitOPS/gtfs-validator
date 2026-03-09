"""Tests for validate_missing_feed_info."""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from gtfs_validator.context import ValidationContext
from gtfs_validator.notices import Severity
from gtfs_validator.validators.missing_feed_info import validate_missing_feed_info

CTX = ValidationContext(country_code="US", date_for_validation=date(2026, 3, 8))


def make_feed_info() -> pl.DataFrame:
    """Minimal feed_info.txt DataFrame with one row."""
    return pl.DataFrame(
        {
            "feed_publisher_name": ["Test Publisher"],
            "feed_publisher_url": ["https://example.com"],
            "feed_lang": ["en"],
        },
        schema={
            "feed_publisher_name": pl.Utf8,
            "feed_publisher_url": pl.Utf8,
            "feed_lang": pl.Utf8,
        },
    )


def make_translations() -> pl.DataFrame:
    """Minimal translations.txt DataFrame with one row."""
    return pl.DataFrame(
        {
            "table_name": ["stops"],
            "field_name": ["stop_name"],
            "language": ["fr"],
            "translation": ["Gare"],
        },
        schema={
            "table_name": pl.Utf8,
            "field_name": pl.Utf8,
            "language": pl.Utf8,
            "translation": pl.Utf8,
        },
    )


def make_empty_feed_info() -> pl.DataFrame:
    """feed_info.txt present but zero data rows."""
    return pl.DataFrame(schema={"feed_publisher_name": pl.Utf8, "feed_lang": pl.Utf8})


def make_empty_translations() -> pl.DataFrame:
    """translations.txt present but zero data rows."""
    return pl.DataFrame(
        schema={
            "table_name": pl.Utf8,
            "field_name": pl.Utf8,
            "language": pl.Utf8,
            "translation": pl.Utf8,
        }
    )


def test_feed_info_present_translations_present_no_notice() -> None:
    feed = {"feed_info": make_feed_info(), "translations": make_translations()}
    notices = validate_missing_feed_info(feed, CTX)
    assert notices == []


def test_feed_info_present_translations_absent_no_notice() -> None:
    feed = {"feed_info": make_feed_info()}
    notices = validate_missing_feed_info(feed, CTX)
    assert notices == []


def test_feed_info_absent_translations_absent_emits_warning() -> None:
    feed: dict = {}
    notices = validate_missing_feed_info(feed, CTX)
    assert len(notices) == 1
    assert notices[0].code == "missing_recommended_file"
    assert notices[0].severity == Severity.WARNING
    assert notices[0].fields == {"filename": "feed_info.txt"}


def test_feed_info_absent_translations_present_emits_error() -> None:
    feed = {"translations": make_translations()}
    notices = validate_missing_feed_info(feed, CTX)
    assert len(notices) == 1
    assert notices[0].code == "missing_required_file"
    assert notices[0].severity == Severity.ERROR
    assert notices[0].fields == {"filename": "feed_info.txt"}


def test_feed_info_present_empty_dataframe_no_notice() -> None:
    feed = {"feed_info": make_empty_feed_info()}
    notices = validate_missing_feed_info(feed, CTX)
    assert notices == []


def test_translations_present_empty_dataframe_triggers_error_when_feed_info_absent() -> None:
    feed = {"translations": make_empty_translations()}
    notices = validate_missing_feed_info(feed, CTX)
    assert len(notices) == 1
    assert notices[0].code == "missing_required_file"
    assert notices[0].severity == Severity.ERROR


def test_ctx_unused_does_not_affect_result() -> None:
    ctx1 = ValidationContext(country_code="US", date_for_validation=date(2026, 1, 1))
    ctx2 = ValidationContext(country_code="US", date_for_validation=date(2026, 12, 31))
    feed: dict = {}
    notices1 = validate_missing_feed_info(feed, ctx1)
    notices2 = validate_missing_feed_info(feed, ctx2)
    assert len(notices1) == 1
    assert len(notices2) == 1
    assert notices1[0].code == notices2[0].code
    assert notices1[0].severity == notices2[0].severity
    assert notices1[0].fields == notices2[0].fields


def test_warning_filename_field_value() -> None:
    feed: dict = {}
    notices = validate_missing_feed_info(feed, CTX)
    assert notices[0].fields["filename"] == "feed_info.txt"


def test_error_filename_field_value() -> None:
    feed = {"translations": make_translations()}
    notices = validate_missing_feed_info(feed, CTX)
    assert notices[0].fields["filename"] == "feed_info.txt"
