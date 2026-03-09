"""Tests for validate_missing_calendar_and_calendar_date."""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from gtfs_validator.context import ValidationContext
from gtfs_validator.notices import Severity
from gtfs_validator.validators.missing_calendar_and_calendar_date import (
    validate_missing_calendar_and_calendar_date,
)

CTX = ValidationContext(country_code="US", date_for_validation=date(2026, 3, 8))


def make_calendar() -> pl.DataFrame:
    """Minimal calendar.txt DataFrame with one service row."""
    return pl.DataFrame(
        {
            "service_id": ["WEEK"],
            "monday": [1],
            "tuesday": [1],
            "wednesday": [1],
            "thursday": [1],
            "friday": [1],
            "saturday": [0],
            "sunday": [0],
            "start_date": ["20210104"],
            "end_date": ["20210410"],
        },
        schema={
            "service_id": pl.Utf8,
            "monday": pl.Int64,
            "tuesday": pl.Int64,
            "wednesday": pl.Int64,
            "thursday": pl.Int64,
            "friday": pl.Int64,
            "saturday": pl.Int64,
            "sunday": pl.Int64,
            "start_date": pl.Utf8,
            "end_date": pl.Utf8,
        },
    )


def make_calendar_dates() -> pl.DataFrame:
    """Minimal calendar_dates.txt DataFrame with one exception row."""
    return pl.DataFrame(
        {
            "service_id": ["WEEK"],
            "date": ["20210101"],
            "exception_type": [2],
        },
        schema={
            "service_id": pl.Utf8,
            "date": pl.Utf8,
            "exception_type": pl.Int64,
        },
    )


def make_empty_calendar() -> pl.DataFrame:
    """calendar.txt present but zero data rows (headers only)."""
    return pl.DataFrame(
        schema={
            "service_id": pl.Utf8,
            "monday": pl.Int64,
        }
    )


def make_empty_calendar_dates() -> pl.DataFrame:
    """calendar_dates.txt present but zero data rows."""
    return pl.DataFrame(
        schema={
            "service_id": pl.Utf8,
            "date": pl.Utf8,
            "exception_type": pl.Int64,
        }
    )


def test_both_files_provided_no_notice() -> None:
    """Both calendar and calendar_dates present — no notice expected."""
    feed = {"calendar": make_calendar(), "calendar_dates": make_calendar_dates()}
    notices = validate_missing_calendar_and_calendar_date(feed, CTX)
    assert notices == []


def test_calendar_only_provided_no_notice() -> None:
    """calendar_dates key present (even empty); no notice expected."""
    feed = {"calendar": make_calendar(), "calendar_dates": make_empty_calendar_dates()}
    notices = validate_missing_calendar_and_calendar_date(feed, CTX)
    assert notices == []


def test_calendar_dates_only_provided_no_notice() -> None:
    """calendar key present (even empty); no notice expected."""
    feed = {"calendar": make_empty_calendar(), "calendar_dates": make_calendar_dates()}
    notices = validate_missing_calendar_and_calendar_date(feed, CTX)
    assert notices == []


def test_both_files_missing_generates_notice() -> None:
    """Neither key present — exactly one ERROR notice expected."""
    feed: dict[str, pl.DataFrame] = {}
    notices = validate_missing_calendar_and_calendar_date(feed, CTX)
    assert len(notices) == 1
    assert notices[0].code == "missing_calendar_and_calendar_date_files"
    assert notices[0].severity == Severity.ERROR
    assert notices[0].fields == {}


def test_only_calendar_missing_no_notice() -> None:
    """calendar_dates present, calendar absent — no notice."""
    feed = {"calendar_dates": make_calendar_dates()}
    notices = validate_missing_calendar_and_calendar_date(feed, CTX)
    assert notices == []


def test_only_calendar_dates_missing_no_notice() -> None:
    """calendar present, calendar_dates absent — no notice."""
    feed = {"calendar": make_calendar()}
    notices = validate_missing_calendar_and_calendar_date(feed, CTX)
    assert notices == []


def test_both_empty_dataframes_present_no_notice() -> None:
    """Both keys present with empty DataFrames — no notice expected."""
    feed = {
        "calendar": make_empty_calendar(),
        "calendar_dates": make_empty_calendar_dates(),
    }
    notices = validate_missing_calendar_and_calendar_date(feed, CTX)
    assert notices == []


def test_notice_is_error_severity() -> None:
    """Notice must be ERROR severity, not WARNING or INFO."""
    feed: dict[str, pl.DataFrame] = {}
    notices = validate_missing_calendar_and_calendar_date(feed, CTX)
    assert len(notices) == 1
    assert notices[0].severity == Severity.ERROR


def test_notice_has_empty_fields() -> None:
    """Notice must carry no additional fields."""
    feed: dict[str, pl.DataFrame] = {}
    notices = validate_missing_calendar_and_calendar_date(feed, CTX)
    assert len(notices) == 1
    assert notices[0].fields == {}


def test_ctx_unused_does_not_affect_result() -> None:
    """Result must be identical regardless of context date."""
    feed: dict[str, pl.DataFrame] = {}
    ctx_future = ValidationContext(country_code="US", date_for_validation=date(2099, 1, 1))
    ctx_past = ValidationContext(country_code="US", date_for_validation=date(2000, 1, 1))
    notices_future = validate_missing_calendar_and_calendar_date(feed, ctx_future)
    notices_past = validate_missing_calendar_and_calendar_date(feed, ctx_past)
    assert len(notices_future) == 1
    assert len(notices_past) == 1
    assert notices_future[0].code == notices_past[0].code
    assert notices_future[0].fields == notices_past[0].fields
