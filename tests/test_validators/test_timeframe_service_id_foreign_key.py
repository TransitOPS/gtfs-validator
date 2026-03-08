"""Tests for validate_timeframe_service_id_foreign_key."""

from datetime import date

import polars as pl
import pytest

from gtfs_validator.context import ValidationContext
from gtfs_validator.notices import Severity
from gtfs_validator.validators.timeframe_service_id_foreign_key import (
    validate_timeframe_service_id_foreign_key,
)

CTX = ValidationContext(country_code="US", date_for_validation=date(2026, 3, 8))


def make_timeframes(service_ids: list[str | None], row_numbers: list[int] | None = None) -> pl.DataFrame:
    if row_numbers is None:
        row_numbers = list(range(2, 2 + len(service_ids)))
    return pl.DataFrame({
        "service_id": service_ids,
        "csvRowNumber": row_numbers,
    })


def make_calendar(service_ids: list[str]) -> pl.DataFrame:
    return pl.DataFrame({"service_id": service_ids})


def make_calendar_dates(service_ids: list[str]) -> pl.DataFrame:
    return pl.DataFrame({"service_id": service_ids})


def test_service_id_in_calendar_no_notice() -> None:
    feed = {
        "timeframes": make_timeframes(["WEEK"]),
        "calendar": make_calendar(["WEEK"]),
    }
    notices = validate_timeframe_service_id_foreign_key(feed, CTX)
    assert notices == []


def test_service_id_in_calendar_dates_no_notice() -> None:
    feed = {
        "timeframes": make_timeframes(["WEEK"]),
        "calendar_dates": make_calendar_dates(["WEEK"]),
    }
    notices = validate_timeframe_service_id_foreign_key(feed, CTX)
    assert notices == []


def test_service_id_not_in_data_generates_notice() -> None:
    feed = {
        "timeframes": make_timeframes(["WEEK"], [2]),
    }
    notices = validate_timeframe_service_id_foreign_key(feed, CTX)
    assert len(notices) == 1
    n = notices[0]
    assert n.code == "foreign_key_violation"
    assert n.severity == Severity.ERROR
    assert n.fields["childFilename"] == "timeframes.txt"
    assert n.fields["childFieldName"] == "service_id"
    assert n.fields["parentFilename"] == "calendar.txt or calendar_dates.txt"
    assert n.fields["parentFieldName"] == "service_id"
    assert n.fields["fieldValue"] == "WEEK"
    assert n.fields["csvRowNumber"] == 2


def test_timeframes_absent_no_notice() -> None:
    notices = validate_timeframe_service_id_foreign_key({}, CTX)
    assert notices == []


def test_timeframes_empty_no_notice() -> None:
    feed = {"timeframes": make_timeframes([])}
    notices = validate_timeframe_service_id_foreign_key(feed, CTX)
    assert notices == []


def test_service_id_in_both_parents_no_notice() -> None:
    feed = {
        "timeframes": make_timeframes(["WEEK"]),
        "calendar": make_calendar(["WEEK"]),
        "calendar_dates": make_calendar_dates(["WEEK"]),
    }
    notices = validate_timeframe_service_id_foreign_key(feed, CTX)
    assert notices == []


def test_calendar_absent_calendar_dates_has_match() -> None:
    feed = {
        "timeframes": make_timeframes(["WEEK"]),
        "calendar_dates": make_calendar_dates(["WEEK"]),
    }
    notices = validate_timeframe_service_id_foreign_key(feed, CTX)
    assert notices == []


def test_calendar_dates_absent_calendar_has_match() -> None:
    feed = {
        "timeframes": make_timeframes(["WEEK"]),
        "calendar": make_calendar(["WEEK"]),
    }
    notices = validate_timeframe_service_id_foreign_key(feed, CTX)
    assert notices == []


def test_both_parents_absent_all_rows_violate() -> None:
    feed = {"timeframes": make_timeframes(["WEEK", "WEEKEND"])}
    notices = validate_timeframe_service_id_foreign_key(feed, CTX)
    assert len(notices) == 2
    assert {n.fields["fieldValue"] for n in notices} == {"WEEK", "WEEKEND"}


def test_same_invalid_service_id_multiple_rows_one_notice_per_row() -> None:
    feed = {
        "timeframes": make_timeframes(["MISSING", "MISSING", "MISSING"], [2, 3, 4]),
    }
    notices = validate_timeframe_service_id_foreign_key(feed, CTX)
    assert len(notices) == 3
    assert [n.fields["csvRowNumber"] for n in notices] == [2, 3, 4]


def test_mixed_valid_and_invalid_rows() -> None:
    feed = {
        "timeframes": make_timeframes(["WEEK", "MISSING"], [2, 3]),
        "calendar": make_calendar(["WEEK"]),
    }
    notices = validate_timeframe_service_id_foreign_key(feed, CTX)
    assert len(notices) == 1
    assert notices[0].fields["fieldValue"] == "MISSING"
    assert notices[0].fields["csvRowNumber"] == 3


def test_multiple_different_invalid_service_ids() -> None:
    feed = {
        "timeframes": make_timeframes(["A", "B", "C"]),
    }
    notices = validate_timeframe_service_id_foreign_key(feed, CTX)
    assert len(notices) == 3
    assert {n.fields["fieldValue"] for n in notices} == {"A", "B", "C"}


def test_null_service_id_skipped() -> None:
    feed = {
        "timeframes": make_timeframes([None]),
    }
    notices = validate_timeframe_service_id_foreign_key(feed, CTX)
    assert notices == []


def test_empty_string_service_id_skipped() -> None:
    feed = {
        "timeframes": make_timeframes([""]),
    }
    notices = validate_timeframe_service_id_foreign_key(feed, CTX)
    assert notices == []


def test_notice_fields_are_complete() -> None:
    feed = {"timeframes": make_timeframes(["MISSING"])}
    notices = validate_timeframe_service_id_foreign_key(feed, CTX)
    assert len(notices) == 1
    assert set(notices[0].fields.keys()) == {
        "childFilename", "childFieldName", "parentFilename", "parentFieldName",
        "fieldValue", "csvRowNumber",
    }
    assert notices[0].fields["childFilename"] == "timeframes.txt"
    assert notices[0].fields["childFieldName"] == "service_id"
    assert notices[0].fields["parentFilename"] == "calendar.txt or calendar_dates.txt"
    assert notices[0].fields["parentFieldName"] == "service_id"


def test_duplicate_service_ids_in_parent_still_valid() -> None:
    feed = {
        "timeframes": make_timeframes(["WEEK"]),
        "calendar": make_calendar(["WEEK", "WEEK"]),
    }
    notices = validate_timeframe_service_id_foreign_key(feed, CTX)
    assert notices == []
