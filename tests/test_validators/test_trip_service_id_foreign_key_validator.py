"""Tests for validate_trip_service_id_foreign_key."""

from datetime import date

import polars as pl
import pytest

from gtfs_validator.context import ValidationContext
from gtfs_validator.notices import Severity
from gtfs_validator.validators.trip_service_id_foreign_key_validator import (
    validate_trip_service_id_foreign_key,
)

CTX = ValidationContext(country_code="US", date_for_validation=date(2026, 3, 8))


def make_trips(service_ids: list[str | None]) -> pl.DataFrame:
    return pl.DataFrame({
        "service_id": service_ids,
        "csvRowNumber": list(range(2, 2 + len(service_ids))),
    })


def make_calendar(service_ids: list[str]) -> pl.DataFrame:
    return pl.DataFrame({"service_id": service_ids})


def make_calendar_dates(service_ids: list[str]) -> pl.DataFrame:
    return pl.DataFrame({"service_id": service_ids})


def test_service_id_in_calendar_no_notice():
    feed = {"trips": make_trips(["WEEK"]), "calendar": make_calendar(["WEEK"])}
    assert validate_trip_service_id_foreign_key(feed, CTX) == []


def test_service_id_in_calendar_dates_no_notice():
    feed = {"trips": make_trips(["WEEK"]), "calendar_dates": make_calendar_dates(["WEEK"])}
    assert validate_trip_service_id_foreign_key(feed, CTX) == []


def test_service_id_not_in_data_generates_notice():
    feed = {"trips": make_trips(["WEEK"])}
    notices = validate_trip_service_id_foreign_key(feed, CTX)
    assert len(notices) == 1
    n = notices[0]
    assert n.code == "foreign_key_violation"
    assert n.severity == Severity.ERROR
    assert n.fields["childFilename"] == "trips.txt"
    assert n.fields["childFieldName"] == "service_id"
    assert n.fields["parentFilename"] == "calendar.txt or calendar_dates.txt"
    assert n.fields["parentFieldName"] == "service_id"
    assert n.fields["fieldValue"] == "WEEK"
    assert n.fields["csvRowNumber"] == 2


def test_empty_service_id_skipped():
    feed = {"trips": make_trips([""])}
    assert validate_trip_service_id_foreign_key(feed, CTX) == []


def test_null_service_id_skipped():
    feed = {"trips": make_trips([None])}
    assert validate_trip_service_id_foreign_key(feed, CTX) == []


def test_service_id_in_both_calendar_and_calendar_dates_no_notice():
    feed = {
        "trips": make_trips(["WEEK"]),
        "calendar": make_calendar(["WEEK"]),
        "calendar_dates": make_calendar_dates(["WEEK"]),
    }
    assert validate_trip_service_id_foreign_key(feed, CTX) == []


def test_mixed_valid_and_invalid_rows():
    feed = {
        "trips": make_trips(["WEEK", "MISSING"]),
        "calendar": make_calendar(["WEEK"]),
    }
    notices = validate_trip_service_id_foreign_key(feed, CTX)
    assert len(notices) == 1
    assert notices[0].fields["csvRowNumber"] == 3
    assert notices[0].fields["fieldValue"] == "MISSING"


def test_calendar_absent_calendar_dates_has_match():
    feed = {"trips": make_trips(["WEEK"]), "calendar_dates": make_calendar_dates(["WEEK"])}
    assert validate_trip_service_id_foreign_key(feed, CTX) == []


def test_both_parent_tables_absent():
    feed = {"trips": make_trips(["WEEK"])}
    notices = validate_trip_service_id_foreign_key(feed, CTX)
    assert len(notices) == 1
    assert notices[0].fields["fieldValue"] == "WEEK"


def test_multiple_trips_same_invalid_service_id():
    feed = {"trips": make_trips(["MISSING", "MISSING", "MISSING"])}
    notices = validate_trip_service_id_foreign_key(feed, CTX)
    assert len(notices) == 3
    assert [n.fields["csvRowNumber"] for n in notices] == [2, 3, 4]


def test_multiple_trips_different_invalid_service_ids():
    feed = {"trips": make_trips(["A", "B", "C"])}
    notices = validate_trip_service_id_foreign_key(feed, CTX)
    assert len(notices) == 3
    assert {n.fields["fieldValue"] for n in notices} == {"A", "B", "C"}


def test_trips_empty_no_notices():
    feed = {
        "trips": pl.DataFrame({"service_id": [], "csvRowNumber": []}).cast(
            {"service_id": pl.Utf8, "csvRowNumber": pl.Int64}
        )
    }
    assert validate_trip_service_id_foreign_key(feed, CTX) == []


def test_notice_fields_are_complete():
    feed = {"trips": make_trips(["MISSING"])}
    notices = validate_trip_service_id_foreign_key(feed, CTX)
    assert len(notices) == 1
    expected_keys = {
        "childFilename",
        "childFieldName",
        "parentFilename",
        "parentFieldName",
        "fieldValue",
        "csvRowNumber",
    }
    assert set(notices[0].fields.keys()) == expected_keys
    assert notices[0].fields["childFilename"] == "trips.txt"
    assert notices[0].fields["parentFilename"] == "calendar.txt or calendar_dates.txt"
    assert notices[0].fields["parentFieldName"] == "service_id"


def test_duplicate_service_ids_in_parents_still_valid():
    feed = {
        "trips": make_trips(["WEEK"]),
        "calendar": make_calendar(["WEEK", "WEEK"]),
    }
    assert validate_trip_service_id_foreign_key(feed, CTX) == []
