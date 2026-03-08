"""Tests for validate_fare_attribute_agency_id."""

from __future__ import annotations

import datetime

import polars as pl
import pytest

from gtfs_validator.context import ValidationContext
from gtfs_validator.notices import Severity
from gtfs_validator.validators.fare_attribute_agency_id import (
    validate_fare_attribute_agency_id,
)

CTX = ValidationContext(
    country_code="US",
    date_for_validation=datetime.date(2024, 1, 1),
)


def make_agency(agency_ids: list[str | None]) -> pl.DataFrame:
    return pl.DataFrame({"agency_id": agency_ids})


def make_fare_attributes(
    agency_ids: list[str | None],
    fare_ids: list[str] | None = None,
) -> pl.DataFrame:
    n = len(agency_ids)
    if fare_ids is None:
        fare_ids = [f"fare{i}" for i in range(n)]
    return pl.DataFrame({"fare_id": fare_ids, "agency_id": agency_ids})


# ---------------------------------------------------------------------------
# Test 1
# ---------------------------------------------------------------------------


def test_multi_agency_one_fare_missing_agency_id() -> None:
    feed = {
        "agency": make_agency(["agency1", "agency2"]),
        "fare_attributes": make_fare_attributes(["agency1", None]),
    }
    notices = validate_fare_attribute_agency_id(feed, CTX)
    assert len(notices) == 1
    n = notices[0]
    assert n.code == "missing_required_agency_id"
    assert n.fields["filename"] == "fare_attributes.txt"
    assert n.fields["csv_row_number"] == 1
    assert n.fields["agency_name"] is None


# ---------------------------------------------------------------------------
# Test 2
# ---------------------------------------------------------------------------


def test_single_agency_fare_missing_agency_id() -> None:
    feed = {
        "agency": make_agency([None]),
        "fare_attributes": make_fare_attributes([None]),
    }
    notices = validate_fare_attribute_agency_id(feed, CTX)
    assert len(notices) == 1
    n = notices[0]
    assert n.code == "missing_recommended_field"
    assert n.fields["filename"] == "fare_attributes.txt"
    assert n.fields["csv_row_number"] == 0
    assert n.fields["field_name"] == "agency_id"


# ---------------------------------------------------------------------------
# Test 3
# ---------------------------------------------------------------------------


def test_multi_agency_all_fares_have_agency_id() -> None:
    feed = {
        "agency": make_agency(["agency1", "agency2"]),
        "fare_attributes": make_fare_attributes(["agency1", "agency1"]),
    }
    notices = validate_fare_attribute_agency_id(feed, CTX)
    assert notices == []


# ---------------------------------------------------------------------------
# Test 4
# ---------------------------------------------------------------------------


def test_single_agency_fare_has_agency_id() -> None:
    feed = {
        "agency": make_agency(["agency1"]),
        "fare_attributes": make_fare_attributes(["agency1", "agency1"]),
    }
    notices = validate_fare_attribute_agency_id(feed, CTX)
    assert notices == []


# ---------------------------------------------------------------------------
# Test 5
# ---------------------------------------------------------------------------


def test_agency_table_none_skips() -> None:
    feed = {
        "fare_attributes": make_fare_attributes(["agency1"]),
    }
    notices = validate_fare_attribute_agency_id(feed, CTX)
    assert notices == []


# ---------------------------------------------------------------------------
# Test 6
# ---------------------------------------------------------------------------


def test_agency_table_empty_skips() -> None:
    feed = {
        "agency": pl.DataFrame({"agency_id": pl.Series([], dtype=pl.Utf8)}),
        "fare_attributes": make_fare_attributes([None]),
    }
    notices = validate_fare_attribute_agency_id(feed, CTX)
    assert notices == []


# ---------------------------------------------------------------------------
# Test 7
# ---------------------------------------------------------------------------


def test_fare_attributes_table_none_skips() -> None:
    feed = {
        "agency": make_agency(["agency1"]),
    }
    notices = validate_fare_attribute_agency_id(feed, CTX)
    assert notices == []


# ---------------------------------------------------------------------------
# Test 8
# ---------------------------------------------------------------------------


def test_fare_attributes_table_empty_skips() -> None:
    feed = {
        "agency": make_agency(["agency1"]),
        "fare_attributes": pl.DataFrame(
            {"fare_id": pl.Series([], dtype=pl.Utf8), "agency_id": pl.Series([], dtype=pl.Utf8)}
        ),
    }
    notices = validate_fare_attribute_agency_id(feed, CTX)
    assert notices == []


# ---------------------------------------------------------------------------
# Test 9
# ---------------------------------------------------------------------------


def test_both_tables_non_empty_validation_runs() -> None:
    feed = {
        "agency": make_agency(["agency1"]),
        "fare_attributes": make_fare_attributes(["a"]),
    }
    notices = validate_fare_attribute_agency_id(feed, CTX)
    assert notices == []


# ---------------------------------------------------------------------------
# Test 10
# ---------------------------------------------------------------------------


def test_multi_agency_multiple_fares_all_missing_agency_id() -> None:
    feed = {
        "agency": make_agency(["agency1", "agency2"]),
        "fare_attributes": make_fare_attributes([None, None, None]),
    }
    notices = validate_fare_attribute_agency_id(feed, CTX)
    assert len(notices) == 3
    for i, n in enumerate(notices):
        assert n.code == "missing_required_agency_id"
        assert n.fields["csv_row_number"] == i


# ---------------------------------------------------------------------------
# Test 11
# ---------------------------------------------------------------------------


def test_empty_string_agency_id_treated_as_missing_multi_agency() -> None:
    feed = {
        "agency": make_agency(["agency1", "agency2"]),
        "fare_attributes": make_fare_attributes([""]),
    }
    notices = validate_fare_attribute_agency_id(feed, CTX)
    assert len(notices) == 1
    assert notices[0].code == "missing_required_agency_id"
    assert notices[0].fields["csv_row_number"] == 0


# ---------------------------------------------------------------------------
# Test 12
# ---------------------------------------------------------------------------


def test_empty_string_agency_id_treated_as_missing_single_agency() -> None:
    feed = {
        "agency": make_agency(["agency1"]),
        "fare_attributes": make_fare_attributes([""]),
    }
    notices = validate_fare_attribute_agency_id(feed, CTX)
    assert len(notices) == 1
    assert notices[0].code == "missing_recommended_field"
    assert notices[0].fields["csv_row_number"] == 0
    assert notices[0].fields["field_name"] == "agency_id"


# ---------------------------------------------------------------------------
# Test 13
# ---------------------------------------------------------------------------


def test_multi_agency_some_missing_some_present() -> None:
    feed = {
        "agency": make_agency(["agency1", "agency2"]),
        "fare_attributes": make_fare_attributes(["agency1", None, "agency2"]),
    }
    notices = validate_fare_attribute_agency_id(feed, CTX)
    assert len(notices) == 1
    assert notices[0].code == "missing_required_agency_id"
    assert notices[0].fields["csv_row_number"] == 1


# ---------------------------------------------------------------------------
# Test 14
# ---------------------------------------------------------------------------


def test_error_severity_multi_agency() -> None:
    feed = {
        "agency": make_agency(["agency1", "agency2"]),
        "fare_attributes": make_fare_attributes([None]),
    }
    notices = validate_fare_attribute_agency_id(feed, CTX)
    assert len(notices) == 1
    assert notices[0].severity == Severity.ERROR


# ---------------------------------------------------------------------------
# Test 15
# ---------------------------------------------------------------------------


def test_warning_severity_single_agency() -> None:
    feed = {
        "agency": make_agency(["agency1"]),
        "fare_attributes": make_fare_attributes([None]),
    }
    notices = validate_fare_attribute_agency_id(feed, CTX)
    assert len(notices) == 1
    assert notices[0].severity == Severity.WARNING


# ---------------------------------------------------------------------------
# Test 16
# ---------------------------------------------------------------------------


def test_agency_name_always_none() -> None:
    feed = {
        "agency": pl.DataFrame({"agency_id": ["a1", "a2"], "agency_name": ["Metro", "Metro"]}),
        "fare_attributes": make_fare_attributes([None]),
    }
    notices = validate_fare_attribute_agency_id(feed, CTX)
    assert len(notices) == 1
    assert notices[0].fields["agency_name"] is None


# ---------------------------------------------------------------------------
# Test 17
# ---------------------------------------------------------------------------


def test_notice_filename_field() -> None:
    feed = {
        "agency": make_agency(["agency1", "agency2"]),
        "fare_attributes": make_fare_attributes([None, None]),
    }
    notices = validate_fare_attribute_agency_id(feed, CTX)
    assert all(n.fields["filename"] == "fare_attributes.txt" for n in notices)
