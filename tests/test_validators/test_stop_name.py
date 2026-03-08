"""Tests for validate_stop_name."""

from datetime import date

import polars as pl
import pytest

from gtfs_validator.context import ValidationContext
from gtfs_validator.notices import Severity
from gtfs_validator.validators.stop_name import validate_stop_name

CTX = ValidationContext(country_code="US", date_for_validation=date(2024, 1, 1))

BASE_SCHEMA = {
    "csv_row_number": pl.Int64,
    "stop_id": pl.Utf8,
    "stop_name": pl.Utf8,
    "stop_desc": pl.Utf8,
    "location_type": pl.Int64,
}


def make_feed(rows: list[dict], schema: dict | None = None) -> dict[str, pl.DataFrame]:
    """Build a minimal feed dict with a stops DataFrame."""
    s = schema or BASE_SCHEMA
    if not rows:
        return {"stops": pl.DataFrame(schema=s)}
    return {"stops": pl.DataFrame(rows).cast(s)}


# ---------------------------------------------------------------------------
# same_name_and_description_for_stop
# ---------------------------------------------------------------------------


def test_same_stop_name_and_desc_generates_notice() -> None:
    feed = make_feed(
        [
            {
                "csv_row_number": 4,
                "stop_id": "stop id value",
                "location_type": 0,
                "stop_name": "duplicate value",
                "stop_desc": "duplicate value",
            }
        ]
    )
    notices = validate_stop_name(feed, CTX)
    assert len(notices) == 1
    n = notices[0]
    assert n.code == "same_name_and_description_for_stop"
    assert n.severity == Severity.WARNING
    assert n.fields["csv_row_number"] == 4
    assert n.fields["stop_id"] == "stop id value"
    assert n.fields["stop_desc"] == "duplicate value"
    assert "stop_name" not in n.fields


def test_different_stop_name_and_desc_no_notice() -> None:
    feed = make_feed(
        [
            {
                "csv_row_number": 4,
                "stop_id": "stop id value",
                "location_type": 0,
                "stop_name": "stop name value",
                "stop_desc": "stop desc value",
            }
        ]
    )
    notices = validate_stop_name(feed, CTX)
    assert notices == []


# ---------------------------------------------------------------------------
# missing_stop_name
# ---------------------------------------------------------------------------


def test_missing_stop_name_for_stop_generates_notice() -> None:
    feed = make_feed(
        [
            {
                "csv_row_number": 4,
                "stop_id": "stop id value",
                "location_type": 0,
                "stop_name": None,
                "stop_desc": None,
            }
        ]
    )
    notices = validate_stop_name(feed, CTX)
    assert len(notices) == 1
    n = notices[0]
    assert n.code == "missing_stop_name"
    assert n.severity == Severity.ERROR
    assert n.fields["csv_row_number"] == 4
    assert n.fields["stop_id"] == "stop id value"
    assert n.fields["location_type"] == 0


def test_missing_stop_name_for_station_generates_notice() -> None:
    feed = make_feed(
        [
            {
                "csv_row_number": 4,
                "stop_id": "stop id value",
                "location_type": 1,
                "stop_name": None,
                "stop_desc": None,
            }
        ]
    )
    notices = validate_stop_name(feed, CTX)
    assert len(notices) == 1
    assert notices[0].code == "missing_stop_name"
    assert notices[0].fields["location_type"] == 1


def test_missing_stop_name_for_entrance_generates_notice() -> None:
    feed = make_feed(
        [
            {
                "csv_row_number": 4,
                "stop_id": "stop id value",
                "location_type": 2,
                "stop_name": None,
                "stop_desc": None,
            }
        ]
    )
    notices = validate_stop_name(feed, CTX)
    assert len(notices) == 1
    assert notices[0].code == "missing_stop_name"
    assert notices[0].fields["location_type"] == 2


def test_missing_stop_name_for_generic_node_no_notice() -> None:
    feed = make_feed(
        [
            {
                "csv_row_number": 4,
                "stop_id": "stop id value",
                "location_type": 3,
                "stop_name": None,
                "stop_desc": None,
            }
        ]
    )
    notices = validate_stop_name(feed, CTX)
    assert notices == []


def test_missing_stop_name_for_boarding_area_no_notice() -> None:
    feed = make_feed(
        [
            {
                "csv_row_number": 4,
                "stop_id": "stop id value",
                "location_type": 4,
                "stop_name": None,
                "stop_desc": None,
            }
        ]
    )
    notices = validate_stop_name(feed, CTX)
    assert notices == []


def test_missing_stop_name_for_unrecognized_location_type_no_notice() -> None:
    feed = make_feed(
        [
            {
                "csv_row_number": 4,
                "stop_id": "stop id value",
                "location_type": 99,
                "stop_name": None,
                "stop_desc": None,
            }
        ]
    )
    notices = validate_stop_name(feed, CTX)
    assert notices == []


# ---------------------------------------------------------------------------
# stop_desc edge cases
# ---------------------------------------------------------------------------


def test_missing_stop_desc_no_notice() -> None:
    feed = make_feed(
        [
            {
                "csv_row_number": 4,
                "stop_id": "stop id value",
                "location_type": 0,
                "stop_name": "stop name value",
                "stop_desc": None,
            }
        ]
    )
    notices = validate_stop_name(feed, CTX)
    assert notices == []


def test_stop_name_null_but_stop_desc_present_no_same_name_notice() -> None:
    feed = make_feed(
        [
            {
                "csv_row_number": 4,
                "stop_id": "stop id value",
                "location_type": 3,
                "stop_name": None,
                "stop_desc": "some desc",
            }
        ]
    )
    notices = validate_stop_name(feed, CTX)
    assert notices == []


def test_case_insensitive_name_desc_match_generates_warning() -> None:
    feed = make_feed(
        [
            {
                "csv_row_number": 4,
                "stop_id": "stop id value",
                "location_type": 0,
                "stop_name": "Stop",
                "stop_desc": "stop",
            }
        ]
    )
    notices = validate_stop_name(feed, CTX)
    assert len(notices) == 1
    assert notices[0].code == "same_name_and_description_for_stop"


# ---------------------------------------------------------------------------
# Guard clauses
# ---------------------------------------------------------------------------


def test_stops_absent_no_notice() -> None:
    notices = validate_stop_name({}, CTX)
    assert notices == []


def test_stops_empty_no_notice() -> None:
    feed = make_feed([])
    notices = validate_stop_name(feed, CTX)
    assert notices == []


def test_both_stop_name_and_location_type_columns_absent_no_notice() -> None:
    schema = {
        "csv_row_number": pl.Int64,
        "stop_id": pl.Utf8,
        "stop_desc": pl.Utf8,
    }
    stops = pl.DataFrame(
        [{"csv_row_number": 1, "stop_id": "s1", "stop_desc": "desc"}]
    ).cast(schema)
    feed = {"stops": stops}
    notices = validate_stop_name(feed, CTX)
    assert notices == []


def test_location_type_column_absent_but_stop_name_present_check_b_still_runs() -> None:
    schema = {
        "csv_row_number": pl.Int64,
        "stop_id": pl.Utf8,
        "stop_name": pl.Utf8,
        "stop_desc": pl.Utf8,
    }
    stops = pl.DataFrame(
        [{"csv_row_number": 1, "stop_id": "s1", "stop_name": "dup", "stop_desc": "dup"}]
    ).cast(schema)
    feed = {"stops": stops}
    notices = validate_stop_name(feed, CTX)
    assert len(notices) == 1
    assert notices[0].code == "same_name_and_description_for_stop"


def test_stop_name_column_absent_but_location_type_present_no_notices() -> None:
    schema = {
        "csv_row_number": pl.Int64,
        "stop_id": pl.Utf8,
        "location_type": pl.Int64,
    }
    stops = pl.DataFrame(
        [{"csv_row_number": 1, "stop_id": "s1", "location_type": 0}]
    ).cast(schema)
    feed = {"stops": stops}
    notices = validate_stop_name(feed, CTX)
    assert notices == []


# ---------------------------------------------------------------------------
# Default location_type
# ---------------------------------------------------------------------------


def test_location_type_defaults_to_stop_triggers_missing_name_notice() -> None:
    feed = make_feed(
        [
            {
                "csv_row_number": 4,
                "stop_id": "stop id value",
                "location_type": 0,
                "stop_name": None,
                "stop_desc": None,
            }
        ]
    )
    notices = validate_stop_name(feed, CTX)
    assert len(notices) == 1
    assert notices[0].code == "missing_stop_name"
    assert notices[0].fields["location_type"] == 0


# ---------------------------------------------------------------------------
# Multiple rows
# ---------------------------------------------------------------------------


def test_multiple_offending_rows_emit_multiple_notices() -> None:
    feed = make_feed(
        [
            {
                "csv_row_number": 1,
                "stop_id": "s1",
                "location_type": 0,
                "stop_name": None,
                "stop_desc": None,
            },
            {
                "csv_row_number": 2,
                "stop_id": "s2",
                "location_type": 1,
                "stop_name": None,
                "stop_desc": None,
            },
            {
                "csv_row_number": 3,
                "stop_id": "s3",
                "location_type": 0,
                "stop_name": "dup",
                "stop_desc": "dup",
            },
        ]
    )
    notices = validate_stop_name(feed, CTX)
    assert len(notices) == 3
    codes = [n.code for n in notices]
    assert codes.count("missing_stop_name") == 2
    assert codes.count("same_name_and_description_for_stop") == 1


# ---------------------------------------------------------------------------
# Notice field completeness
# ---------------------------------------------------------------------------


def test_notice_fields_are_complete_missing_stop_name() -> None:
    feed = make_feed(
        [
            {
                "csv_row_number": 4,
                "stop_id": "stop id value",
                "location_type": 0,
                "stop_name": None,
                "stop_desc": None,
            }
        ]
    )
    notices = validate_stop_name(feed, CTX)
    assert len(notices) == 1
    assert set(notices[0].fields.keys()) == {"csv_row_number", "stop_id", "location_type"}


def test_notice_fields_are_complete_same_name_and_description() -> None:
    feed = make_feed(
        [
            {
                "csv_row_number": 4,
                "stop_id": "stop id value",
                "location_type": 0,
                "stop_name": "dup",
                "stop_desc": "dup",
            }
        ]
    )
    notices = validate_stop_name(feed, CTX)
    assert len(notices) == 1
    assert set(notices[0].fields.keys()) == {"csv_row_number", "stop_id", "stop_desc"}
