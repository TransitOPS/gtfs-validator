"""Tests for StopAccessValidator."""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from gtfs_validator.context import ValidationContext
from gtfs_validator.notices import Severity
from gtfs_validator.validators.stop_access import validate_stop_access

CTX = ValidationContext(country_code="US", date_for_validation=date(2024, 1, 1))


def make_stops(**columns) -> pl.DataFrame:
    """Build a minimal stops DataFrame with only the supplied columns.

    Caller is responsible for providing all columns needed by the test.
    csv_row_number, stop_id, stop_name, stop_access, location_type, parent_station
    are the columns the validator reads.
    """
    return pl.DataFrame(columns)


# --- Test 1 ---
def test_stop_location_without_parent_station_generates_notice():
    """A STOP (location_type=0) with stop_access set and no parent_station emits an error."""
    stops = make_stops(
        csv_row_number=[7],
        stop_id=["S1"],
        stop_name=["Stop 1"],
        stop_access=[1],
        location_type=[0],
        parent_station=[None],
    )
    notices = validate_stop_access({"stops": stops}, CTX)
    assert len(notices) == 1
    n = notices[0]
    assert n.code == "stop_access_specified_for_stop_with_no_parent_station"
    assert n.severity == Severity.ERROR
    assert n.fields["csv_row_number"] == 7
    assert n.fields["stop_id"] == "S1"
    assert n.fields["stop_name"] == "Stop 1"
    assert n.fields["stop_access"] == 1
    assert n.fields["location_type"] == 0


# --- Test 2 ---
def test_non_stop_location_with_stop_access_generates_incorrect_location_notice():
    """A non-STOP (location_type=1) with stop_access set emits incorrect_location error."""
    stops = make_stops(
        csv_row_number=[9],
        stop_id=["S2"],
        stop_name=["Stop 2"],
        stop_access=[1],
        location_type=[1],
    )
    notices = validate_stop_access({"stops": stops}, CTX)
    assert len(notices) == 1
    n = notices[0]
    assert n.code == "stop_access_specified_for_incorrect_location"
    assert n.severity == Severity.ERROR
    assert n.fields["csv_row_number"] == 9
    assert n.fields["stop_id"] == "S2"
    assert n.fields["stop_name"] == "Stop 2"
    assert n.fields["stop_access"] == 1
    assert n.fields["location_type"] == 1


# --- Test 3 ---
def test_stop_with_parent_station_and_stop_access_no_notice():
    """Valid: STOP (location_type=0) with stop_access set AND parent_station present — no notice."""
    stops = make_stops(
        csv_row_number=[1],
        stop_id=["S3"],
        stop_name=["Stop 3"],
        stop_access=[0],
        location_type=[0],
        parent_station=["P1"],
    )
    notices = validate_stop_access({"stops": stops}, CTX)
    assert notices == []


# --- Test 4 ---
def test_stop_access_null_no_notice():
    """Per-row null guard: stop_access is null — no validation runs on this row."""
    stops = make_stops(
        csv_row_number=[1],
        stop_id=["S4"],
        stop_name=["Stop 4"],
        stop_access=[None],
        location_type=[0],
    )
    notices = validate_stop_access({"stops": stops}, CTX)
    assert notices == []


# --- Test 5 ---
def test_stops_absent_no_notice():
    """Feed has no stops table — no notices."""
    notices = validate_stop_access({}, CTX)
    assert notices == []


# --- Test 6 ---
def test_stops_empty_no_notice():
    """stops.txt is present but contains no data rows — no notices."""
    stops = pl.DataFrame(
        {"stop_id": [], "stop_access": [], "location_type": []},
        schema={"stop_id": pl.Utf8, "stop_access": pl.Int64, "location_type": pl.Int64},
    )
    notices = validate_stop_access({"stops": stops}, CTX)
    assert notices == []


# --- Test 7 ---
def test_stop_access_column_absent_no_notice():
    """Column-level guard: stop_access column absent — no notices."""
    stops = make_stops(
        stop_id=["S7"],
        stop_name=["Stop 7"],
        location_type=[1],
    )
    notices = validate_stop_access({"stops": stops}, CTX)
    assert notices == []


# --- Test 8 ---
def test_location_type_defaults_to_stop_triggers_no_parent_notice():
    """When location_type is 0 (defaulted STOP) and parent_station column is absent, emits no_parent notice."""
    stops = make_stops(
        csv_row_number=[2],
        stop_id=["S8"],
        stop_access=[1],
        location_type=[0],
    )
    notices = validate_stop_access({"stops": stops}, CTX)
    assert len(notices) == 1
    assert notices[0].code == "stop_access_specified_for_stop_with_no_parent_station"


# --- Test 9 ---
def test_station_with_parent_station_still_emits_incorrect_location_notice():
    """parent_station is irrelevant for non-STOP location types — still emits incorrect_location."""
    stops = make_stops(
        csv_row_number=[3],
        stop_id=["S9"],
        stop_name=["Stop 9"],
        stop_access=[0],
        location_type=[1],
        parent_station=["P1"],
    )
    notices = validate_stop_access({"stops": stops}, CTX)
    assert len(notices) == 1
    assert notices[0].code == "stop_access_specified_for_incorrect_location"


# --- Test 10 ---
@pytest.mark.parametrize("lt", [1, 2, 3, 4])
def test_all_non_stop_location_types_emit_incorrect_location_notice(lt: int):
    """All non-STOP location types (1-4) with stop_access set emit incorrect_location notice."""
    stops = make_stops(
        csv_row_number=[lt],
        stop_id=[f"S{lt}"],
        stop_name=[f"Stop {lt}"],
        stop_access=[1],
        location_type=[lt],
    )
    notices = validate_stop_access({"stops": stops}, CTX)
    assert len(notices) == 1
    n = notices[0]
    assert n.code == "stop_access_specified_for_incorrect_location"
    assert n.fields["location_type"] == lt


# --- Test 11 ---
def test_multiple_offending_rows_emit_multiple_notices():
    """Three rows all with stop_access set: 1 no_parent_station + 2 incorrect_location."""
    stops = make_stops(
        csv_row_number=[1, 2, 3],
        stop_id=["A", "B", "C"],
        stop_name=["Stop A", "Stop B", "Stop C"],
        stop_access=[1, 1, 1],
        location_type=[0, 1, 3],
        parent_station=[None, None, None],
    )
    notices = validate_stop_access({"stops": stops}, CTX)
    assert len(notices) == 3
    codes = [n.code for n in notices]
    assert codes.count("stop_access_specified_for_stop_with_no_parent_station") == 1
    assert codes.count("stop_access_specified_for_incorrect_location") == 2


# --- Test 12 ---
def test_stop_name_null_passed_through():
    """A null stop_name is passed through to the notice fields as None."""
    stops = make_stops(
        csv_row_number=[5],
        stop_id=["S12"],
        stop_name=[None],
        stop_access=[1],
        location_type=[0],
    )
    notices = validate_stop_access({"stops": stops}, CTX)
    assert len(notices) == 1
    assert notices[0].fields["stop_name"] is None


# --- Test 13 ---
def test_notice_fields_are_complete():
    """Both notice types contain exactly the expected set of field keys."""
    expected_keys = {"csv_row_number", "stop_id", "stop_name", "stop_access", "location_type"}

    # no_parent_station notice
    stops_no_parent = make_stops(
        csv_row_number=[1],
        stop_id=["S1"],
        stop_name=["Stop 1"],
        stop_access=[1],
        location_type=[0],
        parent_station=[None],
    )
    notices_a = validate_stop_access({"stops": stops_no_parent}, CTX)
    assert len(notices_a) == 1
    assert set(notices_a[0].fields.keys()) == expected_keys

    # incorrect_location notice
    stops_wrong_type = make_stops(
        csv_row_number=[2],
        stop_id=["S2"],
        stop_name=["Stop 2"],
        stop_access=[1],
        location_type=[1],
    )
    notices_b = validate_stop_access({"stops": stops_wrong_type}, CTX)
    assert len(notices_b) == 1
    assert set(notices_b[0].fields.keys()) == expected_keys
