"""Tests for LocationTypeSingleEntityValidator."""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from gtfs_validator.context import ValidationContext
from gtfs_validator.notices import Severity
from gtfs_validator.validators.location_type_single_entity import (
    validate_location_type_single_entity,
)

CTX = ValidationContext(country_code="US", date_for_validation=date(2026, 3, 8))


def make_stops(rows: list[dict]) -> pl.DataFrame:
    """Build a stops DataFrame from row dicts."""
    schema = {
        "csvRowNumber": pl.Int64,
        "stop_id": pl.Utf8,
        "stop_name": pl.Utf8,
        "location_type": pl.Int64,
        "parent_station": pl.Utf8,
        "platform_code": pl.Utf8,
    }
    data: dict[str, list] = {col: [] for col in schema}
    for r in rows:
        for col in schema:
            data[col].append(r.get(col))
    return pl.DataFrame(data, schema=schema)


# --- Test 1 ---
def test_station_without_parent_no_notice():
    feed = {"stops": make_stops([
        {"stop_id": "s0", "location_type": 1, "stop_name": "Stop 0",
         "parent_station": None, "csvRowNumber": 1},
    ])}
    notices = validate_location_type_single_entity(feed, CTX)
    assert len(notices) == 0


# --- Test 2 ---
def test_station_with_parent_yields_error():
    feed = {"stops": make_stops([
        {"stop_id": "s0", "location_type": 1, "stop_name": "Stop 0",
         "parent_station": "parent", "csvRowNumber": 1},
    ])}
    notices = validate_location_type_single_entity(feed, CTX)
    assert len(notices) == 1
    n = notices[0]
    assert n.code == "station_with_parent_station"
    assert n.severity == Severity.ERROR
    assert n.fields["csvRowNumber"] == 1
    assert n.fields["stopId"] == "s0"
    assert n.fields["stopName"] == "Stop 0"
    assert n.fields["parentStation"] == "parent"


# --- Test 3 ---
def test_platform_with_parent_no_notice():
    feed = {"stops": make_stops([
        {"stop_id": "s0", "location_type": 0, "stop_name": "Stop 0",
         "parent_station": "parent", "platform_code": "1", "csvRowNumber": 1},
    ])}
    notices = validate_location_type_single_entity(feed, CTX)
    assert len(notices) == 0


# --- Test 4 ---
def test_platform_without_parent_yields_info():
    feed = {"stops": make_stops([
        {"stop_id": "s0", "location_type": 0, "stop_name": "Stop 0",
         "parent_station": None, "platform_code": "1", "csvRowNumber": 1},
    ])}
    notices = validate_location_type_single_entity(feed, CTX)
    assert len(notices) == 1
    n = notices[0]
    assert n.code == "platform_without_parent_station"
    assert n.severity == Severity.INFO
    assert n.fields["csvRowNumber"] == 1
    assert n.fields["stopId"] == "s0"
    assert n.fields["stopName"] == "Stop 0"


# --- Test 5 ---
def test_stop_without_platform_code_and_without_parent_no_notice():
    feed = {"stops": make_stops([
        {"stop_id": "s0", "location_type": 0, "stop_name": "Stop 0",
         "parent_station": None, "platform_code": None, "csvRowNumber": 1},
    ])}
    notices = validate_location_type_single_entity(feed, CTX)
    assert len(notices) == 0


# --- Test 6 ---
@pytest.mark.parametrize("lt", [2, 3, 4])
def test_location_without_parent_yields_error(lt: int):
    feed = {"stops": make_stops([
        {"stop_id": "s0", "location_type": lt, "stop_name": "Stop 0",
         "parent_station": None, "csvRowNumber": 1},
    ])}
    notices = validate_location_type_single_entity(feed, CTX)
    assert len(notices) == 1
    n = notices[0]
    assert n.code == "location_without_parent_station"
    assert n.severity == Severity.ERROR
    assert n.fields["csvRowNumber"] == 1
    assert n.fields["stopId"] == "s0"
    assert n.fields["stopName"] == "Stop 0"
    assert n.fields["locationType"] == lt


# --- Test 7 ---
@pytest.mark.parametrize("lt", [2, 3, 4])
def test_location_with_parent_no_notice(lt: int):
    feed = {"stops": make_stops([
        {"stop_id": "s0", "location_type": lt, "stop_name": "Stop 0",
         "parent_station": "parent", "csvRowNumber": 1},
    ])}
    notices = validate_location_type_single_entity(feed, CTX)
    assert len(notices) == 0


# --- Test 8 ---
def test_stops_missing_no_notices():
    notices = validate_location_type_single_entity({}, CTX)
    assert len(notices) == 0


# --- Test 9 ---
def test_stops_empty_no_notices():
    feed = {"stops": make_stops([])}
    notices = validate_location_type_single_entity(feed, CTX)
    assert len(notices) == 0


# --- Test 10 ---
def test_multiple_violations_across_types():
    feed = {"stops": make_stops([
        # station with parent -> station_with_parent_station ERROR
        {"stop_id": "s1", "location_type": 1, "stop_name": "Station",
         "parent_station": "p1", "csvRowNumber": 1},
        # entrance without parent -> location_without_parent_station ERROR
        {"stop_id": "s2", "location_type": 2, "stop_name": "Entrance",
         "parent_station": None, "csvRowNumber": 2},
        # stop with platform_code but no parent -> platform_without_parent_station INFO
        {"stop_id": "s3", "location_type": 0, "stop_name": "Platform",
         "parent_station": None, "platform_code": "A", "csvRowNumber": 3},
    ])}
    notices = validate_location_type_single_entity(feed, CTX)
    assert len(notices) == 3
    codes = {n.code for n in notices}
    assert codes == {
        "station_with_parent_station",
        "location_without_parent_station",
        "platform_without_parent_station",
    }
    severities = {n.code: n.severity for n in notices}
    assert severities["station_with_parent_station"] == Severity.ERROR
    assert severities["location_without_parent_station"] == Severity.ERROR
    assert severities["platform_without_parent_station"] == Severity.INFO


# --- Test 11 ---
def test_parent_station_empty_string_treated_as_absent():
    feed = {"stops": make_stops([
        {"stop_id": "s0", "location_type": 2, "stop_name": "Stop 0",
         "parent_station": "", "csvRowNumber": 1},
    ])}
    notices = validate_location_type_single_entity(feed, CTX)
    assert len(notices) == 1
    assert notices[0].code == "location_without_parent_station"


# --- Test 12 ---
def test_location_type_defaults_to_zero():
    feed = {"stops": make_stops([
        {"stop_id": "s0", "location_type": 0, "stop_name": "Stop 0",
         "parent_station": None, "platform_code": None, "csvRowNumber": 1},
    ])}
    notices = validate_location_type_single_entity(feed, CTX)
    assert len(notices) == 0
