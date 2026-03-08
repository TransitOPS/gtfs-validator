"""Tests for validate_stop_required_location."""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from gtfs_validator.context import ValidationContext
from gtfs_validator.notices import Severity
from gtfs_validator.validators.stop_required_location import validate_stop_required_location

CTX = ValidationContext(country_code="US", date_for_validation=date(2024, 1, 1))


def make_stops(rows: list[dict]) -> dict[str, pl.DataFrame]:
    """Build a minimal feed dict from a list of row dicts for stops.txt."""
    return {"stops": pl.DataFrame(rows)}


# ---------------------------------------------------------------------------
# Tests derived from Java contracts
# ---------------------------------------------------------------------------


def test_missing_lat_lon_for_stop_emits_notice():
    feed = make_stops(
        [{"csv_row_number": 4, "stop_id": "stop id value", "location_type": 0, "stop_lat": None, "stop_lon": None}]
    )
    notices = validate_stop_required_location(feed, CTX)
    assert len(notices) == 1
    n = notices[0]
    assert n.code == "stop_without_location"
    assert n.severity == Severity.ERROR
    assert n.fields["csv_row_number"] == 4
    assert n.fields["stop_id"] == "stop id value"
    assert n.fields["location_type"] == 0


def test_missing_lat_lon_for_station_emits_notice():
    feed = make_stops(
        [{"csv_row_number": 4, "stop_id": "stop id value", "location_type": 1, "stop_lat": None, "stop_lon": None}]
    )
    notices = validate_stop_required_location(feed, CTX)
    assert len(notices) == 1
    assert notices[0].fields["location_type"] == 1


def test_missing_lat_lon_for_entrance_emits_notice():
    feed = make_stops(
        [{"csv_row_number": 4, "stop_id": "stop id value", "location_type": 2, "stop_lat": None, "stop_lon": None}]
    )
    notices = validate_stop_required_location(feed, CTX)
    assert len(notices) == 1
    assert notices[0].fields["location_type"] == 2


def test_missing_lat_lon_for_generic_node_emits_no_notice():
    feed = make_stops(
        [{"csv_row_number": 4, "stop_id": "stop id value", "location_type": 3, "stop_lat": None, "stop_lon": None}]
    )
    notices = validate_stop_required_location(feed, CTX)
    assert len(notices) == 0


def test_missing_lat_lon_for_boarding_area_emits_no_notice():
    feed = make_stops(
        [{"csv_row_number": 4, "stop_id": "stop id value", "location_type": 4, "stop_lat": None, "stop_lon": None}]
    )
    notices = validate_stop_required_location(feed, CTX)
    assert len(notices) == 0


def test_both_coords_present_for_stop_emits_no_notice():
    feed = make_stops(
        [{"csv_row_number": 4, "stop_id": "S1", "location_type": 0, "stop_lat": 25.25, "stop_lon": 35.35}]
    )
    notices = validate_stop_required_location(feed, CTX)
    assert len(notices) == 0


def test_both_coords_present_for_station_emits_no_notice():
    feed = make_stops(
        [{"csv_row_number": 4, "stop_id": "S1", "location_type": 1, "stop_lat": 25.25, "stop_lon": 35.35}]
    )
    notices = validate_stop_required_location(feed, CTX)
    assert len(notices) == 0


def test_both_coords_present_for_entrance_emits_no_notice():
    feed = make_stops(
        [{"csv_row_number": 4, "stop_id": "S1", "location_type": 2, "stop_lat": 25.25, "stop_lon": 35.35}]
    )
    notices = validate_stop_required_location(feed, CTX)
    assert len(notices) == 0


def test_lat_present_lon_missing_for_stop_emits_notice():
    feed = make_stops(
        [{"csv_row_number": 4, "stop_id": "S1", "location_type": 0, "stop_lat": 25.25, "stop_lon": None}]
    )
    notices = validate_stop_required_location(feed, CTX)
    assert len(notices) == 1
    assert notices[0].code == "stop_without_location"


def test_lon_present_lat_missing_for_stop_emits_notice():
    feed = make_stops(
        [{"csv_row_number": 4, "stop_id": "S1", "location_type": 0, "stop_lat": None, "stop_lon": 25.25}]
    )
    notices = validate_stop_required_location(feed, CTX)
    assert len(notices) == 1
    assert notices[0].code == "stop_without_location"


def test_lat_present_lon_missing_for_station_emits_notice():
    feed = make_stops(
        [{"csv_row_number": 4, "stop_id": "S1", "location_type": 1, "stop_lat": 25.25, "stop_lon": None}]
    )
    notices = validate_stop_required_location(feed, CTX)
    assert len(notices) == 1
    assert notices[0].fields["location_type"] == 1


def test_lon_present_lat_missing_for_station_emits_notice():
    feed = make_stops(
        [{"csv_row_number": 4, "stop_id": "S1", "location_type": 1, "stop_lat": None, "stop_lon": 25.25}]
    )
    notices = validate_stop_required_location(feed, CTX)
    assert len(notices) == 1
    assert notices[0].fields["location_type"] == 1


def test_lat_present_lon_missing_for_entrance_emits_notice():
    feed = make_stops(
        [{"csv_row_number": 4, "stop_id": "S1", "location_type": 2, "stop_lat": 25.25, "stop_lon": None}]
    )
    notices = validate_stop_required_location(feed, CTX)
    assert len(notices) == 1
    assert notices[0].fields["location_type"] == 2


def test_lon_present_lat_missing_for_entrance_emits_notice():
    feed = make_stops(
        [{"csv_row_number": 4, "stop_id": "S1", "location_type": 2, "stop_lat": None, "stop_lon": 25.25}]
    )
    notices = validate_stop_required_location(feed, CTX)
    assert len(notices) == 1
    assert notices[0].fields["location_type"] == 2


# ---------------------------------------------------------------------------
# Additional Python edge-case tests
# ---------------------------------------------------------------------------


def test_stops_table_absent_emits_no_notice():
    notices = validate_stop_required_location({}, CTX)
    assert len(notices) == 0


def test_stops_table_empty_emits_no_notice():
    feed = {
        "stops": pl.DataFrame(
            {"stop_id": [], "location_type": [], "stop_lat": [], "stop_lon": [], "csv_row_number": []},
            schema={
                "stop_id": pl.Utf8,
                "location_type": pl.Int64,
                "stop_lat": pl.Float64,
                "stop_lon": pl.Float64,
                "csv_row_number": pl.Int64,
            },
        )
    }
    notices = validate_stop_required_location(feed, CTX)
    assert len(notices) == 0


def test_location_type_absent_from_header_treats_rows_as_stop():
    # No location_type column — all rows default to STOP (0), requiring coordinates.
    feed = {
        "stops": pl.DataFrame(
            [{"stop_id": "S1", "stop_lat": None, "stop_lon": None, "csv_row_number": 2}]
        )
    }
    notices = validate_stop_required_location(feed, CTX)
    assert len(notices) == 1
    assert notices[0].fields["location_type"] == 0


def test_stop_lat_column_absent_from_header_emits_notice():
    # No stop_lat column — treated as all-null lat.
    feed = {
        "stops": pl.DataFrame(
            [{"stop_id": "S1", "location_type": 0, "stop_lon": 35.35, "csv_row_number": 3}]
        )
    }
    notices = validate_stop_required_location(feed, CTX)
    assert len(notices) == 1


def test_stop_lon_column_absent_from_header_emits_notice():
    # No stop_lon column — treated as all-null lon.
    feed = {
        "stops": pl.DataFrame(
            [{"stop_id": "S1", "location_type": 0, "stop_lat": 25.25, "csv_row_number": 3}]
        )
    }
    notices = validate_stop_required_location(feed, CTX)
    assert len(notices) == 1


def test_both_coord_columns_absent_from_header_emits_notice():
    # Neither stop_lat nor stop_lon — both treated as all-null.
    feed = {
        "stops": pl.DataFrame(
            [{"stop_id": "S1", "location_type": 0, "csv_row_number": 3}]
        )
    }
    notices = validate_stop_required_location(feed, CTX)
    assert len(notices) == 1


def test_multiple_offending_rows_emit_one_notice_each():
    feed = make_stops(
        [
            {"csv_row_number": 2, "stop_id": "A", "location_type": 0, "stop_lat": None, "stop_lon": None},
            {"csv_row_number": 3, "stop_id": "B", "location_type": 3, "stop_lat": None, "stop_lon": None},
            {"csv_row_number": 4, "stop_id": "C", "location_type": 0, "stop_lat": None, "stop_lon": None},
        ]
    )
    notices = validate_stop_required_location(feed, CTX)
    assert len(notices) == 2
    row_numbers = {n.fields["csv_row_number"] for n in notices}
    assert row_numbers == {2, 4}


def test_unrecognized_location_type_emits_no_notice():
    feed = make_stops(
        [{"csv_row_number": 5, "stop_id": "S1", "location_type": 99, "stop_lat": None, "stop_lon": None}]
    )
    notices = validate_stop_required_location(feed, CTX)
    assert len(notices) == 0


def test_notice_fields_are_complete():
    feed = make_stops(
        [{"csv_row_number": 7, "stop_id": "S1", "location_type": 0, "stop_lat": None, "stop_lon": None}]
    )
    notices = validate_stop_required_location(feed, CTX)
    assert len(notices) == 1
    assert set(notices[0].fields.keys()) == {"csv_row_number", "stop_id", "location_type"}
