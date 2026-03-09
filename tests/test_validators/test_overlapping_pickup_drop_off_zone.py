"""Tests for validate_overlapping_pickup_drop_off_zone."""

from datetime import date

import polars as pl
import pytest

from gtfs_validator.context import ValidationContext
from gtfs_validator.notices import Severity
from gtfs_validator.validators.overlapping_pickup_drop_off_zone import (
    validate_overlapping_pickup_drop_off_zone,
)

CTX = ValidationContext(country_code="US", date_for_validation=date(2026, 3, 8))

# Two overlapping squares: Feature "1" covers (0,0)-(4,4), Feature "2" covers (2,2)-(6,6).
# Their interiors partially overlap, so geom1.overlaps(geom2) is True.
_GEOJSON_OVERLAPPING = [
    {
        "id": "1",
        "geometry_type": "Polygon",
        "coordinates": [[[0, 0], [4, 0], [4, 4], [0, 4], [0, 0]]],
    },
    {
        "id": "2",
        "geometry_type": "Polygon",
        "coordinates": [[[2, 2], [6, 2], [6, 6], [2, 6], [2, 2]]],
    },
]

# Two non-overlapping squares: Feature "1" covers (0,0)-(2,2), Feature "2" covers (5,5)-(7,7).
_GEOJSON_NON_OVERLAPPING = [
    {
        "id": "1",
        "geometry_type": "Polygon",
        "coordinates": [[[0, 0], [2, 0], [2, 2], [0, 2], [0, 0]]],
    },
    {
        "id": "2",
        "geometry_type": "Polygon",
        "coordinates": [[[5, 5], [7, 5], [7, 7], [5, 7], [5, 5]]],
    },
]


def make_stop_times(rows: list[dict]) -> pl.DataFrame:
    """Build a minimal stop_times DataFrame from a list of row dicts."""
    return pl.DataFrame(
        {
            "trip_id": [r["trip_id"] for r in rows],
            "stop_sequence": [r["stop_sequence"] for r in rows],
            "location_id": [r.get("location_id") for r in rows],
            "pickup_type": [r.get("pickup_type") for r in rows],
            "drop_off_type": [r.get("drop_off_type", 0) for r in rows],
            "start_pickup_drop_off_window": [r.get("start_pickup_drop_off_window") for r in rows],
            "end_pickup_drop_off_window": [r.get("end_pickup_drop_off_window") for r in rows],
        },
        schema={
            "trip_id": pl.Utf8,
            "stop_sequence": pl.Int64,
            "location_id": pl.Utf8,
            "pickup_type": pl.Int64,
            "drop_off_type": pl.Int64,
            "start_pickup_drop_off_window": pl.Utf8,
            "end_pickup_drop_off_window": pl.Utf8,
        },
    )


def test_overlapping_zones_and_windows_generates_notice() -> None:
    feed = {
        "stop_times": make_stop_times([
            {"trip_id": "t0", "stop_sequence": 1, "location_id": "1",
             "pickup_type": 1, "drop_off_type": 0,
             "start_pickup_drop_off_window": "05:00:00",
             "end_pickup_drop_off_window": "07:00:00"},
            {"trip_id": "t0", "stop_sequence": 2, "location_id": "2",
             "pickup_type": 1, "drop_off_type": 0,
             "start_pickup_drop_off_window": "06:00:00",
             "end_pickup_drop_off_window": "08:00:00"},
        ]),
        "locations_geojson": _GEOJSON_OVERLAPPING,
    }
    notices = validate_overlapping_pickup_drop_off_zone(feed, CTX)
    assert len(notices) == 1
    n = notices[0]
    assert n.code == "overlapping_zone_and_pickup_drop_off_window"
    assert n.severity == Severity.ERROR
    assert n.fields["trip_id"] == "t0"
    assert n.fields["stop_sequence1"] == 1
    assert n.fields["location_id1"] == "1"
    assert n.fields["start_pickup_drop_off_window1"] == "05:00:00"
    assert n.fields["end_pickup_drop_off_window1"] == "07:00:00"
    assert n.fields["stop_sequence2"] == 2
    assert n.fields["location_id2"] == "2"
    assert n.fields["start_pickup_drop_off_window2"] == "06:00:00"
    assert n.fields["end_pickup_drop_off_window2"] == "08:00:00"


def test_non_overlapping_windows_no_notice() -> None:
    feed = {
        "stop_times": make_stop_times([
            {"trip_id": "t0", "stop_sequence": 1, "location_id": "1",
             "pickup_type": 0, "drop_off_type": 0,
             "start_pickup_drop_off_window": "05:00:00",
             "end_pickup_drop_off_window": "07:00:00"},
            {"trip_id": "t0", "stop_sequence": 2, "location_id": "2",
             "pickup_type": 0, "drop_off_type": 0,
             "start_pickup_drop_off_window": "07:00:00",
             "end_pickup_drop_off_window": "08:00:00"},
        ]),
        "locations_geojson": _GEOJSON_OVERLAPPING,
    }
    assert validate_overlapping_pickup_drop_off_zone(feed, CTX) == []


def test_non_overlapping_zones_no_notice() -> None:
    feed = {
        "stop_times": make_stop_times([
            {"trip_id": "t0", "stop_sequence": 1, "location_id": "1",
             "pickup_type": 0, "drop_off_type": 0,
             "start_pickup_drop_off_window": "05:00:00",
             "end_pickup_drop_off_window": "07:00:00"},
            {"trip_id": "t0", "stop_sequence": 2, "location_id": "2",
             "pickup_type": 0, "drop_off_type": 0,
             "start_pickup_drop_off_window": "06:00:00",
             "end_pickup_drop_off_window": "08:00:00"},
        ]),
        "locations_geojson": _GEOJSON_NON_OVERLAPPING,
    }
    assert validate_overlapping_pickup_drop_off_zone(feed, CTX) == []


def test_stop_times_absent() -> None:
    assert validate_overlapping_pickup_drop_off_zone(
        {"locations_geojson": _GEOJSON_OVERLAPPING}, CTX
    ) == []


def test_locations_geojson_absent() -> None:
    feed = {
        "stop_times": make_stop_times([
            {"trip_id": "t0", "stop_sequence": 1, "location_id": "1",
             "pickup_type": 0, "drop_off_type": 0,
             "start_pickup_drop_off_window": "05:00:00",
             "end_pickup_drop_off_window": "07:00:00"},
        ]),
    }
    assert validate_overlapping_pickup_drop_off_zone(feed, CTX) == []


def test_locations_geojson_empty() -> None:
    feed = {
        "stop_times": make_stop_times([
            {"trip_id": "t0", "stop_sequence": 1, "location_id": "1",
             "pickup_type": 0, "drop_off_type": 0,
             "start_pickup_drop_off_window": "05:00:00",
             "end_pickup_drop_off_window": "07:00:00"},
            {"trip_id": "t0", "stop_sequence": 2, "location_id": "2",
             "pickup_type": 0, "drop_off_type": 0,
             "start_pickup_drop_off_window": "06:00:00",
             "end_pickup_drop_off_window": "08:00:00"},
        ]),
        "locations_geojson": [],
    }
    assert validate_overlapping_pickup_drop_off_zone(feed, CTX) == []


def test_single_stop_time_per_trip() -> None:
    feed = {
        "stop_times": make_stop_times([
            {"trip_id": "t0", "stop_sequence": 1, "location_id": "1",
             "pickup_type": 0, "drop_off_type": 0,
             "start_pickup_drop_off_window": "05:00:00",
             "end_pickup_drop_off_window": "07:00:00"},
        ]),
        "locations_geojson": _GEOJSON_OVERLAPPING,
    }
    assert validate_overlapping_pickup_drop_off_zone(feed, CTX) == []


def test_same_location_id_skipped() -> None:
    feed = {
        "stop_times": make_stop_times([
            {"trip_id": "t0", "stop_sequence": 1, "location_id": "1",
             "pickup_type": 0, "drop_off_type": 0,
             "start_pickup_drop_off_window": "05:00:00",
             "end_pickup_drop_off_window": "07:00:00"},
            {"trip_id": "t0", "stop_sequence": 2, "location_id": "1",
             "pickup_type": 0, "drop_off_type": 0,
             "start_pickup_drop_off_window": "06:00:00",
             "end_pickup_drop_off_window": "08:00:00"},
        ]),
        "locations_geojson": _GEOJSON_OVERLAPPING,
    }
    assert validate_overlapping_pickup_drop_off_zone(feed, CTX) == []


def test_both_pickup_and_dropoff_types_differ_skipped() -> None:
    feed = {
        "stop_times": make_stop_times([
            {"trip_id": "t0", "stop_sequence": 1, "location_id": "1",
             "pickup_type": 0, "drop_off_type": 1,
             "start_pickup_drop_off_window": "05:00:00",
             "end_pickup_drop_off_window": "07:00:00"},
            {"trip_id": "t0", "stop_sequence": 2, "location_id": "2",
             "pickup_type": 1, "drop_off_type": 0,
             "start_pickup_drop_off_window": "06:00:00",
             "end_pickup_drop_off_window": "08:00:00"},
        ]),
        "locations_geojson": _GEOJSON_OVERLAPPING,
    }
    assert validate_overlapping_pickup_drop_off_zone(feed, CTX) == []


def test_unrecognized_pickup_type_skipped() -> None:
    feed = {
        "stop_times": make_stop_times([
            {"trip_id": "t0", "stop_sequence": 1, "location_id": "1",
             "pickup_type": 5, "drop_off_type": 0,
             "start_pickup_drop_off_window": "05:00:00",
             "end_pickup_drop_off_window": "07:00:00"},
            {"trip_id": "t0", "stop_sequence": 2, "location_id": "2",
             "pickup_type": 5, "drop_off_type": 0,
             "start_pickup_drop_off_window": "06:00:00",
             "end_pickup_drop_off_window": "08:00:00"},
        ]),
        "locations_geojson": _GEOJSON_OVERLAPPING,
    }
    assert validate_overlapping_pickup_drop_off_zone(feed, CTX) == []


def test_null_window_fields_skipped() -> None:
    df = pl.DataFrame(
        {
            "trip_id": ["t0", "t0"],
            "stop_sequence": [1, 2],
            "location_id": ["1", "2"],
            "pickup_type": [0, 0],
            "drop_off_type": [0, 0],
            "start_pickup_drop_off_window": [None, "06:00:00"],
            "end_pickup_drop_off_window": ["07:00:00", "08:00:00"],
        },
        schema={
            "trip_id": pl.Utf8,
            "stop_sequence": pl.Int64,
            "location_id": pl.Utf8,
            "pickup_type": pl.Int64,
            "drop_off_type": pl.Int64,
            "start_pickup_drop_off_window": pl.Utf8,
            "end_pickup_drop_off_window": pl.Utf8,
        },
    )
    feed = {"stop_times": df, "locations_geojson": _GEOJSON_OVERLAPPING}
    assert validate_overlapping_pickup_drop_off_zone(feed, CTX) == []


def test_null_location_id_skipped() -> None:
    df = pl.DataFrame(
        {
            "trip_id": ["t0", "t0"],
            "stop_sequence": [1, 2],
            "location_id": [None, "2"],
            "pickup_type": [0, 0],
            "drop_off_type": [0, 0],
            "start_pickup_drop_off_window": ["05:00:00", "06:00:00"],
            "end_pickup_drop_off_window": ["07:00:00", "08:00:00"],
        },
        schema={
            "trip_id": pl.Utf8,
            "stop_sequence": pl.Int64,
            "location_id": pl.Utf8,
            "pickup_type": pl.Int64,
            "drop_off_type": pl.Int64,
            "start_pickup_drop_off_window": pl.Utf8,
            "end_pickup_drop_off_window": pl.Utf8,
        },
    )
    feed = {"stop_times": df, "locations_geojson": _GEOJSON_OVERLAPPING}
    assert validate_overlapping_pickup_drop_off_zone(feed, CTX) == []


def test_location_id_not_in_geojson_skipped() -> None:
    feed = {
        "stop_times": make_stop_times([
            {"trip_id": "t0", "stop_sequence": 1, "location_id": "99",
             "pickup_type": 0, "drop_off_type": 0,
             "start_pickup_drop_off_window": "05:00:00",
             "end_pickup_drop_off_window": "07:00:00"},
            {"trip_id": "t0", "stop_sequence": 2, "location_id": "2",
             "pickup_type": 0, "drop_off_type": 0,
             "start_pickup_drop_off_window": "06:00:00",
             "end_pickup_drop_off_window": "08:00:00"},
        ]),
        "locations_geojson": _GEOJSON_OVERLAPPING,
    }
    assert validate_overlapping_pickup_drop_off_zone(feed, CTX) == []


def test_multiple_qualifying_pairs_emit_multiple_notices() -> None:
    # Three mutually overlapping squares arranged in a triangle overlap pattern
    geojson = [
        {"id": "1", "geometry_type": "Polygon",
         "coordinates": [[[0, 0], [4, 0], [4, 4], [0, 4], [0, 0]]]},
        {"id": "2", "geometry_type": "Polygon",
         "coordinates": [[[2, 0], [6, 0], [6, 4], [2, 4], [2, 0]]]},
        {"id": "3", "geometry_type": "Polygon",
         "coordinates": [[[1, 0], [5, 0], [5, 4], [1, 4], [1, 0]]]},
    ]
    feed = {
        "stop_times": make_stop_times([
            {"trip_id": "t0", "stop_sequence": 1, "location_id": "1",
             "pickup_type": 0, "drop_off_type": 0,
             "start_pickup_drop_off_window": "05:00:00",
             "end_pickup_drop_off_window": "08:00:00"},
            {"trip_id": "t0", "stop_sequence": 2, "location_id": "2",
             "pickup_type": 0, "drop_off_type": 0,
             "start_pickup_drop_off_window": "06:00:00",
             "end_pickup_drop_off_window": "09:00:00"},
            {"trip_id": "t0", "stop_sequence": 3, "location_id": "3",
             "pickup_type": 0, "drop_off_type": 0,
             "start_pickup_drop_off_window": "07:00:00",
             "end_pickup_drop_off_window": "10:00:00"},
        ]),
        "locations_geojson": geojson,
    }
    notices = validate_overlapping_pickup_drop_off_zone(feed, CTX)
    assert len(notices) == 3
    for n in notices:
        assert n.code == "overlapping_zone_and_pickup_drop_off_window"
        assert n.severity == Severity.ERROR


def test_different_trips_are_independent() -> None:
    feed = {
        "stop_times": make_stop_times([
            # trip t0: non-overlapping windows
            {"trip_id": "t0", "stop_sequence": 1, "location_id": "1",
             "pickup_type": 0, "drop_off_type": 0,
             "start_pickup_drop_off_window": "05:00:00",
             "end_pickup_drop_off_window": "07:00:00"},
            {"trip_id": "t0", "stop_sequence": 2, "location_id": "2",
             "pickup_type": 0, "drop_off_type": 0,
             "start_pickup_drop_off_window": "07:00:00",
             "end_pickup_drop_off_window": "09:00:00"},
            # trip t1: overlapping windows and overlapping zones
            {"trip_id": "t1", "stop_sequence": 1, "location_id": "1",
             "pickup_type": 0, "drop_off_type": 0,
             "start_pickup_drop_off_window": "05:00:00",
             "end_pickup_drop_off_window": "07:00:00"},
            {"trip_id": "t1", "stop_sequence": 2, "location_id": "2",
             "pickup_type": 0, "drop_off_type": 0,
             "start_pickup_drop_off_window": "06:00:00",
             "end_pickup_drop_off_window": "08:00:00"},
        ]),
        "locations_geojson": _GEOJSON_OVERLAPPING,
    }
    notices = validate_overlapping_pickup_drop_off_zone(feed, CTX)
    assert len(notices) == 1
    assert notices[0].fields["trip_id"] == "t1"


def test_missing_flex_columns_returns_no_notices() -> None:
    df = pl.DataFrame(
        {"trip_id": ["t0"], "stop_sequence": [1], "stop_id": ["s1"]},
        schema={"trip_id": pl.Utf8, "stop_sequence": pl.Int64, "stop_id": pl.Utf8},
    )
    feed = {"stop_times": df, "locations_geojson": _GEOJSON_OVERLAPPING}
    assert validate_overlapping_pickup_drop_off_zone(feed, CTX) == []


def test_same_pickup_type_different_dropoff_type_not_skipped() -> None:
    feed = {
        "stop_times": make_stop_times([
            {"trip_id": "t0", "stop_sequence": 1, "location_id": "1",
             "pickup_type": 0, "drop_off_type": 0,
             "start_pickup_drop_off_window": "05:00:00",
             "end_pickup_drop_off_window": "07:00:00"},
            {"trip_id": "t0", "stop_sequence": 2, "location_id": "2",
             "pickup_type": 0, "drop_off_type": 1,
             "start_pickup_drop_off_window": "06:00:00",
             "end_pickup_drop_off_window": "08:00:00"},
        ]),
        "locations_geojson": _GEOJSON_OVERLAPPING,
    }
    notices = validate_overlapping_pickup_drop_off_zone(feed, CTX)
    assert len(notices) == 1
    assert notices[0].code == "overlapping_zone_and_pickup_drop_off_window"
