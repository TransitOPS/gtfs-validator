"""Tests for ShapeToStopMatchingValidator."""

from __future__ import annotations

import datetime

import polars as pl
import pytest

from gtfs_validator.context import ValidationContext
from gtfs_validator.notices import Severity
from gtfs_validator.validators.shape_to_stop_matching import validate_shape_to_stop_matching
from gtfs_validator.validators.shape_to_stop_matching_util import MatchSettings

CTX = ValidationContext(country_code="CH", date_for_validation=datetime.date(2024, 1, 1))

# ---------------------------------------------------------------------------
# Canonical test shape (Zurich area, as specified in spec)
# ---------------------------------------------------------------------------
# seq | lat        | lon
#   0 | 47.365873  | 8.525414
#   1 | 47.365872  | 8.525376
#   2 | 47.366099  | 8.525154
#   3 | 47.365925  | 8.525735
#   4 | 47.364108  | 8.525715

_SHAPE_ROWS = [
    {"csv_row_number": 100, "shape_id": "S1", "shape_pt_sequence": 0,
     "shape_pt_lat": 47.365873, "shape_pt_lon": 8.525414, "shape_dist_traveled": None},
    {"csv_row_number": 101, "shape_id": "S1", "shape_pt_sequence": 1,
     "shape_pt_lat": 47.365872, "shape_pt_lon": 8.525376, "shape_dist_traveled": None},
    {"csv_row_number": 102, "shape_id": "S1", "shape_pt_sequence": 2,
     "shape_pt_lat": 47.366099, "shape_pt_lon": 8.525154, "shape_dist_traveled": None},
    {"csv_row_number": 103, "shape_id": "S1", "shape_pt_sequence": 3,
     "shape_pt_lat": 47.365925, "shape_pt_lon": 8.525735, "shape_dist_traveled": None},
    {"csv_row_number": 104, "shape_id": "S1", "shape_pt_sequence": 4,
     "shape_pt_lat": 47.364108, "shape_pt_lon": 8.525715, "shape_dist_traveled": None},
]

# Looping shape for test 6.3 (too_many_matches):
# Shape passes near (47.365, 8.525) three separate times with large detours in between.
_LOOPING_SHAPE_ROWS = [
    {"csv_row_number": 200, "shape_id": "SL", "shape_pt_sequence": 0,
     "shape_pt_lat": 47.365000, "shape_pt_lon": 8.5240, "shape_dist_traveled": None},
    {"csv_row_number": 201, "shape_id": "SL", "shape_pt_sequence": 1,
     "shape_pt_lat": 47.365000, "shape_pt_lon": 8.5260, "shape_dist_traveled": None},
    {"csv_row_number": 202, "shape_id": "SL", "shape_pt_sequence": 2,
     "shape_pt_lat": 47.374000, "shape_pt_lon": 8.5270, "shape_dist_traveled": None},
    {"csv_row_number": 203, "shape_id": "SL", "shape_pt_sequence": 3,
     "shape_pt_lat": 47.374000, "shape_pt_lon": 8.5230, "shape_dist_traveled": None},
    {"csv_row_number": 204, "shape_id": "SL", "shape_pt_sequence": 4,
     "shape_pt_lat": 47.365000, "shape_pt_lon": 8.5230, "shape_dist_traveled": None},
    {"csv_row_number": 205, "shape_id": "SL", "shape_pt_sequence": 5,
     "shape_pt_lat": 47.365000, "shape_pt_lon": 8.5260, "shape_dist_traveled": None},
    {"csv_row_number": 206, "shape_id": "SL", "shape_pt_sequence": 6,
     "shape_pt_lat": 47.374000, "shape_pt_lon": 8.5260, "shape_dist_traveled": None},
    {"csv_row_number": 207, "shape_id": "SL", "shape_pt_sequence": 7,
     "shape_pt_lat": 47.374000, "shape_pt_lon": 8.5220, "shape_dist_traveled": None},
    {"csv_row_number": 208, "shape_id": "SL", "shape_pt_sequence": 8,
     "shape_pt_lat": 47.365000, "shape_pt_lon": 8.5220, "shape_dist_traveled": None},
    {"csv_row_number": 209, "shape_id": "SL", "shape_pt_sequence": 9,
     "shape_pt_lat": 47.365000, "shape_pt_lon": 8.5260, "shape_dist_traveled": None},
]

_SHAPES_SCHEMA = {
    "csv_row_number": pl.Int64,
    "shape_id": pl.Utf8,
    "shape_pt_sequence": pl.Int64,
    "shape_pt_lat": pl.Float64,
    "shape_pt_lon": pl.Float64,
    "shape_dist_traveled": pl.Float64,
}

_STOPS_SCHEMA = {
    "csv_row_number": pl.Int64,
    "stop_id": pl.Utf8,
    "stop_name": pl.Utf8,
    "stop_lat": pl.Float64,
    "stop_lon": pl.Float64,
    "parent_station": pl.Utf8,
    "location_type": pl.Int64,
}

_TRIPS_SCHEMA = {
    "csv_row_number": pl.Int64,
    "trip_id": pl.Utf8,
    "route_id": pl.Utf8,
    "shape_id": pl.Utf8,
}

_ROUTES_SCHEMA = {
    "csv_row_number": pl.Int64,
    "route_id": pl.Utf8,
    "route_type": pl.Int64,
}

_STOP_TIMES_SCHEMA = {
    "csv_row_number": pl.Int64,
    "trip_id": pl.Utf8,
    "stop_id": pl.Utf8,
    "stop_sequence": pl.Int64,
    "shape_dist_traveled": pl.Float64,
}


def make_shapes(rows: list[dict]) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame(schema=_SHAPES_SCHEMA)
    return pl.DataFrame(rows, schema=_SHAPES_SCHEMA)


def make_stops(rows: list[dict]) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame(schema=_STOPS_SCHEMA)
    return pl.DataFrame(rows, schema=_STOPS_SCHEMA)


def make_trips(rows: list[dict]) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame(schema=_TRIPS_SCHEMA)
    return pl.DataFrame(rows, schema=_TRIPS_SCHEMA)


def make_routes(rows: list[dict]) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame(schema=_ROUTES_SCHEMA)
    return pl.DataFrame(rows, schema=_ROUTES_SCHEMA)


def make_stop_times(rows: list[dict]) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame(schema=_STOP_TIMES_SCHEMA)
    return pl.DataFrame(rows, schema=_STOP_TIMES_SCHEMA)


def _default_feed(
    stops: list[dict],
    stop_times: list[dict],
    shapes: list[dict] | None = None,
    trips: list[dict] | None = None,
    routes: list[dict] | None = None,
    shape_id: str = "S1",
) -> dict[str, pl.DataFrame]:
    if shapes is None:
        shapes = _SHAPE_ROWS
    if trips is None:
        trips = [{"csv_row_number": 1, "trip_id": "T1", "route_id": "R1", "shape_id": shape_id}]
    if routes is None:
        routes = [{"csv_row_number": 1, "route_id": "R1", "route_type": 3}]  # BUS

    return {
        "stops": make_stops(stops),
        "trips": make_trips(trips),
        "routes": make_routes(routes),
        "stop_times": make_stop_times(stop_times),
        "shapes": make_shapes(shapes),
    }


# ---------------------------------------------------------------------------
# Test 6.1: Good feed yields no notices
# ---------------------------------------------------------------------------


def test_good_feed_yields_no_notices() -> None:
    # 3 stops at exact shape points — zero distance → no notice
    stops = [
        {"csv_row_number": 1, "stop_id": "STP0", "stop_name": "Stop 0",
         "stop_lat": 47.365873, "stop_lon": 8.525414, "parent_station": None, "location_type": 0},
        {"csv_row_number": 2, "stop_id": "STP1", "stop_name": "Stop 1",
         "stop_lat": 47.366099, "stop_lon": 8.525154, "parent_station": None, "location_type": 0},
        {"csv_row_number": 3, "stop_id": "STP2", "stop_name": "Stop 2",
         "stop_lat": 47.364108, "stop_lon": 8.525715, "parent_station": None, "location_type": 0},
    ]
    stop_times = [
        {"csv_row_number": 1, "trip_id": "T1", "stop_id": "STP0", "stop_sequence": 0, "shape_dist_traveled": None},
        {"csv_row_number": 2, "trip_id": "T1", "stop_id": "STP1", "stop_sequence": 1, "shape_dist_traveled": None},
        {"csv_row_number": 3, "trip_id": "T1", "stop_id": "STP2", "stop_sequence": 2, "shape_dist_traveled": None},
    ]
    feed = _default_feed(stops, stop_times)
    notices = validate_shape_to_stop_matching(feed, CTX)
    assert notices == []


# ---------------------------------------------------------------------------
# Test 6.2: Too far yields stop_too_far_from_shape
# ---------------------------------------------------------------------------


def test_too_far_yields_stop_too_far_from_shape() -> None:
    # Stop "S1" at (47.366375, 8.527333) is ~130m from the closest shape point.
    # With max_distance=100m, it should fire.
    stops = [
        {"csv_row_number": 1, "stop_id": "STP0", "stop_name": "Stop 0",
         "stop_lat": 47.365873, "stop_lon": 8.525414, "parent_station": None, "location_type": 0},
        {"csv_row_number": 2, "stop_id": "S1", "stop_name": "Stop 1",
         "stop_lat": 47.366375, "stop_lon": 8.527333, "parent_station": None, "location_type": 0},
        {"csv_row_number": 3, "stop_id": "STP2", "stop_name": "Stop 2",
         "stop_lat": 47.364108, "stop_lon": 8.525715, "parent_station": None, "location_type": 0},
    ]
    stop_times = [
        {"csv_row_number": 1, "trip_id": "T1", "stop_id": "STP0", "stop_sequence": 0, "shape_dist_traveled": None},
        {"csv_row_number": 2, "trip_id": "T1", "stop_id": "S1", "stop_sequence": 1, "shape_dist_traveled": None},
        {"csv_row_number": 3, "trip_id": "T1", "stop_id": "STP2", "stop_sequence": 2, "shape_dist_traveled": None},
    ]
    # 100m threshold: stop at ~130m → should fire
    settings = MatchSettings(max_distance_meters=100.0)
    feed = _default_feed(stops, stop_times)
    notices = validate_shape_to_stop_matching(feed, CTX, settings=settings)

    too_far = [n for n in notices if n.code == "stop_too_far_from_shape"]
    assert len(too_far) >= 1, f"Expected stop_too_far_from_shape notice, got {len(too_far)}"
    n = too_far[0]
    assert n.severity == Severity.WARNING
    assert n.fields["stop_id"] == "S1"
    assert n.fields["stop_name"] == "Stop 1"
    assert n.fields["shape_id"] == "S1"
    # Closest point on shape to (47.366375, 8.527333) is (47.365925, 8.525735) at ~130m
    assert abs(n.fields["match"]["lat"] - 47.365925) < 1e-3
    assert abs(n.fields["match"]["lng"] - 8.525735) < 1e-3
    assert abs(n.fields["geo_distance_to_shape"] - 130.0) < 5.0


# ---------------------------------------------------------------------------
# Test 6.3: Too many matches
# ---------------------------------------------------------------------------


def test_too_many_matches_yields_stop_has_too_many_matches_for_shape() -> None:
    # Stop at (47.365, 8.525) near the looping shape that passes nearby 3 times.
    # With threshold=2, the 3 candidates exceed the threshold.
    stops = [
        {"csv_row_number": 1, "stop_id": "STP0", "stop_name": "Stop 0",
         "stop_lat": 47.365000, "stop_lon": 8.525000, "parent_station": None, "location_type": 0},
    ]
    stop_times = [
        {"csv_row_number": 1, "trip_id": "T1", "stop_id": "STP0", "stop_sequence": 0, "shape_dist_traveled": None},
    ]
    # potential_matches_threshold=2 → 3 candidates exceed the threshold
    settings = MatchSettings(max_distance_meters=150.0, potential_matches_threshold=2)
    trips = [{"csv_row_number": 1, "trip_id": "T1", "route_id": "R1", "shape_id": "SL"}]
    feed = _default_feed(stops, stop_times, shapes=_LOOPING_SHAPE_ROWS, trips=trips, shape_id="SL")
    notices = validate_shape_to_stop_matching(feed, CTX, settings=settings)

    too_many = [n for n in notices if n.code == "stop_has_too_many_matches_for_shape"]
    assert len(too_many) == 1, f"Expected 1 too_many_matches notice, got {len(too_many)}"
    n = too_many[0]
    assert n.fields["match_count"] == 3


# ---------------------------------------------------------------------------
# Test 6.4: Out of order
# ---------------------------------------------------------------------------


def test_out_of_order_yields_stops_match_shape_out_of_order() -> None:
    # Stop SA (seq=0) is near the shape end (geo_dist ~283m from start).
    # Stop SB (seq=1) is near the shape beginning (geo_dist ~25m from start).
    # This means the trip stop order is inverted relative to shape order.
    stops = [
        {"csv_row_number": 1, "stop_id": "SA", "stop_name": "Stop A",
         "stop_lat": 47.364107, "stop_lon": 8.525701, "parent_station": None, "location_type": 0},
        {"csv_row_number": 2, "stop_id": "SB", "stop_name": "Stop B",
         "stop_lat": 47.366026, "stop_lon": 8.525189, "parent_station": None, "location_type": 0},
    ]
    stop_times = [
        {"csv_row_number": 1, "trip_id": "T1", "stop_id": "SA", "stop_sequence": 0, "shape_dist_traveled": None},
        {"csv_row_number": 2, "trip_id": "T1", "stop_id": "SB", "stop_sequence": 1, "shape_dist_traveled": None},
    ]
    settings = MatchSettings(max_distance_meters=20.0)
    feed = _default_feed(stops, stop_times)
    notices = validate_shape_to_stop_matching(feed, CTX, settings=settings)

    ooo = [n for n in notices if n.code == "stops_match_shape_out_of_order"]
    assert len(ooo) == 1, f"Expected 1 out-of-order notice, got {len(ooo)}"
    n = ooo[0]
    assert n.severity == Severity.WARNING
    # stop_id1 is the later stop in trip sequence that broke ordering (SB, idx=1)
    assert n.fields["stop_id1"] == "SB"
    # stop_id2 is the earlier stop in trip sequence (SA, idx=0)
    assert n.fields["stop_id2"] == "SA"
    # SB matches near shape beginning
    assert abs(n.fields["match1"]["lat"] - 47.366037) < 1e-3
    assert abs(n.fields["match1"]["lng"] - 8.525214) < 1e-3
    # SA matches near shape end
    assert abs(n.fields["match2"]["lat"] - 47.364108) < 1e-3
    assert abs(n.fields["match2"]["lng"] - 8.525715) < 1e-3


# ---------------------------------------------------------------------------
# Test 6.5: Too far using user distance
# ---------------------------------------------------------------------------


def test_too_far_using_user_distance_yields_notice() -> None:
    # Shape with user_distance; stop with shape_dist_traveled that maps to far point.
    shape_rows = [
        {"csv_row_number": 100, "shape_id": "S1", "shape_pt_sequence": 0,
         "shape_pt_lat": 47.365873, "shape_pt_lon": 8.525414, "shape_dist_traveled": 0.0},
        {"csv_row_number": 101, "shape_id": "S1", "shape_pt_sequence": 1,
         "shape_pt_lat": 47.366099, "shape_pt_lon": 8.525154, "shape_dist_traveled": 100.0},
        {"csv_row_number": 102, "shape_id": "S1", "shape_pt_sequence": 2,
         "shape_pt_lat": 47.364108, "shape_pt_lon": 8.525715, "shape_dist_traveled": 500.0},
    ]
    # Stop is near seq=0 but shape_dist_traveled=500 maps to seq=2 (~280m away)
    stops = [
        {"csv_row_number": 1, "stop_id": "STP0", "stop_name": "Stop 0",
         "stop_lat": 47.365873, "stop_lon": 8.525414, "parent_station": None, "location_type": 0},
    ]
    stop_times = [
        {"csv_row_number": 1, "trip_id": "T1", "stop_id": "STP0",
         "stop_sequence": 0, "shape_dist_traveled": 500.0},
    ]
    settings = MatchSettings(max_distance_meters=100.0)
    feed = _default_feed(stops, stop_times, shapes=shape_rows)
    notices = validate_shape_to_stop_matching(feed, CTX, settings=settings)

    user_dist_notices = [
        n for n in notices if n.code == "stop_too_far_from_shape_using_user_distance"
    ]
    assert len(user_dist_notices) >= 1


# ---------------------------------------------------------------------------
# Test 6.6: Missing table yields no notices (parametrized)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("missing_table", ["stops", "trips", "routes", "stop_times", "shapes"])
def test_missing_table_yields_no_notices(missing_table: str) -> None:
    stops = [
        {"csv_row_number": 1, "stop_id": "STP0", "stop_name": "Stop 0",
         "stop_lat": 47.365873, "stop_lon": 8.525414, "parent_station": None, "location_type": 0},
    ]
    stop_times = [
        {"csv_row_number": 1, "trip_id": "T1", "stop_id": "STP0", "stop_sequence": 0, "shape_dist_traveled": None},
    ]
    feed = _default_feed(stops, stop_times)
    del feed[missing_table]
    notices = validate_shape_to_stop_matching(feed, CTX)
    assert notices == []


# ---------------------------------------------------------------------------
# Test 6.7: Empty table yields no notices
# ---------------------------------------------------------------------------


def test_empty_table_yields_no_notices() -> None:
    stops = [
        {"csv_row_number": 1, "stop_id": "STP0", "stop_name": "Stop 0",
         "stop_lat": 47.365873, "stop_lon": 8.525414, "parent_station": None, "location_type": 0},
    ]
    stop_times = [
        {"csv_row_number": 1, "trip_id": "T1", "stop_id": "STP0", "stop_sequence": 0, "shape_dist_traveled": None},
    ]
    feed = _default_feed(stops, stop_times)
    feed["shapes"] = make_shapes([])  # empty
    notices = validate_shape_to_stop_matching(feed, CTX)
    assert notices == []


# ---------------------------------------------------------------------------
# Test 6.8: Shape without trips yields no notices
# ---------------------------------------------------------------------------


def test_shape_without_trips_yields_no_notices() -> None:
    stops = [
        {"csv_row_number": 1, "stop_id": "STP0", "stop_name": "Stop 0",
         "stop_lat": 47.365873, "stop_lon": 8.525414, "parent_station": None, "location_type": 0},
    ]
    stop_times = [
        {"csv_row_number": 1, "trip_id": "T1", "stop_id": "STP0", "stop_sequence": 0, "shape_dist_traveled": None},
    ]
    # Trip references "OTHER_SHAPE" but shapes table only has "S1"
    trips = [{"csv_row_number": 1, "trip_id": "T1", "route_id": "R1", "shape_id": "OTHER_SHAPE"}]
    feed = _default_feed(stops, stop_times, trips=trips)
    notices = validate_shape_to_stop_matching(feed, CTX)
    assert notices == []


# ---------------------------------------------------------------------------
# Test 6.9: Duplicate trip fingerprint reported once
# ---------------------------------------------------------------------------


def test_duplicate_trip_fingerprint_reported_once() -> None:
    stops = [
        {"csv_row_number": 1, "stop_id": "SFAR", "stop_name": "Far Stop",
         "stop_lat": 47.366375, "stop_lon": 8.527333, "parent_station": None, "location_type": 0},
    ]
    # Two trips with identical stop_time sequences → same hash → T2 is skipped
    trips = [
        {"csv_row_number": 1, "trip_id": "T1", "route_id": "R1", "shape_id": "S1"},
        {"csv_row_number": 2, "trip_id": "T2", "route_id": "R1", "shape_id": "S1"},
    ]
    stop_times = [
        {"csv_row_number": 1, "trip_id": "T1", "stop_id": "SFAR", "stop_sequence": 0, "shape_dist_traveled": None},
        {"csv_row_number": 2, "trip_id": "T2", "stop_id": "SFAR", "stop_sequence": 0, "shape_dist_traveled": None},
    ]
    settings = MatchSettings(max_distance_meters=50.0)
    feed = _default_feed(stops, stop_times, trips=trips)
    notices = validate_shape_to_stop_matching(feed, CTX, settings=settings)

    too_far = [n for n in notices if n.code == "stop_too_far_from_shape"]
    # T2 has same fingerprint as T1 → skipped → only 1 notice
    assert len(too_far) == 1


# ---------------------------------------------------------------------------
# Test 6.10: Same stop too far in two distinct trips reported once
# ---------------------------------------------------------------------------


def test_same_stop_too_far_in_two_trips_reported_once() -> None:
    stops = [
        {"csv_row_number": 1, "stop_id": "SFAR", "stop_name": "Far Stop",
         "stop_lat": 47.366375, "stop_lon": 8.527333, "parent_station": None, "location_type": 0},
        {"csv_row_number": 2, "stop_id": "SNEAR", "stop_name": "Near Stop",
         "stop_lat": 47.365873, "stop_lon": 8.525414, "parent_station": None, "location_type": 0},
    ]
    trips = [
        {"csv_row_number": 1, "trip_id": "T1", "route_id": "R1", "shape_id": "S1"},
        {"csv_row_number": 2, "trip_id": "T2", "route_id": "R1", "shape_id": "S1"},
    ]
    stop_times = [
        # T1: SFAR then SNEAR
        {"csv_row_number": 1, "trip_id": "T1", "stop_id": "SFAR", "stop_sequence": 0, "shape_dist_traveled": None},
        {"csv_row_number": 2, "trip_id": "T1", "stop_id": "SNEAR", "stop_sequence": 1, "shape_dist_traveled": None},
        # T2: SNEAR then SFAR (different sequence → different hash)
        {"csv_row_number": 3, "trip_id": "T2", "stop_id": "SNEAR", "stop_sequence": 0, "shape_dist_traveled": None},
        {"csv_row_number": 4, "trip_id": "T2", "stop_id": "SFAR", "stop_sequence": 1, "shape_dist_traveled": None},
    ]
    settings = MatchSettings(max_distance_meters=50.0)
    feed = _default_feed(stops, stop_times, trips=trips)
    notices = validate_shape_to_stop_matching(feed, CTX, settings=settings)

    too_far = [n for n in notices if n.code == "stop_too_far_from_shape"
               and n.fields["stop_id"] == "SFAR"]
    # SFAR should only be reported once (deduplication via reported_stop_ids)
    assert len(too_far) == 1, f"Expected 1 notice for SFAR, got {len(too_far)}"


# ---------------------------------------------------------------------------
# Test 6.11: Empty stop_id in stop_time is skipped
# ---------------------------------------------------------------------------


def test_empty_stop_id_in_stop_time_is_skipped() -> None:
    stops = [
        {"csv_row_number": 1, "stop_id": "STP0", "stop_name": "Stop 0",
         "stop_lat": 47.365873, "stop_lon": 8.525414, "parent_station": None, "location_type": 0},
    ]
    # stop_id="" → falls back to (0,0) coords and would be too far, but notice skipped
    stop_times = [
        {"csv_row_number": 1, "trip_id": "T1", "stop_id": "", "stop_sequence": 0, "shape_dist_traveled": None},
    ]
    settings = MatchSettings(max_distance_meters=50.0)
    feed = _default_feed(stops, stop_times)
    notices = validate_shape_to_stop_matching(feed, CTX, settings=settings)
    assert notices == [], f"Expected no notices for empty stop_id, got {notices}"


# ---------------------------------------------------------------------------
# Test 6.12: Stop not in stops table falls back to (0, 0)
# ---------------------------------------------------------------------------


def test_stop_not_in_stops_table_falls_back_to_zero_zero() -> None:
    stops = [
        {"csv_row_number": 1, "stop_id": "KNOWN", "stop_name": "Known",
         "stop_lat": 47.365873, "stop_lon": 8.525414, "parent_station": None, "location_type": 0},
    ]
    stop_times = [
        # UNKNOWN_STOP not in stops table → (0.0, 0.0) → far from shape
        {"csv_row_number": 1, "trip_id": "T1", "stop_id": "UNKNOWN_STOP",
         "stop_sequence": 0, "shape_dist_traveled": None},
    ]
    settings = MatchSettings(max_distance_meters=100.0)
    feed = _default_feed(stops, stop_times)
    notices = validate_shape_to_stop_matching(feed, CTX, settings=settings)

    too_far = [n for n in notices if n.code == "stop_too_far_from_shape"]
    assert len(too_far) == 1
    assert too_far[0].fields["stop_id"] == "UNKNOWN_STOP"
    assert too_far[0].fields["stop_name"] == ""


# ---------------------------------------------------------------------------
# Test 6.13: Stop with parent station resolves coordinates
# ---------------------------------------------------------------------------


def test_stop_with_parent_station_resolves_coordinates() -> None:
    stops = [
        # Child stop has no lat/lon but has a parent
        {"csv_row_number": 1, "stop_id": "CHILD", "stop_name": "Child",
         "stop_lat": None, "stop_lon": None, "parent_station": "PARENT", "location_type": 0},
        # Parent stop has coordinates near the shape
        {"csv_row_number": 2, "stop_id": "PARENT", "stop_name": "Parent",
         "stop_lat": 47.365873, "stop_lon": 8.525414, "parent_station": None, "location_type": 1},
    ]
    stop_times = [
        {"csv_row_number": 1, "trip_id": "T1", "stop_id": "CHILD",
         "stop_sequence": 0, "shape_dist_traveled": None},
    ]
    settings = MatchSettings(max_distance_meters=100.0)
    feed = _default_feed(stops, stop_times)
    notices = validate_shape_to_stop_matching(feed, CTX, settings=settings)

    too_far = [n for n in notices if n.code == "stop_too_far_from_shape"]
    assert len(too_far) == 0, f"Expected no too_far after parent resolution, got {notices}"


# ---------------------------------------------------------------------------
# Test 6.14: RAIL route uses large station threshold
# ---------------------------------------------------------------------------


def test_rail_route_uses_large_station_threshold() -> None:
    # SFIRST is ~130m from the shape.
    # BUS: threshold=100m → notice fires.
    # RAIL: first stop threshold = 100m * 4 = 400m → no notice.
    stops = [
        {"csv_row_number": 1, "stop_id": "SFIRST", "stop_name": "First",
         "stop_lat": 47.366375, "stop_lon": 8.527333, "parent_station": None, "location_type": 0},
        {"csv_row_number": 2, "stop_id": "SLAST", "stop_name": "Last",
         "stop_lat": 47.364108, "stop_lon": 8.525715, "parent_station": None, "location_type": 0},
    ]
    stop_times = [
        {"csv_row_number": 1, "trip_id": "T1", "stop_id": "SFIRST", "stop_sequence": 0, "shape_dist_traveled": None},
        {"csv_row_number": 2, "trip_id": "T1", "stop_id": "SLAST", "stop_sequence": 1, "shape_dist_traveled": None},
    ]
    rail_routes = [{"csv_row_number": 1, "route_id": "R1", "route_type": 2}]  # RAIL
    bus_routes = [{"csv_row_number": 1, "route_id": "R1", "route_type": 3}]   # BUS

    # RAIL first stop → 400m threshold → no notice for SFIRST at 130m
    feed_rail = _default_feed(stops, stop_times, routes=rail_routes)
    notices_rail = validate_shape_to_stop_matching(feed_rail, CTX)
    too_far_rail = [n for n in notices_rail if n.code == "stop_too_far_from_shape"
                    and n.fields["stop_id"] == "SFIRST"]
    assert len(too_far_rail) == 0, (
        f"RAIL first stop within 400m should not trigger notice, got {too_far_rail}"
    )

    # BUS first stop → 100m threshold → notice fires for SFIRST at 130m
    feed_bus = _default_feed(stops, stop_times, routes=bus_routes)
    notices_bus = validate_shape_to_stop_matching(feed_bus, CTX)
    too_far_bus = [n for n in notices_bus if n.code == "stop_too_far_from_shape"
                   and n.fields["stop_id"] == "SFIRST"]
    assert len(too_far_bus) >= 1, (
        f"BUS first stop at 130m should trigger notice with 100m threshold"
    )


# ---------------------------------------------------------------------------
# Test 6.15: Only stop user_distance, no shape user_distance → skip user matching
# ---------------------------------------------------------------------------


def test_only_stop_user_distance_no_shape_user_distance_skips_user_matching() -> None:
    # Shapes have no shape_dist_traveled (all None)
    shapes_no_user = [
        {"csv_row_number": 100, "shape_id": "S1", "shape_pt_sequence": 0,
         "shape_pt_lat": 47.365873, "shape_pt_lon": 8.525414, "shape_dist_traveled": None},
        {"csv_row_number": 101, "shape_id": "S1", "shape_pt_sequence": 1,
         "shape_pt_lat": 47.366099, "shape_pt_lon": 8.525154, "shape_dist_traveled": None},
        {"csv_row_number": 102, "shape_id": "S1", "shape_pt_sequence": 2,
         "shape_pt_lat": 47.364108, "shape_pt_lon": 8.525715, "shape_dist_traveled": None},
    ]
    # Stop has shape_dist_traveled but shape does not → user-distance matching should be skipped
    stops = [
        {"csv_row_number": 1, "stop_id": "SFAR", "stop_name": "Far",
         "stop_lat": 47.366375, "stop_lon": 8.527333, "parent_station": None, "location_type": 0},
    ]
    stop_times = [
        {"csv_row_number": 1, "trip_id": "T1", "stop_id": "SFAR",
         "stop_sequence": 0, "shape_dist_traveled": 100.0},
    ]
    settings = MatchSettings(max_distance_meters=50.0)
    feed = _default_feed(stops, stop_times, shapes=shapes_no_user)
    notices = validate_shape_to_stop_matching(feed, CTX, settings=settings)

    codes = {n.code for n in notices}
    assert "stop_too_far_from_shape_using_user_distance" not in codes, (
        "User-distance matching should be skipped when shape has no shape_dist_traveled"
    )
    # Geo-distance matching should still fire (stop is ~130m from shape, threshold=50m)
    assert "stop_too_far_from_shape" in codes
