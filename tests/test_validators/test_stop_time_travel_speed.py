"""Tests for StopTimeTravelSpeedValidator."""

from __future__ import annotations

import datetime

import polars as pl
import pytest

from gtfs_validator.context import ValidationContext, build_stop_location_cache
from gtfs_validator.notices import Severity
from gtfs_validator.validators.stop_time_travel_speed import (
    get_speed_kph,
    haversine_km,
    validate_stop_time_travel_speed,
)

CTX = ValidationContext(country_code="US", date_for_validation=datetime.date(2024, 1, 1))

# ---------------------------------------------------------------------------
# DataFrame schema helpers
# ---------------------------------------------------------------------------

_STOP_TIMES_SCHEMA: dict[str, type] = {
    "csv_row_number": pl.Int64,
    "trip_id": pl.Utf8,
    "stop_id": pl.Utf8,
    "stop_sequence": pl.Int64,
    "arrival_time": pl.Int64,
    "departure_time": pl.Int64,
}

_TRIPS_SCHEMA: dict[str, type] = {
    "csv_row_number": pl.Int64,
    "trip_id": pl.Utf8,
    "route_id": pl.Utf8,
}

_ROUTES_SCHEMA: dict[str, type] = {
    "route_id": pl.Utf8,
    "route_type": pl.Int64,
}

_STOPS_SCHEMA: dict[str, type] = {
    "stop_id": pl.Utf8,
    "stop_name": pl.Utf8,
    "stop_lat": pl.Float64,
    "stop_lon": pl.Float64,
    "parent_station": pl.Utf8,
}


def make_feed(
    stop_times: list[dict],
    trips: list[dict],
    routes: list[dict],
    stops: list[dict],
) -> dict[str, pl.DataFrame]:
    return {
        "stop_times": pl.DataFrame(stop_times, schema=_STOP_TIMES_SCHEMA),
        "trips": pl.DataFrame(trips, schema=_TRIPS_SCHEMA),
        "routes": pl.DataFrame(routes, schema=_ROUTES_SCHEMA),
        "stops": pl.DataFrame(stops, schema=_STOPS_SCHEMA),
    }


# ---------------------------------------------------------------------------
# Two-stop base feed builder
# ---------------------------------------------------------------------------

def _two_stop_feed(
    route_type: int,
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
    dep1: int,
    arr2: int,
) -> dict[str, pl.DataFrame]:
    return make_feed(
        stop_times=[
            {"csv_row_number": 1, "trip_id": "T1", "stop_id": "S1",
             "stop_sequence": 1, "arrival_time": dep1, "departure_time": dep1},
            {"csv_row_number": 2, "trip_id": "T1", "stop_id": "S2",
             "stop_sequence": 2, "arrival_time": arr2, "departure_time": arr2},
        ],
        trips=[{"csv_row_number": 10, "trip_id": "T1", "route_id": "R1"}],
        routes=[{"route_id": "R1", "route_type": route_type}],
        stops=[
            {"stop_id": "S1", "stop_name": "Stop 1", "stop_lat": lat1, "stop_lon": lon1,
             "parent_station": None},
            {"stop_id": "S2", "stop_name": "Stop 2", "stop_lat": lat2, "stop_lon": lon2,
             "parent_station": None},
        ],
    )


# Approx 5 km apart: (0, 0) to (0, 0.045)
LAT1, LON1 = 0.0, 0.0
LAT2, LON2 = 0.0, 0.045  # ≈ 5.01 km


# ---------------------------------------------------------------------------
# Unit tests for pure helpers
# ---------------------------------------------------------------------------


def test_backward_in_time_speed_computation() -> None:
    """Backward-in-time data treated as 1-minute forward travel."""
    # dep=28860 (08:01:00), arr=28800 (08:00:00) → raw=-60 → use 60 s
    speed = get_speed_kph(1.0, 28860, 28800)
    assert abs(speed - 60.0) < 1e-6


def test_teleport_speed_computation() -> None:
    """Zero time delta handled without division by zero."""
    t = 36000  # arbitrary
    speed = get_speed_kph(2.0, t, t)
    assert abs(speed - 120.0) < 1e-6


# ---------------------------------------------------------------------------
# Guard clause tests
# ---------------------------------------------------------------------------


def test_missing_stop_times_returns_empty() -> None:
    feed = {
        "trips": pl.DataFrame(schema=_TRIPS_SCHEMA),
        "routes": pl.DataFrame(schema=_ROUTES_SCHEMA),
        "stops": pl.DataFrame(schema=_STOPS_SCHEMA),
    }
    assert validate_stop_time_travel_speed(feed, CTX) == []


def test_missing_trips_returns_empty() -> None:
    feed = {
        "stop_times": pl.DataFrame(schema=_STOP_TIMES_SCHEMA),
        "routes": pl.DataFrame(schema=_ROUTES_SCHEMA),
        "stops": pl.DataFrame(schema=_STOPS_SCHEMA),
    }
    assert validate_stop_time_travel_speed(feed, CTX) == []


def test_missing_routes_returns_empty() -> None:
    feed = {
        "stop_times": pl.DataFrame(schema=_STOP_TIMES_SCHEMA),
        "trips": pl.DataFrame(schema=_TRIPS_SCHEMA),
        "stops": pl.DataFrame(schema=_STOPS_SCHEMA),
    }
    assert validate_stop_time_travel_speed(feed, CTX) == []


def test_missing_stops_returns_empty() -> None:
    feed = {
        "stop_times": pl.DataFrame(schema=_STOP_TIMES_SCHEMA),
        "trips": pl.DataFrame(schema=_TRIPS_SCHEMA),
        "routes": pl.DataFrame(schema=_ROUTES_SCHEMA),
    }
    assert validate_stop_time_travel_speed(feed, CTX) == []


# ---------------------------------------------------------------------------
# Consecutive-stop tests
# ---------------------------------------------------------------------------


def test_bus_100kph_no_notice() -> None:
    """Bus route, ~5 km, 200 s travel → ~90 km/h → no notice."""
    feed = _two_stop_feed(3, LAT1, LON1, LAT2, LON2, dep1=0, arr2=200)
    assert validate_stop_time_travel_speed(feed, CTX) == []


def test_bus_180kph_consecutive_notice() -> None:
    """Bus route, ~5 km, 100 s → ~180 km/h > 150 → one notice."""
    feed = _two_stop_feed(3, LAT1, LON1, LAT2, LON2, dep1=0, arr2=100)
    notices = validate_stop_time_travel_speed(feed, CTX)
    assert len(notices) == 1
    n = notices[0]
    assert n.code == "fast_travel_between_consecutive_stops"
    assert n.severity == Severity.WARNING
    assert n.fields["stop_id1"] == "S1"
    assert n.fields["stop_id2"] == "S2"
    assert n.fields["speed_kph"] > 150.0


def test_rail_180kph_no_notice() -> None:
    """Rail (max 500 km/h), same setup → no notice."""
    feed = _two_stop_feed(2, LAT1, LON1, LAT2, LON2, dep1=0, arr2=100)
    assert validate_stop_time_travel_speed(feed, CTX) == []


def test_minute_accuracy_no_notice() -> None:
    """Minute-resolution buffer: same minute boundary → effective time = 60 s, speed well under 500."""
    # ~2 km apart: (0,0) to (0,0.018)
    lat2, lon2 = 0.0, 0.018  # ≈ 2 km
    t = 3600  # exact minute boundary, % 60 == 0
    feed = _two_stop_feed(2, 0.0, 0.0, lat2, lon2, dep1=t, arr2=t)
    assert validate_stop_time_travel_speed(feed, CTX) == []


def test_ferry_speed_threshold() -> None:
    """Ferry (max 80 km/h), ~5 km, 200 s → ~90 km/h > 80 → one notice."""
    feed = _two_stop_feed(4, LAT1, LON1, LAT2, LON2, dep1=0, arr2=200)
    notices = validate_stop_time_travel_speed(feed, CTX)
    assert len(notices) == 1
    assert notices[0].code == "fast_travel_between_consecutive_stops"


def test_cable_tram_speed_threshold() -> None:
    """Cable tram (max 30 km/h), ~1 km, 100 s → 36 km/h > 30 → one notice."""
    # (0,0) to (0,0.009): ≈ 1 km
    feed = _two_stop_feed(5, 0.0, 0.0, 0.0, 0.009, dep1=0, arr2=100)
    notices = validate_stop_time_travel_speed(feed, CTX)
    assert len(notices) == 1
    assert notices[0].code == "fast_travel_between_consecutive_stops"


def test_unknown_route_type_uses_default() -> None:
    """Unknown route type 99 → default 200 km/h. 50 km in 600 s → 300 km/h > 200."""
    # ~50 km: (0,0) to (0,0.45) — also exceeds 10 km so a far-stop notice may also appear.
    feed = _two_stop_feed(99, 0.0, 0.0, 0.0, 0.45, dep1=0, arr2=600)
    notices = validate_stop_time_travel_speed(feed, CTX)
    assert len(notices) >= 1
    assert any(n.code == "fast_travel_between_consecutive_stops" for n in notices)


def test_single_stop_per_trip_no_notice() -> None:
    """Only one stop time per trip — no pairs to evaluate."""
    feed = make_feed(
        stop_times=[
            {"csv_row_number": 1, "trip_id": "T1", "stop_id": "S1",
             "stop_sequence": 1, "arrival_time": 0, "departure_time": 0},
        ],
        trips=[{"csv_row_number": 10, "trip_id": "T1", "route_id": "R1"}],
        routes=[{"route_id": "R1", "route_type": 3}],
        stops=[
            {"stop_id": "S1", "stop_name": "Stop 1", "stop_lat": 0.0, "stop_lon": 0.0,
             "parent_station": None},
        ],
    )
    assert validate_stop_time_travel_speed(feed, CTX) == []


def test_trip_id_not_in_trips_skipped() -> None:
    """stop_times references trip not in trips → null route_type → filtered out."""
    feed = make_feed(
        stop_times=[
            {"csv_row_number": 1, "trip_id": "MISSING", "stop_id": "S1",
             "stop_sequence": 1, "arrival_time": 0, "departure_time": 0},
            {"csv_row_number": 2, "trip_id": "MISSING", "stop_id": "S2",
             "stop_sequence": 2, "arrival_time": 100, "departure_time": 100},
        ],
        trips=[{"csv_row_number": 10, "trip_id": "T1", "route_id": "R1"}],
        routes=[{"route_id": "R1", "route_type": 3}],
        stops=[
            {"stop_id": "S1", "stop_name": "Stop 1", "stop_lat": 0.0, "stop_lon": 0.0,
             "parent_station": None},
            {"stop_id": "S2", "stop_name": "Stop 2", "stop_lat": 0.0, "stop_lon": 0.045,
             "parent_station": None},
        ],
    )
    assert validate_stop_time_travel_speed(feed, CTX) == []


def test_route_id_not_in_routes_skipped() -> None:
    """trip references route not in routes → null route_type → filtered."""
    feed = make_feed(
        stop_times=[
            {"csv_row_number": 1, "trip_id": "T1", "stop_id": "S1",
             "stop_sequence": 1, "arrival_time": 0, "departure_time": 0},
            {"csv_row_number": 2, "trip_id": "T1", "stop_id": "S2",
             "stop_sequence": 2, "arrival_time": 100, "departure_time": 100},
        ],
        trips=[{"csv_row_number": 10, "trip_id": "T1", "route_id": "MISSING"}],
        routes=[{"route_id": "R1", "route_type": 3}],
        stops=[
            {"stop_id": "S1", "stop_name": "Stop 1", "stop_lat": 0.0, "stop_lon": 0.0,
             "parent_station": None},
            {"stop_id": "S2", "stop_name": "Stop 2", "stop_lat": 0.0, "stop_lon": 0.045,
             "parent_station": None},
        ],
    )
    assert validate_stop_time_travel_speed(feed, CTX) == []


# ---------------------------------------------------------------------------
# Parent station coordinate tests
# ---------------------------------------------------------------------------


def test_stop_with_parent_station_coordinates() -> None:
    """Stop with null lat/lon but valid parent_station → coordinate resolved from parent."""
    # S1 has no coords, parent_station=S0 which has coords.
    # S2 has coords. Fast travel should be detected.
    # ~5 km between S0 and S2; 100 s travel → ~180 km/h > 150 bus limit
    feed = make_feed(
        stop_times=[
            {"csv_row_number": 1, "trip_id": "T1", "stop_id": "S1",
             "stop_sequence": 1, "arrival_time": 0, "departure_time": 0},
            {"csv_row_number": 2, "trip_id": "T1", "stop_id": "S2",
             "stop_sequence": 2, "arrival_time": 100, "departure_time": 100},
        ],
        trips=[{"csv_row_number": 10, "trip_id": "T1", "route_id": "R1"}],
        routes=[{"route_id": "R1", "route_type": 3}],
        stops=[
            {"stop_id": "S0", "stop_name": "Parent", "stop_lat": 0.0, "stop_lon": 0.0,
             "parent_station": None},
            {"stop_id": "S1", "stop_name": "Child", "stop_lat": None, "stop_lon": None,
             "parent_station": "S0"},
            {"stop_id": "S2", "stop_name": "Stop 2", "stop_lat": 0.0, "stop_lon": 0.045,
             "parent_station": None},
        ],
    )
    notices = validate_stop_time_travel_speed(feed, CTX)
    assert len(notices) >= 1
    assert notices[0].code == "fast_travel_between_consecutive_stops"


# ---------------------------------------------------------------------------
# Far-stop tests
# ---------------------------------------------------------------------------

# Three stops: 0°0' → 0°0.05' → 0°0.10' ≈ 5.56 km each leg → ~11.1 km total
_S0_LAT, _S0_LON = 0.0, 0.0
_S1_LAT, _S1_LON = 0.0, 0.05
_S2_LAT, _S2_LON = 0.0, 0.10


def _three_stop_feed(route_type: int, t0_dep: int, t1_arr: int, t1_dep: int, t2_arr: int) -> dict[str, pl.DataFrame]:
    return make_feed(
        stop_times=[
            {"csv_row_number": 1, "trip_id": "T1", "stop_id": "S0",
             "stop_sequence": 1, "arrival_time": t0_dep, "departure_time": t0_dep},
            {"csv_row_number": 2, "trip_id": "T1", "stop_id": "S1",
             "stop_sequence": 2, "arrival_time": t1_arr, "departure_time": t1_dep},
            {"csv_row_number": 3, "trip_id": "T1", "stop_id": "S2",
             "stop_sequence": 3, "arrival_time": t2_arr, "departure_time": t2_arr},
        ],
        trips=[{"csv_row_number": 10, "trip_id": "T1", "route_id": "R1"}],
        routes=[{"route_id": "R1", "route_type": route_type}],
        stops=[
            {"stop_id": "S0", "stop_name": "Stop 0", "stop_lat": _S0_LAT, "stop_lon": _S0_LON,
             "parent_station": None},
            {"stop_id": "S1", "stop_name": "Stop 1", "stop_lat": _S1_LAT, "stop_lon": _S1_LON,
             "parent_station": None},
            {"stop_id": "S2", "stop_name": "Stop 2", "stop_lat": _S2_LAT, "stop_lon": _S2_LON,
             "parent_station": None},
        ],
    )


def test_far_stops_acceptable_speed_no_notice() -> None:
    """Three stops, ~11 km total, 300 s travel → ~126 km/h < 150 → no notice."""
    # Each leg ≈ 5.56 km; total 300 s
    feed = _three_stop_feed(3, t0_dep=0, t1_arr=150, t1_dep=150, t2_arr=300)
    notices = validate_stop_time_travel_speed(feed, CTX)
    assert notices == []


def test_far_stops_too_fast_all_three_notices() -> None:
    """Three stops, ~11 km total, 150 s travel → ~252 km/h > 150. Expect 3 notices."""
    feed = _three_stop_feed(3, t0_dep=0, t1_arr=75, t1_dep=75, t2_arr=150)
    notices = validate_stop_time_travel_speed(feed, CTX)
    codes = [n.code for n in notices]
    consecutive = [c for c in codes if c == "fast_travel_between_consecutive_stops"]
    far = [c for c in codes if c == "fast_travel_between_far_stops"]
    assert len(consecutive) == 2
    assert len(far) == 1


def test_notice_ordering_for_consecutive_and_far_notices() -> None:
    feed = _three_stop_feed(3, t0_dep=0, t1_arr=75, t1_dep=75, t2_arr=150)
    notices = validate_stop_time_travel_speed(feed, CTX)
    assert [n.code for n in notices] == [
        "fast_travel_between_consecutive_stops",
        "fast_travel_between_consecutive_stops",
        "fast_travel_between_far_stops",
    ]


def test_middle_stop_missing_time_bridges_over() -> None:
    """Middle stop with unknown stop_id (no coords) → start stays at stop 0; stop 2 fast."""
    # Middle stop_id is "UNKNOWN" — not in stops table → resolve_stop_latlng returns None
    # So consecutive check: start=0, end=UNKNOWN → no coords → continue; start stays 0
    #                        start=0, end=S2 → bridges across, distance=straight 0→2
    # Far-stop: aborts because UNKNOWN not in stops_index
    feed = make_feed(
        stop_times=[
            {"csv_row_number": 1, "trip_id": "T1", "stop_id": "S0",
             "stop_sequence": 1, "arrival_time": 0, "departure_time": 0},
            {"csv_row_number": 2, "trip_id": "T1", "stop_id": "UNKNOWN",
             "stop_sequence": 2, "arrival_time": None, "departure_time": None},
            {"csv_row_number": 3, "trip_id": "T1", "stop_id": "S2",
             "stop_sequence": 3, "arrival_time": 150, "departure_time": 150},
        ],
        trips=[{"csv_row_number": 10, "trip_id": "T1", "route_id": "R1"}],
        routes=[{"route_id": "R1", "route_type": 3}],
        stops=[
            {"stop_id": "S0", "stop_name": "Stop 0", "stop_lat": _S0_LAT, "stop_lon": _S0_LON,
             "parent_station": None},
            {"stop_id": "S2", "stop_name": "Stop 2", "stop_lat": _S2_LAT, "stop_lon": _S2_LON,
             "parent_station": None},
        ],
    )
    notices = validate_stop_time_travel_speed(feed, CTX)
    consecutive = [n for n in notices if n.code == "fast_travel_between_consecutive_stops"]
    # Should emit one consecutive notice bridging stop 0 to stop 2
    assert len(consecutive) == 1
    assert consecutive[0].fields["stop_id1"] == "S0"
    assert consecutive[0].fields["stop_id2"] == "S2"


def test_cached_stop_lookup_matches_uncached_behavior() -> None:
    feed = make_feed(
        stop_times=[
            {"csv_row_number": 1, "trip_id": "T1", "stop_id": "S1",
             "stop_sequence": 1, "arrival_time": 0, "departure_time": 0},
            {"csv_row_number": 2, "trip_id": "T1", "stop_id": "S2",
             "stop_sequence": 2, "arrival_time": 100, "departure_time": 100},
        ],
        trips=[{"csv_row_number": 10, "trip_id": "T1", "route_id": "R1"}],
        routes=[{"route_id": "R1", "route_type": 3}],
        stops=[
            {"stop_id": "S0", "stop_name": "Parent", "stop_lat": 0.0, "stop_lon": 0.0,
             "parent_station": None},
            {"stop_id": "S1", "stop_name": "Child", "stop_lat": None, "stop_lon": None,
             "parent_station": "S0"},
            {"stop_id": "S2", "stop_name": "Stop 2", "stop_lat": 0.0, "stop_lon": 0.045,
             "parent_station": None},
        ],
    )
    uncached = validate_stop_time_travel_speed(feed, CTX)
    cached_ctx = ValidationContext(
        country_code=CTX.country_code,
        date_for_validation=CTX.date_for_validation,
        stop_location_cache=build_stop_location_cache(feed["stops"]),
    )
    cached = validate_stop_time_travel_speed(feed, cached_ctx)
    assert cached == uncached


def test_far_stop_check_emits_at_most_one_notice_per_trip() -> None:
    """Four stops forming a chain > 10 km total with all pairs too fast → one far notice."""
    # Four stops ~5.56 km each: total ≈ 16.7 km.  Very short time → very fast.
    feed = make_feed(
        stop_times=[
            {"csv_row_number": 1, "trip_id": "T1", "stop_id": "S0",
             "stop_sequence": 1, "arrival_time": 0, "departure_time": 0},
            {"csv_row_number": 2, "trip_id": "T1", "stop_id": "S1",
             "stop_sequence": 2, "arrival_time": 50, "departure_time": 50},
            {"csv_row_number": 3, "trip_id": "T1", "stop_id": "S2",
             "stop_sequence": 3, "arrival_time": 100, "departure_time": 100},
            {"csv_row_number": 4, "trip_id": "T1", "stop_id": "S3",
             "stop_sequence": 4, "arrival_time": 150, "departure_time": 150},
        ],
        trips=[{"csv_row_number": 10, "trip_id": "T1", "route_id": "R1"}],
        routes=[{"route_id": "R1", "route_type": 3}],
        stops=[
            {"stop_id": "S0", "stop_name": "Stop 0", "stop_lat": 0.0, "stop_lon": 0.0,
             "parent_station": None},
            {"stop_id": "S1", "stop_name": "Stop 1", "stop_lat": 0.0, "stop_lon": 0.05,
             "parent_station": None},
            {"stop_id": "S2", "stop_name": "Stop 2", "stop_lat": 0.0, "stop_lon": 0.10,
             "parent_station": None},
            {"stop_id": "S3", "stop_name": "Stop 3", "stop_lat": 0.0, "stop_lon": 0.15,
             "parent_station": None},
        ],
    )
    notices = validate_stop_time_travel_speed(feed, CTX)
    far_notices = [n for n in notices if n.code == "fast_travel_between_far_stops"]
    assert len(far_notices) == 1


def test_geojson_location_distance_zero() -> None:
    """Stop_id absent from stops table → both adjacent segment distances are 0.0."""
    from gtfs_validator.validators.stop_time_travel_speed import (
        resolve_stop_latlng,
        haversine_km as hkm,
    )
    stops_index = {
        "S0": {"stop_id": "S0", "stop_lat": 0.0, "stop_lon": 0.0, "stop_name": "A", "parent_station": None},
        "S1": {"stop_id": "S1", "stop_lat": 0.0, "stop_lon": 0.045, "stop_name": "B", "parent_station": None},
    }
    rows = [
        {"stop_id": "S0"},
        {"stop_id": "GEOJSON_UNKNOWN"},
        {"stop_id": "S1"},
    ]
    distances: list[float] = []
    for i in range(len(rows) - 1):
        start_ll = resolve_stop_latlng(rows[i]["stop_id"], stops_index)
        end_ll = resolve_stop_latlng(rows[i + 1]["stop_id"], stops_index)
        if start_ll is None or end_ll is None:
            distances.append(0.0)
        else:
            distances.append(hkm(*start_ll, *end_ll))

    assert distances[0] == 0.0  # S0 → GEOJSON_UNKNOWN (unknown end)
    assert distances[1] == 0.0  # GEOJSON_UNKNOWN → S1 (unknown start)
