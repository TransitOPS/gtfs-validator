"""Unit tests for shape_to_stop_matching_util geometry helpers."""

from __future__ import annotations

import math

import pytest

from gtfs_validator.validators.shape_to_stop_matching_util import (
    CandidateMatch,
    ShapeArrays,
    ShapePoint,
    StopPoint,
    build_shape_arrays,
    build_shape_points,
    closest_point_on_edge,
    compute_trip_hash,
    find_best_assignment,
    find_closest_on_shape,
    find_potential_matches,
    geo_distance_meters,
    latlng_to_unit_vector,
    resolve_stop_location,
    unit_vector_to_latlng,
    MatchSettings,
    _closest_points_on_edges_vec,
    _haversine_meters_vec,
    _seg_fraction_vec,
    _uv_to_latlng_vec,
    build_shape_spatial_index,
)

# ---------------------------------------------------------------------------
# geo_distance_meters
# ---------------------------------------------------------------------------


def test_geo_distance_meters_known_values() -> None:
    # Zurich to Geneva (approx 225 km)
    d = geo_distance_meters(47.3769, 8.5417, 46.2044, 6.1432)
    assert abs(d - 225_000) < 5_000, f"Expected ~225 km, got {d}"

    # Same point
    assert geo_distance_meters(47.0, 8.0, 47.0, 8.0) == pytest.approx(0.0, abs=1e-6)

    # New York to London (approx 5.5 Mm)
    d2 = geo_distance_meters(40.7128, -74.0060, 51.5074, -0.1278)
    assert abs(d2 - 5_570_000) < 50_000, f"Expected ~5.57 Mm, got {d2}"


# ---------------------------------------------------------------------------
# latlng_to_unit_vector / unit_vector_to_latlng round-trip
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "lat,lon",
    [
        (0.0, 0.0),
        (47.366, 8.525),
        (-33.8688, 151.2093),
        (90.0, 0.0),
        (-90.0, 0.0),
        (0.0, 180.0),
        (0.0, -180.0),
    ],
)
def test_latlng_roundtrip(lat: float, lon: float) -> None:
    v = latlng_to_unit_vector(lat, lon)
    lat2, lon2 = unit_vector_to_latlng(v)
    assert abs(lat2 - lat) < 1e-9, f"lat mismatch: {lat} -> {lat2}"
    # Longitude is degenerate at poles
    if abs(lat) < 89.9:
        assert abs(lon2 - lon) < 1e-9, f"lon mismatch: {lon} -> {lon2}"


# ---------------------------------------------------------------------------
# closest_point_on_edge
# ---------------------------------------------------------------------------


def test_closest_point_on_edge_midpoint() -> None:
    """Stop at the midpoint of an edge should project to near the midpoint."""
    # Two points on the equator separated by a small angle
    a_lat, a_lon = 0.0, 0.0
    b_lat, b_lon = 0.0, 1.0
    mid_lat, mid_lon = 0.0, 0.5

    # Stop just north of midpoint (should project to midpoint on edge)
    stop_lat, stop_lon = 0.001, 0.5

    av = latlng_to_unit_vector(a_lat, a_lon)
    bv = latlng_to_unit_vector(b_lat, b_lon)
    pv = latlng_to_unit_vector(stop_lat, stop_lon)

    rlat, rlon = closest_point_on_edge(pv, av, bv)
    assert abs(rlat - mid_lat) < 0.01
    assert abs(rlon - mid_lon) < 0.01


def test_closest_point_on_edge_clamps_to_endpoint() -> None:
    """Stop beyond the end of the arc should return nearest endpoint."""
    # Arc from (0,0) to (0,1)
    av = latlng_to_unit_vector(0.0, 0.0)
    bv = latlng_to_unit_vector(0.0, 1.0)

    # Stop at (0, 2) — beyond b
    pv = latlng_to_unit_vector(0.0, 2.0)
    rlat, rlon = closest_point_on_edge(pv, av, bv)
    assert abs(rlat - 0.0) < 1e-6
    assert abs(rlon - 1.0) < 0.01  # clamps to b

    # Stop at (0, -1) — before a
    pv2 = latlng_to_unit_vector(0.0, -1.0)
    rlat2, rlon2 = closest_point_on_edge(pv2, av, bv)
    assert abs(rlat2 - 0.0) < 1e-6
    assert abs(rlon2 - 0.0) < 0.01  # clamps to a


# ---------------------------------------------------------------------------
# build_shape_points
# ---------------------------------------------------------------------------


def test_build_shape_points_accumulates_distance() -> None:
    rows = [
        {"shape_pt_sequence": 1, "shape_pt_lat": 47.365873, "shape_pt_lon": 8.525414, "shape_dist_traveled": None},
        {"shape_pt_sequence": 2, "shape_pt_lat": 47.365872, "shape_pt_lon": 8.525376, "shape_dist_traveled": None},
        {"shape_pt_sequence": 3, "shape_pt_lat": 47.366099, "shape_pt_lon": 8.525154, "shape_dist_traveled": None},
    ]
    pts = build_shape_points(rows)
    assert len(pts) == 3
    assert pts[0].geo_distance == pytest.approx(0.0)
    assert pts[1].geo_distance > 0.0
    assert pts[2].geo_distance > pts[1].geo_distance


def test_build_shape_points_running_max_user_distance() -> None:
    rows = [
        {"shape_pt_sequence": 1, "shape_pt_lat": 47.0, "shape_pt_lon": 8.0, "shape_dist_traveled": 0.0},
        {"shape_pt_sequence": 2, "shape_pt_lat": 47.001, "shape_pt_lon": 8.001, "shape_dist_traveled": 100.0},
        # Non-monotone value should be clamped to max
        {"shape_pt_sequence": 3, "shape_pt_lat": 47.002, "shape_pt_lon": 8.002, "shape_dist_traveled": 50.0},
        {"shape_pt_sequence": 4, "shape_pt_lat": 47.003, "shape_pt_lon": 8.003, "shape_dist_traveled": 200.0},
    ]
    pts = build_shape_points(rows)
    assert pts[0].user_distance == pytest.approx(0.0)
    assert pts[1].user_distance == pytest.approx(100.0)
    # Running max: 50 < 100, so stays 100
    assert pts[2].user_distance == pytest.approx(100.0)
    assert pts[3].user_distance == pytest.approx(200.0)


# ---------------------------------------------------------------------------
# resolve_stop_location
# ---------------------------------------------------------------------------


def test_resolve_stop_location_direct() -> None:
    stops = {
        "S1": {"stop_id": "S1", "stop_lat": 47.3, "stop_lon": 8.5, "stop_name": "Stop 1"},
    }
    lat, lon = resolve_stop_location("S1", stops)
    assert lat == pytest.approx(47.3)
    assert lon == pytest.approx(8.5)


def test_resolve_stop_location_via_parent() -> None:
    stops = {
        "S1": {"stop_id": "S1", "stop_lat": None, "stop_lon": None, "stop_name": "Stop 1", "parent_station": "P1"},
        "P1": {"stop_id": "P1", "stop_lat": 47.4, "stop_lon": 8.6, "stop_name": "Parent"},
    }
    lat, lon = resolve_stop_location("S1", stops)
    assert lat == pytest.approx(47.4)
    assert lon == pytest.approx(8.6)


def test_resolve_stop_location_max_hops() -> None:
    """After 3 parent hops with no coordinates, returns (0, 0)."""
    stops = {
        "S1": {"stop_id": "S1", "stop_lat": None, "stop_lon": None, "parent_station": "P1"},
        "P1": {"stop_id": "P1", "stop_lat": None, "stop_lon": None, "parent_station": "P2"},
        "P2": {"stop_id": "P2", "stop_lat": None, "stop_lon": None, "parent_station": "P3"},
        "P3": {"stop_id": "P3", "stop_lat": None, "stop_lon": None, "parent_station": "P4"},
        "P4": {"stop_id": "P4", "stop_lat": 47.0, "stop_lon": 8.0},
    }
    # 4 iterations: S1 → P1 → P2 → P3 → P4 not reached
    lat, lon = resolve_stop_location("S1", stops)
    assert lat == pytest.approx(0.0)
    assert lon == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# compute_trip_hash
# ---------------------------------------------------------------------------


def test_compute_trip_hash_identical_sequences() -> None:
    stop_times = [
        {"stop_id": "A", "stop_sequence": 1, "shape_dist_traveled": 0.0},
        {"stop_id": "B", "stop_sequence": 2, "shape_dist_traveled": 100.0},
    ]
    h1 = compute_trip_hash(stop_times)
    h2 = compute_trip_hash(stop_times)
    assert h1 == h2


def test_compute_trip_hash_differs_on_stop_id() -> None:
    st1 = [{"stop_id": "A", "stop_sequence": 1, "shape_dist_traveled": 0.0}]
    st2 = [{"stop_id": "B", "stop_sequence": 1, "shape_dist_traveled": 0.0}]
    assert compute_trip_hash(st1) != compute_trip_hash(st2)


def test_compute_trip_hash_differs_on_dist() -> None:
    st1 = [{"stop_id": "A", "stop_sequence": 1, "shape_dist_traveled": 0.0}]
    st2 = [{"stop_id": "A", "stop_sequence": 1, "shape_dist_traveled": 99.0}]
    assert compute_trip_hash(st1) != compute_trip_hash(st2)


# ---------------------------------------------------------------------------
# find_potential_matches
# ---------------------------------------------------------------------------


# Canonical test shape from spec
SHAPE_ROWS = [
    {"shape_pt_sequence": 0, "shape_pt_lat": 47.365873, "shape_pt_lon": 8.525414, "shape_dist_traveled": None},
    {"shape_pt_sequence": 1, "shape_pt_lat": 47.365872, "shape_pt_lon": 8.525376, "shape_dist_traveled": None},
    {"shape_pt_sequence": 2, "shape_pt_lat": 47.366099, "shape_pt_lon": 8.525154, "shape_dist_traveled": None},
    {"shape_pt_sequence": 3, "shape_pt_lat": 47.365925, "shape_pt_lon": 8.525735, "shape_dist_traveled": None},
    {"shape_pt_sequence": 4, "shape_pt_lat": 47.364108, "shape_pt_lon": 8.525715, "shape_dist_traveled": None},
]

SHAPE_POINTS = build_shape_points(SHAPE_ROWS)


def _make_stop(lat: float, lon: float) -> StopPoint:
    return StopPoint(
        lat=lat,
        lon=lon,
        user_distance=0.0,
        stop_time_row={"stop_id": "X", "stop_sequence": 0, "csv_row_number": 1},
        is_large_station=False,
    )


def test_find_potential_matches_returns_local_minima() -> None:
    """A stop near a shape that passes close to it multiple times returns multiple candidates.

    Uses a shape that has 3 separate passes near the stop with large detours in between,
    ensuring the run-based algorithm produces 3 distinct local minima.
    """
    # Shape that passes near (47.365, 8.525) three times (segments 0-1, 4-5, 8-9)
    # with large northward detours in between that are > 150m away.
    looping_shape_rows = [
        {"shape_pt_sequence": 0, "shape_pt_lat": 47.365000, "shape_pt_lon": 8.5240, "shape_dist_traveled": None},
        {"shape_pt_sequence": 1, "shape_pt_lat": 47.365000, "shape_pt_lon": 8.5260, "shape_dist_traveled": None},
        {"shape_pt_sequence": 2, "shape_pt_lat": 47.374000, "shape_pt_lon": 8.5270, "shape_dist_traveled": None},
        {"shape_pt_sequence": 3, "shape_pt_lat": 47.374000, "shape_pt_lon": 8.5230, "shape_dist_traveled": None},
        {"shape_pt_sequence": 4, "shape_pt_lat": 47.365000, "shape_pt_lon": 8.5230, "shape_dist_traveled": None},
        {"shape_pt_sequence": 5, "shape_pt_lat": 47.365000, "shape_pt_lon": 8.5260, "shape_dist_traveled": None},
        {"shape_pt_sequence": 6, "shape_pt_lat": 47.374000, "shape_pt_lon": 8.5260, "shape_dist_traveled": None},
        {"shape_pt_sequence": 7, "shape_pt_lat": 47.374000, "shape_pt_lon": 8.5220, "shape_dist_traveled": None},
        {"shape_pt_sequence": 8, "shape_pt_lat": 47.365000, "shape_pt_lon": 8.5220, "shape_dist_traveled": None},
        {"shape_pt_sequence": 9, "shape_pt_lat": 47.365000, "shape_pt_lon": 8.5260, "shape_dist_traveled": None},
    ]
    looping_points = build_shape_points(looping_shape_rows)
    stop = _make_stop(47.365000, 8.525000)
    matches = find_potential_matches(stop, looping_points, max_dist=150.0)
    assert len(matches) == 3, f"Expected 3 local minima, got {len(matches)}"


# ---------------------------------------------------------------------------
# find_best_assignment
# ---------------------------------------------------------------------------


def test_find_best_assignment_prefers_lower_cost() -> None:
    """DP should pick the assignment with minimum total distance."""
    # Two stops; each has two candidates at different shape positions
    c0a = CandidateMatch(geo_distance_to_shape=5.0, shape_geo_distance=10.0, lat=47.0, lon=8.0)
    c0b = CandidateMatch(geo_distance_to_shape=50.0, shape_geo_distance=5.0, lat=47.0, lon=8.1)
    c1a = CandidateMatch(geo_distance_to_shape=5.0, shape_geo_distance=20.0, lat=47.1, lon=8.0)
    c1b = CandidateMatch(geo_distance_to_shape=50.0, shape_geo_distance=15.0, lat=47.1, lon=8.1)

    # stop 0 prefers c0a (5 m), stop 1 prefers c1a (5 m)
    # c0a.shape_geo_distance (10) <= c1a.shape_geo_distance (20) → valid assignment

    stops = [
        StopPoint(lat=47.0, lon=8.0, user_distance=0.0,
                  stop_time_row={"stop_id": "A", "stop_sequence": 0, "csv_row_number": 1},
                  is_large_station=False),
        StopPoint(lat=47.1, lon=8.0, user_distance=0.0,
                  stop_time_row={"stop_id": "B", "stop_sequence": 1, "csv_row_number": 2},
                  is_large_station=False),
    ]

    assignment, out_of_order = find_best_assignment([[c0a, c0b], [c1a, c1b]], stops)
    assert len(assignment) == 2
    assert assignment[0] is c0a
    assert assignment[1] is c1a
    assert len(out_of_order) == 0


def test_find_best_assignment_out_of_order() -> None:
    """When no non-decreasing assignment is possible, emit out_of_order problems."""
    # Stop 0 can only match at shape_geo_distance=100
    # Stop 1 can only match at shape_geo_distance=50 (which is < 100 → out of order)
    c0 = CandidateMatch(geo_distance_to_shape=5.0, shape_geo_distance=100.0, lat=47.0, lon=8.0)
    c1 = CandidateMatch(geo_distance_to_shape=5.0, shape_geo_distance=50.0, lat=47.1, lon=8.0)

    stops = [
        StopPoint(lat=47.0, lon=8.0, user_distance=0.0,
                  stop_time_row={"stop_id": "A", "stop_sequence": 0, "csv_row_number": 1},
                  is_large_station=False),
        StopPoint(lat=47.1, lon=8.0, user_distance=0.0,
                  stop_time_row={"stop_id": "B", "stop_sequence": 1, "csv_row_number": 2},
                  is_large_station=False),
    ]

    assignment, out_of_order = find_best_assignment([[c0], [c1]], stops)
    assert len(out_of_order) == 1
    stop_idx, match1, match2 = out_of_order[0]
    assert stop_idx == 1


def test_find_best_assignment_fast_path_single_candidate_per_stop() -> None:
    """Fast path should preserve in-order single-candidate assignments."""
    c0 = CandidateMatch(geo_distance_to_shape=5.0, shape_geo_distance=10.0, lat=47.0, lon=8.0)
    c1 = CandidateMatch(geo_distance_to_shape=6.0, shape_geo_distance=20.0, lat=47.1, lon=8.1)
    stops = [
        StopPoint(
            lat=47.0,
            lon=8.0,
            user_distance=0.0,
            stop_time_row={"stop_id": "A", "stop_sequence": 0, "csv_row_number": 1},
            is_large_station=False,
        ),
        StopPoint(
            lat=47.1,
            lon=8.1,
            user_distance=0.0,
            stop_time_row={"stop_id": "B", "stop_sequence": 1, "csv_row_number": 2},
            is_large_station=False,
        ),
    ]

    assignment, out_of_order = find_best_assignment([[c0], [c1]], stops)
    assert assignment == [c0, c1]
    assert out_of_order == []


def test_find_best_assignment_fast_path_handles_empty_candidate_list() -> None:
    """Fast path should keep assignment slots aligned when a stop has no candidates."""
    c0 = CandidateMatch(geo_distance_to_shape=5.0, shape_geo_distance=100.0, lat=47.0, lon=8.0)
    c2 = CandidateMatch(geo_distance_to_shape=6.0, shape_geo_distance=50.0, lat=47.2, lon=8.2)
    stops = [
        StopPoint(
            lat=47.0,
            lon=8.0,
            user_distance=0.0,
            stop_time_row={"stop_id": "A", "stop_sequence": 0, "csv_row_number": 1},
            is_large_station=False,
        ),
        StopPoint(
            lat=47.1,
            lon=8.1,
            user_distance=0.0,
            stop_time_row={"stop_id": "B", "stop_sequence": 1, "csv_row_number": 2},
            is_large_station=False,
        ),
        StopPoint(
            lat=47.2,
            lon=8.2,
            user_distance=0.0,
            stop_time_row={"stop_id": "C", "stop_sequence": 2, "csv_row_number": 3},
            is_large_station=False,
        ),
    ]

    assignment, out_of_order = find_best_assignment([[c0], [], [c2]], stops)
    assert assignment == [c0, None, c2]
    assert len(out_of_order) == 1
    stop_idx, match1, match2 = out_of_order[0]
    assert stop_idx == 2
    assert match1 is c2
    assert match2 is c0


# ---------------------------------------------------------------------------
# Vectorised geometry cross-checks
# ---------------------------------------------------------------------------

import numpy as np
from gtfs_validator.validators.shape_to_stop_matching_util import (
    _cross,
    _dot,
    _norm,
    _normalize,
    _seg_fraction,
)


def test_haversine_meters_vec_matches_scalar() -> None:
    """Vectorised haversine produces same results as scalar version."""
    pairs = [
        (47.3769, 8.5417, 46.2044, 6.1432),
        (47.0, 8.0, 47.0, 8.0),
        (40.7128, -74.006, 51.5074, -0.1278),
        (0.0, 0.0, 0.0, 1.0),
    ]
    for lat1, lon1, lat2, lon2 in pairs:
        scalar = geo_distance_meters(lat1, lon1, lat2, lon2)
        vec = float(_haversine_meters_vec(
            np.array([lat1]), np.array([lon1]),
            np.array([lat2]), np.array([lon2]),
        )[0])
        assert abs(scalar - vec) < 0.01, f"Mismatch for ({lat1},{lon1})->({lat2},{lon2}): {scalar} vs {vec}"


def test_closest_points_on_edges_vec_matches_scalar() -> None:
    """Vectorised closest_point_on_edge matches scalar for several cases."""
    # Arc on equator (0,0)-(0,1), stop just north of midpoint
    av = latlng_to_unit_vector(0.0, 0.0)
    bv = latlng_to_unit_vector(0.0, 1.0)
    pv = latlng_to_unit_vector(0.001, 0.5)

    scalar_lat, scalar_lon = closest_point_on_edge(pv, av, bv)

    a_arr = np.array([av])
    b_arr = np.array([bv])
    p_arr = np.array(pv)
    result = _closest_points_on_edges_vec(p_arr, a_arr, b_arr)
    vec_lats, vec_lons = _uv_to_latlng_vec(result)

    assert abs(scalar_lat - vec_lats[0]) < 1e-6
    assert abs(scalar_lon - vec_lons[0]) < 1e-6


def test_seg_fraction_vec_matches_scalar() -> None:
    """Vectorised seg_fraction matches scalar for several cases."""
    av = latlng_to_unit_vector(0.0, 0.0)
    bv = latlng_to_unit_vector(0.0, 1.0)

    test_points = [
        latlng_to_unit_vector(0.001, 0.5),   # midpoint
        latlng_to_unit_vector(0.0, 2.0),      # beyond b
        latlng_to_unit_vector(0.0, -1.0),     # before a
        latlng_to_unit_vector(0.001, 0.25),   # quarter
    ]

    for pv in test_points:
        scalar = _seg_fraction(pv, av, bv)
        vec = float(_seg_fraction_vec(
            np.array(pv), np.array([av]), np.array([bv]),
        )[0])
        assert abs(scalar - vec) < 1e-6, f"Mismatch: scalar={scalar}, vec={vec}"


def test_find_potential_matches_vec_matches_scalar() -> None:
    """Vectorised find_potential_matches returns same candidates as scalar."""
    shape_points = build_shape_points(SHAPE_ROWS)
    shape_arrays = build_shape_arrays(shape_points)
    stop = _make_stop(47.3659, 8.5254)

    scalar_matches = find_potential_matches(stop, shape_points, 150.0)
    vec_matches = find_potential_matches(stop, shape_points, 150.0, shape_arrays)

    assert len(scalar_matches) == len(vec_matches), (
        f"Count mismatch: scalar={len(scalar_matches)}, vec={len(vec_matches)}"
    )
    for sm, vm in zip(scalar_matches, vec_matches):
        assert abs(sm.geo_distance_to_shape - vm.geo_distance_to_shape) < 0.5
        assert abs(sm.shape_geo_distance - vm.shape_geo_distance) < 0.5


def test_find_closest_on_shape_vec_matches_scalar() -> None:
    """Vectorised find_closest_on_shape returns same result as scalar."""
    shape_points = build_shape_points(SHAPE_ROWS)
    shape_arrays = build_shape_arrays(shape_points)
    # A stop that's far from the shape
    stop = _make_stop(47.370, 8.530)

    scalar = find_closest_on_shape(stop, shape_points)
    vec = find_closest_on_shape(stop, shape_points, shape_arrays)

    assert abs(scalar.geo_distance_to_shape - vec.geo_distance_to_shape) < 0.5
    assert abs(scalar.shape_geo_distance - vec.shape_geo_distance) < 0.5
    assert abs(scalar.lat - vec.lat) < 1e-4
    assert abs(scalar.lon - vec.lon) < 1e-4


def test_find_potential_matches_spatial_index_matches_scalar() -> None:
    """Spatial-indexed scalar path should match plain scalar candidates."""
    shape_points = build_shape_points(SHAPE_ROWS * 30)
    spatial_index = build_shape_spatial_index(shape_points)
    assert spatial_index is not None
    stop = _make_stop(47.3659, 8.5254)

    scalar_matches = find_potential_matches(stop, shape_points, 150.0)
    indexed_matches = find_potential_matches(
        stop,
        shape_points,
        150.0,
        spatial_index=spatial_index,
    )

    assert len(scalar_matches) == len(indexed_matches)
    for sm, im in zip(scalar_matches, indexed_matches):
        assert abs(sm.geo_distance_to_shape - im.geo_distance_to_shape) < 0.5
        assert abs(sm.shape_geo_distance - im.shape_geo_distance) < 0.5


def test_find_potential_matches_looping_shape_vec() -> None:
    """Vectorised path also finds 3 local minima on the looping shape."""
    looping_shape_rows = [
        {"shape_pt_sequence": 0, "shape_pt_lat": 47.365000, "shape_pt_lon": 8.5240, "shape_dist_traveled": None},
        {"shape_pt_sequence": 1, "shape_pt_lat": 47.365000, "shape_pt_lon": 8.5260, "shape_dist_traveled": None},
        {"shape_pt_sequence": 2, "shape_pt_lat": 47.374000, "shape_pt_lon": 8.5270, "shape_dist_traveled": None},
        {"shape_pt_sequence": 3, "shape_pt_lat": 47.374000, "shape_pt_lon": 8.5230, "shape_dist_traveled": None},
        {"shape_pt_sequence": 4, "shape_pt_lat": 47.365000, "shape_pt_lon": 8.5230, "shape_dist_traveled": None},
        {"shape_pt_sequence": 5, "shape_pt_lat": 47.365000, "shape_pt_lon": 8.5260, "shape_dist_traveled": None},
        {"shape_pt_sequence": 6, "shape_pt_lat": 47.374000, "shape_pt_lon": 8.5260, "shape_dist_traveled": None},
        {"shape_pt_sequence": 7, "shape_pt_lat": 47.374000, "shape_pt_lon": 8.5220, "shape_dist_traveled": None},
        {"shape_pt_sequence": 8, "shape_pt_lat": 47.365000, "shape_pt_lon": 8.5220, "shape_dist_traveled": None},
        {"shape_pt_sequence": 9, "shape_pt_lat": 47.365000, "shape_pt_lon": 8.5260, "shape_dist_traveled": None},
    ]
    looping_points = build_shape_points(looping_shape_rows)
    looping_arrays = build_shape_arrays(looping_points)
    stop = _make_stop(47.365000, 8.525000)
    matches = find_potential_matches(stop, looping_points, 150.0, looping_arrays)
    assert len(matches) == 3, f"Expected 3 local minima, got {len(matches)}"
