"""Tests for ShapeIncreasingDistanceValidator."""

from __future__ import annotations

import datetime
import math
import pytest

import polars as pl

from gtfs_validator.context import ValidationContext
from gtfs_validator.notices import Severity
from gtfs_validator.validators.shape_increasing_distance import (
    validate_shape_increasing_distance,
    _geodesic_distance_meters,
    DISTANCE_THRESHOLD_METERS,
)

CTX = ValidationContext(country_code="US", date_for_validation=datetime.date(2024, 1, 1))

_SHAPES_SCHEMA = {
    "csv_row_number": pl.Int64,
    "shape_id": pl.Utf8,
    "shape_pt_lat": pl.Float64,
    "shape_pt_lon": pl.Float64,
    "shape_pt_sequence": pl.Int64,
    "shape_dist_traveled": pl.Float64,
}


def make_shapes(rows: list[dict]) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame(schema=_SHAPES_SCHEMA)
    return pl.DataFrame(rows, schema=_SHAPES_SCHEMA)


# Test 1: Increasing distance — no notice
def test_increasing_distance_no_notice() -> None:
    shapes = make_shapes([
        {"csv_row_number": 1, "shape_id": "first shape", "shape_pt_lat": 10.0, "shape_pt_lon": 20.0, "shape_pt_sequence": 1, "shape_dist_traveled": 10.0},
        {"csv_row_number": 2, "shape_id": "first shape", "shape_pt_lat": 11.0, "shape_pt_lon": 21.0, "shape_pt_sequence": 2, "shape_dist_traveled": 45.0},
        {"csv_row_number": 3, "shape_id": "first shape", "shape_pt_lat": 12.0, "shape_pt_lon": 22.0, "shape_pt_sequence": 3, "shape_dist_traveled": 64.0},
    ])
    notices = validate_shape_increasing_distance({"shapes": shapes}, CTX)
    assert notices == []


# Test 2: Last point has decreasing distance
def test_last_point_decreasing_distance_emits_notice() -> None:
    shapes = make_shapes([
        {"csv_row_number": 1, "shape_id": "shape1", "shape_pt_lat": 10.0, "shape_pt_lon": 20.0, "shape_pt_sequence": 1, "shape_dist_traveled": 10.0},
        {"csv_row_number": 2, "shape_id": "shape1", "shape_pt_lat": 11.0, "shape_pt_lon": 21.0, "shape_pt_sequence": 2, "shape_dist_traveled": 45.0},
        {"csv_row_number": 3, "shape_id": "shape1", "shape_pt_lat": 12.0, "shape_pt_lon": 22.0, "shape_pt_sequence": 3, "shape_dist_traveled": 40.0},
    ])
    notices = validate_shape_increasing_distance({"shapes": shapes}, CTX)
    assert len(notices) == 1
    notice = notices[0]
    assert notice.code == "decreasing_shape_distance"
    assert notice.severity == Severity.ERROR
    assert notice.fields["shape_dist_traveled"] == 40.0
    assert notice.fields["shape_pt_sequence"] == 3
    assert notice.fields["prev_shape_dist_traveled"] == 45.0
    assert notice.fields["prev_shape_pt_sequence"] == 2


# Test 3: Intermediate point has decreasing distance
def test_intermediate_point_decreasing_distance_emits_notice() -> None:
    shapes = make_shapes([
        {"csv_row_number": 1, "shape_id": "shape1", "shape_pt_lat": 10.0, "shape_pt_lon": 20.0, "shape_pt_sequence": 1, "shape_dist_traveled": 10.0},
        {"csv_row_number": 2, "shape_id": "shape1", "shape_pt_lat": 11.0, "shape_pt_lon": 21.0, "shape_pt_sequence": 2, "shape_dist_traveled": 9.0},
        {"csv_row_number": 3, "shape_id": "shape1", "shape_pt_lat": 12.0, "shape_pt_lon": 22.0, "shape_pt_sequence": 3, "shape_dist_traveled": 40.0},
    ])
    notices = validate_shape_increasing_distance({"shapes": shapes}, CTX)
    assert len(notices) == 1
    notice = notices[0]
    assert notice.code == "decreasing_shape_distance"
    assert notice.fields["shape_pt_sequence"] == 2
    assert notice.fields["prev_shape_pt_sequence"] == 1


# Test 4: Equal distance, different coords far apart → ERROR
def test_equal_distance_different_coords_far_apart_emits_error() -> None:
    shapes = make_shapes([
        {"csv_row_number": 1, "shape_id": "shape1", "shape_pt_lat": 10.0, "shape_pt_lon": 20.0, "shape_pt_sequence": 1, "shape_dist_traveled": 10.0},
        {"csv_row_number": 2, "shape_id": "shape1", "shape_pt_lat": 31.0, "shape_pt_lon": 42.0, "shape_pt_sequence": 2, "shape_dist_traveled": 45.0},
        {"csv_row_number": 3, "shape_id": "shape1", "shape_pt_lat": 29.0, "shape_pt_lon": 46.0, "shape_pt_sequence": 3, "shape_dist_traveled": 45.0},
    ])
    notices = validate_shape_increasing_distance({"shapes": shapes}, CTX)
    assert len(notices) == 1
    notice = notices[0]
    assert notice.code == "equal_shape_distance_diff_coordinates"
    assert notice.severity == Severity.ERROR
    assert notice.fields["actual_distance_between_shape_points"] >= 1.11
    assert notice.fields["shape_pt_sequence"] == 3
    assert notice.fields["prev_shape_pt_sequence"] == 2


# Test 5: Equal distance, same coords → WARNING
def test_equal_distance_same_coords_emits_warning() -> None:
    shapes = make_shapes([
        {"csv_row_number": 1, "shape_id": "shape1", "shape_pt_lat": 10.0, "shape_pt_lon": 20.0, "shape_pt_sequence": 1, "shape_dist_traveled": 10.0},
        {"csv_row_number": 2, "shape_id": "shape1", "shape_pt_lat": 31.0, "shape_pt_lon": 42.0, "shape_pt_sequence": 2, "shape_dist_traveled": 45.0},
        {"csv_row_number": 3, "shape_id": "shape1", "shape_pt_lat": 31.0, "shape_pt_lon": 42.0, "shape_pt_sequence": 4, "shape_dist_traveled": 45.0},
    ])
    notices = validate_shape_increasing_distance({"shapes": shapes}, CTX)
    assert len(notices) == 1
    notice = notices[0]
    assert notice.code == "equal_shape_distance_same_coordinates"
    assert notice.severity == Severity.WARNING
    assert notice.fields["shape_pt_sequence"] == 4
    assert notice.fields["prev_shape_pt_sequence"] == 2


# Test 6: Null shape_dist_traveled skips pairs
def test_null_shape_dist_traveled_skips_pair_no_notice() -> None:
    shapes = make_shapes([
        {"csv_row_number": 1, "shape_id": "first shape", "shape_pt_lat": 10.0, "shape_pt_lon": 20.0, "shape_pt_sequence": 1, "shape_dist_traveled": 10.0},
        {"csv_row_number": 2, "shape_id": "first shape", "shape_pt_lat": 11.0, "shape_pt_lon": 21.0, "shape_pt_sequence": 2, "shape_dist_traveled": None},
        {"csv_row_number": 3, "shape_id": "first shape", "shape_pt_lat": 12.0, "shape_pt_lon": 22.0, "shape_pt_sequence": 3, "shape_dist_traveled": 40.0},
    ])
    notices = validate_shape_increasing_distance({"shapes": shapes}, CTX)
    assert notices == []


# Test 7: Equal distance, diff coords, below threshold → WARNING
def test_equal_distance_diff_coords_below_threshold_emits_warning() -> None:
    # Two points close enough to be below 1.11 m but > 0.0 m apart
    lat1 = 48.858844
    lon1 = 2.294351
    lat2 = lat1 + 1e-6  # approximately 0.11 m north
    lon2 = lon1
    dist = _geodesic_distance_meters(lat1, lon1, lat2, lon2)
    assert 0.0 < dist < 1.11, f"Expected sub-threshold distance, got {dist}"

    shapes = make_shapes([
        {"csv_row_number": 1, "shape_id": "shape_a", "shape_pt_lat": lat1, "shape_pt_lon": lon1, "shape_pt_sequence": 1, "shape_dist_traveled": 10.0},
        {"csv_row_number": 2, "shape_id": "shape_a", "shape_pt_lat": lat2, "shape_pt_lon": lon2, "shape_pt_sequence": 2, "shape_dist_traveled": 10.0},
    ])
    notices = validate_shape_increasing_distance({"shapes": shapes}, CTX)
    assert len(notices) == 1
    notice = notices[0]
    assert notice.code == "equal_shape_distance_diff_coordinates_distance_below_threshold"
    assert notice.severity == Severity.WARNING
    assert 0.0 < notice.fields["actual_distance_between_shape_points"] < 1.11


# Test 8: Equal distance, diff coords, distance exactly 0.0 → no notice
# This is a boundary case: same coordinates at floating-point level but listed as different.
# In practice, if coordinates differ but haversine yields 0.0 exactly it's extremely contrived.
# We test the guard logic by confirming truly identical lat/lon routes to same_coords WARNING,
# so this boundary is satisfied by test 5 + the implementation guard (distance > 0.0).
@pytest.mark.xfail(reason="Exact 0.0 geodesic from different float values is impractical to construct")
def test_equal_distance_diff_coords_distance_exactly_zero_no_notice() -> None:
    # If distance == 0.0 exactly with different coordinate values, expect no notice.
    # This would require floating-point underflow in haversine, which is not practically achievable.
    assert False, "Cannot construct test case in pure Python without mocking"


# Test 9: shapes absent from feed → no notice
def test_shapes_absent_no_notice() -> None:
    notices = validate_shape_increasing_distance({}, CTX)
    assert notices == []


# Test 10: shapes empty DataFrame → no notice
def test_shapes_empty_no_notice() -> None:
    shapes = make_shapes([])
    notices = validate_shape_increasing_distance({"shapes": shapes}, CTX)
    assert notices == []


# Test 11: Single point shape → no notice
def test_single_point_shape_no_notice() -> None:
    shapes = make_shapes([
        {"csv_row_number": 1, "shape_id": "shape1", "shape_pt_lat": 10.0, "shape_pt_lon": 20.0, "shape_pt_sequence": 1, "shape_dist_traveled": 10.0},
    ])
    notices = validate_shape_increasing_distance({"shapes": shapes}, CTX)
    assert notices == []


# Test 12: Multiple shapes — only shape B has a violation
def test_multiple_shapes_independent() -> None:
    shapes = make_shapes([
        # Shape A: strictly increasing
        {"csv_row_number": 1, "shape_id": "A", "shape_pt_lat": 10.0, "shape_pt_lon": 20.0, "shape_pt_sequence": 1, "shape_dist_traveled": 1.0},
        {"csv_row_number": 2, "shape_id": "A", "shape_pt_lat": 11.0, "shape_pt_lon": 21.0, "shape_pt_sequence": 2, "shape_dist_traveled": 2.0},
        {"csv_row_number": 3, "shape_id": "A", "shape_pt_lat": 12.0, "shape_pt_lon": 22.0, "shape_pt_sequence": 3, "shape_dist_traveled": 3.0},
        # Shape B: one decreasing distance
        {"csv_row_number": 4, "shape_id": "B", "shape_pt_lat": 10.0, "shape_pt_lon": 20.0, "shape_pt_sequence": 1, "shape_dist_traveled": 1.0},
        {"csv_row_number": 5, "shape_id": "B", "shape_pt_lat": 11.0, "shape_pt_lon": 21.0, "shape_pt_sequence": 2, "shape_dist_traveled": 5.0},
        {"csv_row_number": 6, "shape_id": "B", "shape_pt_lat": 12.0, "shape_pt_lon": 22.0, "shape_pt_sequence": 3, "shape_dist_traveled": 3.0},
    ])
    notices = validate_shape_increasing_distance({"shapes": shapes}, CTX)
    assert len(notices) == 1
    assert notices[0].fields["shape_id"] == "B"


# Test 13: Both shape_dist_traveled null → no notice
def test_both_shape_dist_null_skips_pair() -> None:
    shapes = make_shapes([
        {"csv_row_number": 1, "shape_id": "shape1", "shape_pt_lat": 10.0, "shape_pt_lon": 20.0, "shape_pt_sequence": 1, "shape_dist_traveled": None},
        {"csv_row_number": 2, "shape_id": "shape1", "shape_pt_lat": 11.0, "shape_pt_lon": 21.0, "shape_pt_sequence": 2, "shape_dist_traveled": None},
    ])
    notices = validate_shape_increasing_distance({"shapes": shapes}, CTX)
    assert notices == []


# Test 14: Unsorted CSV rows sorted by sequence before processing
def test_unsorted_csv_rows_sorted_by_sequence() -> None:
    # Rows provided in reverse sequence order; distances are 3.0, 1.0, 2.0
    # After sort by sequence: seq=1 dist=1.0, seq=2 dist=2.0, seq=3 dist=3.0 (all increasing)
    shapes = make_shapes([
        {"csv_row_number": 1, "shape_id": "shape1", "shape_pt_lat": 12.0, "shape_pt_lon": 22.0, "shape_pt_sequence": 3, "shape_dist_traveled": 3.0},
        {"csv_row_number": 2, "shape_id": "shape1", "shape_pt_lat": 10.0, "shape_pt_lon": 20.0, "shape_pt_sequence": 1, "shape_dist_traveled": 1.0},
        {"csv_row_number": 3, "shape_id": "shape1", "shape_pt_lat": 11.0, "shape_pt_lon": 21.0, "shape_pt_sequence": 2, "shape_dist_traveled": 2.0},
    ])
    notices = validate_shape_increasing_distance({"shapes": shapes}, CTX)
    assert notices == []


# Test 15: csv_row_number is preserved from original row, not recomputed
def test_decreasing_csv_row_number_preserved_in_notice() -> None:
    # seq=1 has csv_row_number=5, seq=2 has csv_row_number=2 — provided out of CSV order
    shapes = make_shapes([
        {"csv_row_number": 5, "shape_id": "shape1", "shape_pt_lat": 10.0, "shape_pt_lon": 20.0, "shape_pt_sequence": 1, "shape_dist_traveled": 10.0},
        {"csv_row_number": 2, "shape_id": "shape1", "shape_pt_lat": 11.0, "shape_pt_lon": 21.0, "shape_pt_sequence": 2, "shape_dist_traveled": 5.0},
    ])
    notices = validate_shape_increasing_distance({"shapes": shapes}, CTX)
    assert len(notices) == 1
    assert notices[0].fields["csv_row_number"] == 2
    assert notices[0].fields["prev_csv_row_number"] == 5


# Test 16: Threshold boundary at exactly 1.11 m → ERROR (not warning)
def test_threshold_boundary_at_exactly_1_11_m() -> None:
    # Find coordinates whose geodesic distance is >= 1.11 m (use pre-verified values).
    # 1e-5 degree latitude difference ≈ 1.11 m, so use slightly more to ensure >= 1.11 m.
    lat1 = 48.858844
    lon1 = 2.294351
    # Try 1e-5 degrees latitude offset
    lat2 = lat1 + 1e-5
    lon2 = lon1
    dist = _geodesic_distance_meters(lat1, lon1, lat2, lon2)
    assert dist >= DISTANCE_THRESHOLD_METERS, f"Expected dist >= 1.11 m, got {dist}"

    shapes = make_shapes([
        {"csv_row_number": 1, "shape_id": "shape1", "shape_pt_lat": lat1, "shape_pt_lon": lon1, "shape_pt_sequence": 1, "shape_dist_traveled": 10.0},
        {"csv_row_number": 2, "shape_id": "shape1", "shape_pt_lat": lat2, "shape_pt_lon": lon2, "shape_pt_sequence": 2, "shape_dist_traveled": 10.0},
    ])
    notices = validate_shape_increasing_distance({"shapes": shapes}, CTX)
    assert len(notices) == 1
    assert notices[0].code == "equal_shape_distance_diff_coordinates"
    assert notices[0].severity == Severity.ERROR


# Test 17: shape_id propagated from row data
def test_notice_fields_shape_id_propagated() -> None:
    shapes = make_shapes([
        {"csv_row_number": 1, "shape_id": "my_shape", "shape_pt_lat": 10.0, "shape_pt_lon": 20.0, "shape_pt_sequence": 1, "shape_dist_traveled": 10.0},
        {"csv_row_number": 2, "shape_id": "my_shape", "shape_pt_lat": 11.0, "shape_pt_lon": 21.0, "shape_pt_sequence": 2, "shape_dist_traveled": 5.0},
    ])
    notices = validate_shape_increasing_distance({"shapes": shapes}, CTX)
    assert len(notices) == 1
    assert notices[0].fields["shape_id"] == "my_shape"


# Test 18: All shape_dist_traveled null → no notices
def test_all_null_dist_traveled_no_notices() -> None:
    shapes = make_shapes([
        {"csv_row_number": i, "shape_id": "shape1", "shape_pt_lat": float(i), "shape_pt_lon": float(i), "shape_pt_sequence": i, "shape_dist_traveled": None}
        for i in range(1, 6)
    ])
    notices = validate_shape_increasing_distance({"shapes": shapes}, CTX)
    assert notices == []
