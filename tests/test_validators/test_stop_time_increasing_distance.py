"""Tests for StopTimeIncreasingDistanceValidator."""

from __future__ import annotations

import datetime

import polars as pl
import pytest

from gtfs_validator.context import ValidationContext
from gtfs_validator.notices import Severity
from gtfs_validator.validators.stop_time_increasing_distance import (
    validate_stop_time_increasing_distance,
)

CTX = ValidationContext(country_code="US", date_for_validation=datetime.date(2024, 1, 1))

_STOP_TIMES_SCHEMA = {
    "csv_row_number": pl.Int64,
    "trip_id": pl.Utf8,
    "stop_id": pl.Utf8,
    "stop_sequence": pl.Int64,
    "shape_dist_traveled": pl.Float64,
}


def make_feed(rows: list[dict]) -> dict[str, pl.DataFrame]:
    if not rows:
        return {"stop_times": pl.DataFrame(schema=_STOP_TIMES_SCHEMA)}
    return {"stop_times": pl.DataFrame(rows, schema=_STOP_TIMES_SCHEMA)}


# ---------------------------------------------------------------------------
# Tests derived from Java contracts
# ---------------------------------------------------------------------------

def test_increasing_distance_no_notice() -> None:
    """Covers: increasingDistanceAlongShapeShouldNotGenerateNotice."""
    feed = make_feed([
        {"csv_row_number": 1, "trip_id": "first trip", "stop_id": "s0", "stop_sequence": 2, "shape_dist_traveled": 10.0},
        {"csv_row_number": 2, "trip_id": "first trip", "stop_id": "s1", "stop_sequence": 42, "shape_dist_traveled": 45.0},
        {"csv_row_number": 3, "trip_id": "first trip", "stop_id": "s2", "stop_sequence": 46, "shape_dist_traveled": 64.0},
    ])
    notices = validate_stop_time_increasing_distance(feed, CTX)
    assert notices == []


def test_last_stop_decreasing_distance_emits_notice() -> None:
    """Covers: lastShapeWithDecreasingDistanceAlongShapeShouldGenerateNotice."""
    feed = make_feed([
        {"csv_row_number": 1, "trip_id": "first trip", "stop_id": "s0", "stop_sequence": 2, "shape_dist_traveled": 10.0},
        {"csv_row_number": 2, "trip_id": "first trip", "stop_id": "s1", "stop_sequence": 42, "shape_dist_traveled": 45.0},
        {"csv_row_number": 3, "trip_id": "first trip", "stop_id": "s2", "stop_sequence": 46, "shape_dist_traveled": 4.0},
    ])
    notices = validate_stop_time_increasing_distance(feed, CTX)
    assert len(notices) == 1
    n = notices[0]
    assert n.code == "decreasing_or_equal_stop_time_distance"
    assert n.severity == Severity.ERROR
    assert n.fields["trip_id"] == "first trip"
    assert n.fields["stop_id"] == "s2"
    assert n.fields["csv_row_number"] == 3
    assert n.fields["shape_dist_traveled"] == 4.0
    assert n.fields["stop_sequence"] == 46
    assert n.fields["prev_csv_row_number"] == 2
    assert n.fields["prev_shape_dist_traveled"] == 45.0
    assert n.fields["prev_stop_sequence"] == 42


def test_equal_distance_emits_notice() -> None:
    """Covers: twoShapesWithTheSameDistanceShouldGenerateNotice."""
    feed = make_feed([
        {"csv_row_number": 1, "trip_id": "first trip", "stop_id": "s0", "stop_sequence": 2, "shape_dist_traveled": 10.0},
        {"csv_row_number": 2, "trip_id": "first trip", "stop_id": "s1", "stop_sequence": 42, "shape_dist_traveled": 45.0},
        {"csv_row_number": 3, "trip_id": "first trip", "stop_id": "s2", "stop_sequence": 46, "shape_dist_traveled": 45.0},
    ])
    notices = validate_stop_time_increasing_distance(feed, CTX)
    assert len(notices) == 1
    n = notices[0]
    assert n.fields["shape_dist_traveled"] == 45.0
    assert n.fields["prev_shape_dist_traveled"] == 45.0
    assert n.fields["stop_sequence"] == 46
    assert n.fields["prev_stop_sequence"] == 42


def test_intermediate_decreasing_distance_emits_notice() -> None:
    """Covers: oneIntermediateShapeWithDecreasingDistanceAlongShapeShouldGenerateNotice."""
    feed = make_feed([
        {"csv_row_number": 1, "trip_id": "first trip", "stop_id": "s0", "stop_sequence": 2, "shape_dist_traveled": 10.0},
        {"csv_row_number": 2, "trip_id": "first trip", "stop_id": "s1", "stop_sequence": 42, "shape_dist_traveled": 8.6},
        {"csv_row_number": 3, "trip_id": "first trip", "stop_id": "s2", "stop_sequence": 46, "shape_dist_traveled": 46.0},
    ])
    notices = validate_stop_time_increasing_distance(feed, CTX)
    assert len(notices) == 1
    n = notices[0]
    assert n.fields["stop_id"] == "s1"
    assert n.fields["csv_row_number"] == 2
    assert n.fields["shape_dist_traveled"] == 8.6
    assert n.fields["stop_sequence"] == 42
    assert n.fields["prev_csv_row_number"] == 1
    assert n.fields["prev_shape_dist_traveled"] == 10.0
    assert n.fields["prev_stop_sequence"] == 2


def test_null_stop_id_row_skipped_no_notice() -> None:
    """Covers: oneIntermediateShapeWithoutStopShouldBeIgnored."""
    feed = make_feed([
        {"csv_row_number": 1, "trip_id": "first trip", "stop_id": "s0", "stop_sequence": 1, "shape_dist_traveled": 0.0},
        {"csv_row_number": 2, "trip_id": "first trip", "stop_id": "s1", "stop_sequence": 2, "shape_dist_traveled": 45.0},
        {"csv_row_number": 3, "trip_id": "first trip", "stop_id": None, "stop_sequence": 3, "shape_dist_traveled": 4.0},
        {"csv_row_number": 4, "trip_id": "first trip", "stop_id": "s3", "stop_sequence": 4, "shape_dist_traveled": 64.0},
    ])
    notices = validate_stop_time_increasing_distance(feed, CTX)
    assert notices == []


def test_null_stop_id_does_not_reset_prev_emits_notice() -> None:
    """Covers: oneIntermediateShapeWithoutStopAndPreviousDecreasingDistanceShouldGenerateNotice."""
    feed = make_feed([
        {"csv_row_number": 1, "trip_id": "first trip", "stop_id": "s0", "stop_sequence": 1, "shape_dist_traveled": 0.0},
        {"csv_row_number": 2, "trip_id": "first trip", "stop_id": "s1", "stop_sequence": 2, "shape_dist_traveled": 85.0},
        {"csv_row_number": 3, "trip_id": "first trip", "stop_id": None, "stop_sequence": 3, "shape_dist_traveled": 444.0},
        {"csv_row_number": 4, "trip_id": "first trip", "stop_id": "s3", "stop_sequence": 4, "shape_dist_traveled": 64.0},
    ])
    notices = validate_stop_time_increasing_distance(feed, CTX)
    assert len(notices) == 1
    n = notices[0]
    assert n.fields["stop_id"] == "s3"
    assert n.fields["csv_row_number"] == 4
    assert n.fields["shape_dist_traveled"] == 64.0
    assert n.fields["stop_sequence"] == 4
    assert n.fields["prev_csv_row_number"] == 2
    assert n.fields["prev_shape_dist_traveled"] == 85.0
    assert n.fields["prev_stop_sequence"] == 2


def test_multiple_null_stop_id_rows_skipped_emits_notice() -> None:
    """Covers: multipleIntermediateShapeWithoutStopAndPreviousDecreasingDistanceShouldGenerateNotice."""
    feed = make_feed([
        {"csv_row_number": 1, "trip_id": "first trip", "stop_id": "s0", "stop_sequence": 1, "shape_dist_traveled": 0.0},
        {"csv_row_number": 2, "trip_id": "first trip", "stop_id": "s1", "stop_sequence": 2, "shape_dist_traveled": 85.0},
        {"csv_row_number": 3, "trip_id": "first trip", "stop_id": None, "stop_sequence": 3, "shape_dist_traveled": 114.0},
        {"csv_row_number": 4, "trip_id": "first trip", "stop_id": None, "stop_sequence": 4, "shape_dist_traveled": 444.0},
        {"csv_row_number": 5, "trip_id": "first trip", "stop_id": "s3", "stop_sequence": 5, "shape_dist_traveled": 64.0},
    ])
    notices = validate_stop_time_increasing_distance(feed, CTX)
    assert len(notices) == 1
    n = notices[0]
    assert n.fields["stop_id"] == "s3"
    assert n.fields["csv_row_number"] == 5
    assert n.fields["shape_dist_traveled"] == 64.0
    assert n.fields["stop_sequence"] == 5
    assert n.fields["prev_csv_row_number"] == 2
    assert n.fields["prev_shape_dist_traveled"] == 85.0
    assert n.fields["prev_stop_sequence"] == 2


def test_null_stop_id_at_end_ignored_no_notice() -> None:
    """Covers: lastShapeWithoutStopShouldBeIgnored."""
    feed = make_feed([
        {"csv_row_number": 1, "trip_id": "first trip", "stop_id": "s0", "stop_sequence": 1, "shape_dist_traveled": 0.0},
        {"csv_row_number": 2, "trip_id": "first trip", "stop_id": "s1", "stop_sequence": 2, "shape_dist_traveled": 2.0},
        {"csv_row_number": 3, "trip_id": "first trip", "stop_id": None, "stop_sequence": 3, "shape_dist_traveled": 114.0},
        {"csv_row_number": 4, "trip_id": "first trip", "stop_id": None, "stop_sequence": 4, "shape_dist_traveled": 444.0},
        {"csv_row_number": 5, "trip_id": "first trip", "stop_id": "s4", "stop_sequence": 5, "shape_dist_traveled": 3.0},
        {"csv_row_number": 6, "trip_id": "first trip", "stop_id": None, "stop_sequence": 6, "shape_dist_traveled": 1.0},
    ])
    notices = validate_stop_time_increasing_distance(feed, CTX)
    assert notices == []


# ---------------------------------------------------------------------------
# Additional Python edge-case tests
# ---------------------------------------------------------------------------

def test_stop_times_table_absent_no_notice() -> None:
    notices = validate_stop_time_increasing_distance({}, CTX)
    assert notices == []


def test_stop_times_table_empty_no_notice() -> None:
    notices = validate_stop_time_increasing_distance(make_feed([]), CTX)
    assert notices == []


def test_stop_id_column_absent_no_notice() -> None:
    df = pl.DataFrame({
        "csv_row_number": [1, 2],
        "trip_id": ["T1", "T1"],
        "stop_sequence": [1, 2],
        "shape_dist_traveled": [10.0, 5.0],
    })
    feed = {"stop_times": df}
    notices = validate_stop_time_increasing_distance(feed, CTX)
    assert notices == []


def test_shape_dist_traveled_column_absent_no_notice() -> None:
    df = pl.DataFrame({
        "csv_row_number": [1, 2],
        "trip_id": ["T1", "T1"],
        "stop_id": ["s0", "s1"],
        "stop_sequence": [1, 2],
    })
    feed = {"stop_times": df}
    notices = validate_stop_time_increasing_distance(feed, CTX)
    assert notices == []


def test_single_eligible_stop_no_notice() -> None:
    feed = make_feed([
        {"csv_row_number": 1, "trip_id": "T1", "stop_id": "s0", "stop_sequence": 1, "shape_dist_traveled": 10.0},
    ])
    notices = validate_stop_time_increasing_distance(feed, CTX)
    assert notices == []


def test_stop_id_present_shape_dist_null_updates_prev_no_notice() -> None:
    """Row with stop_id but null shape_dist_traveled updates prev; next comparison is skipped."""
    feed = make_feed([
        {"csv_row_number": 1, "trip_id": "T1", "stop_id": "s0", "stop_sequence": 1, "shape_dist_traveled": 10.0},
        {"csv_row_number": 2, "trip_id": "T1", "stop_id": "s1", "stop_sequence": 2, "shape_dist_traveled": None},
        {"csv_row_number": 3, "trip_id": "T1", "stop_id": "s2", "stop_sequence": 3, "shape_dist_traveled": 5.0},
    ])
    notices = validate_stop_time_increasing_distance(feed, CTX)
    assert notices == []


def test_multiple_trips_isolated() -> None:
    """Violations in separate trips are independent; exactly 2 notices emitted."""
    feed = make_feed([
        {"csv_row_number": 1, "trip_id": "T1", "stop_id": "s0", "stop_sequence": 1, "shape_dist_traveled": 10.0},
        {"csv_row_number": 2, "trip_id": "T1", "stop_id": "s1", "stop_sequence": 2, "shape_dist_traveled": 5.0},
        {"csv_row_number": 3, "trip_id": "T2", "stop_id": "s0", "stop_sequence": 1, "shape_dist_traveled": 20.0},
        {"csv_row_number": 4, "trip_id": "T2", "stop_id": "s1", "stop_sequence": 2, "shape_dist_traveled": 8.0},
    ])
    notices = validate_stop_time_increasing_distance(feed, CTX)
    assert len(notices) == 2
    assert {n.fields["trip_id"] for n in notices} == {"T1", "T2"}


def test_stop_times_out_of_sequence_order_in_csv() -> None:
    """Sort by stop_sequence must precede the per-trip scan."""
    feed = make_feed([
        {"csv_row_number": 1, "trip_id": "T1", "stop_id": "s0", "stop_sequence": 3, "shape_dist_traveled": 3.0},
        {"csv_row_number": 2, "trip_id": "T1", "stop_id": "s1", "stop_sequence": 1, "shape_dist_traveled": 1.0},
        {"csv_row_number": 3, "trip_id": "T1", "stop_id": "s2", "stop_sequence": 2, "shape_dist_traveled": 2.0},
    ])
    notices = validate_stop_time_increasing_distance(feed, CTX)
    assert notices == []


def test_multiple_offending_pairs_same_trip() -> None:
    """One notice per offending adjacent pair, not per trip."""
    feed = make_feed([
        {"csv_row_number": 1, "trip_id": "T1", "stop_id": "s0", "stop_sequence": 1, "shape_dist_traveled": 10.0},
        {"csv_row_number": 2, "trip_id": "T1", "stop_id": "s1", "stop_sequence": 2, "shape_dist_traveled": 8.0},
        {"csv_row_number": 3, "trip_id": "T1", "stop_id": "s2", "stop_sequence": 3, "shape_dist_traveled": 5.0},
        {"csv_row_number": 4, "trip_id": "T1", "stop_id": "s3", "stop_sequence": 4, "shape_dist_traveled": 20.0},
    ])
    notices = validate_stop_time_increasing_distance(feed, CTX)
    assert len(notices) == 2
    seqs = {n.fields["stop_sequence"] for n in notices}
    assert 2 in seqs  # pair (10.0 → 8.0)
    assert 3 in seqs  # pair (8.0 → 5.0)


def test_notice_fields_are_complete() -> None:
    """The notice fields dict must contain exactly the specified keys."""
    feed = make_feed([
        {"csv_row_number": 1, "trip_id": "T1", "stop_id": "s0", "stop_sequence": 1, "shape_dist_traveled": 10.0},
        {"csv_row_number": 2, "trip_id": "T1", "stop_id": "s1", "stop_sequence": 2, "shape_dist_traveled": 5.0},
    ])
    notices = validate_stop_time_increasing_distance(feed, CTX)
    assert len(notices) == 1
    expected_keys = {
        "trip_id",
        "stop_id",
        "csv_row_number",
        "shape_dist_traveled",
        "stop_sequence",
        "prev_csv_row_number",
        "prev_shape_dist_traveled",
        "prev_stop_sequence",
    }
    assert set(notices[0].fields.keys()) == expected_keys
