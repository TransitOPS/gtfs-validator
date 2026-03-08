"""Tests for TripAndShapeDistanceValidator."""

from __future__ import annotations

from datetime import date
from unittest.mock import patch

import polars as pl
import pytest

from gtfs_validator.context import ValidationContext
from gtfs_validator.notices import Severity
from gtfs_validator.validators.trip_and_shape_distance import (
    _haversine_meters,
    validate_trip_and_shape_distance,
)

CTX = ValidationContext(country_code="US", date_for_validation=date(2024, 1, 1))


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def make_trips(n: int) -> pl.DataFrame:
    """Create n trips: trip_id='t0'..'t{n-1}', shape_id='s0'..'s{n-1}'."""
    return pl.DataFrame({
        "trip_id": [f"t{i}" for i in range(n)],
        "shape_id": [f"s{i}" for i in range(n)],
    })


def make_stop_times(n: int, base_dist: float) -> pl.DataFrame:
    """Create n stop times: trip_id='t0'..'t{n-1}', stop_id='st0'..'st{n-1}',
    stop_sequence=0 for all, shape_dist_traveled=base_dist+i."""
    return pl.DataFrame({
        "trip_id": [f"t{i}" for i in range(n)],
        "stop_id": [f"st{i}" for i in range(n)],
        "stop_sequence": [0] * n,
        "shape_dist_traveled": [base_dist + i for i in range(n)],
    })


def make_shapes(n: int, base_dist: float, lat_lon: float) -> pl.DataFrame:
    """Create n shapes: shape_id='s0'..'s{n-1}',
    shape_dist_traveled=base_dist+i, lat/lon=lat_lon."""
    return pl.DataFrame({
        "shape_id": [f"s{i}" for i in range(n)],
        "shape_pt_lat": [lat_lon] * n,
        "shape_pt_lon": [lat_lon] * n,
        "shape_pt_sequence": list(range(n)),
        "shape_dist_traveled": [base_dist + i for i in range(n)],
    })


def make_stops(n: int) -> pl.DataFrame:
    """Create n stops: stop_id='st0'..'st{n-1}', lat/lon=0.0."""
    return pl.DataFrame({
        "stop_id": [f"st{i}" for i in range(n)],
        "stop_lat": [0.0] * n,
        "stop_lon": [0.0] * n,
    })


# ---------------------------------------------------------------------------
# Tests derived from Java contracts
# ---------------------------------------------------------------------------


def test_trip_distance_exceeds_shape_distance_emits_error() -> None:
    """Mirrors Java's testTripDistanceExceedsShapeDistance."""
    feed = {
        "trips": make_trips(2),
        "stop_times": make_stop_times(1, 10.0),  # t0: shape_dist_traveled=10.0
        "shapes": make_shapes(1, 9.0, 10.0),     # s0: shape_dist_traveled=9.0, lat/lon=10.0
        "stops": make_stops(1),                  # st0: lat/lon=0.0
    }
    notices = validate_trip_and_shape_distance(feed, CTX)
    assert len(notices) == 1
    n = notices[0]
    assert n.code == "trip_distance_exceeds_shape_distance"
    assert n.severity == Severity.ERROR
    assert n.fields["trip_id"] == "t0"
    assert n.fields["shape_id"] == "s0"
    assert n.fields["max_trip_distance_traveled"] == 10.0
    assert n.fields["max_shape_distance_traveled"] == 9.0
    assert n.fields["geo_distance_to_shape"] > 11.1


def test_zero_shape_distance_emits_no_notice() -> None:
    """Mirrors Java's testTripDistanceExceedsShapeDistanceNoShapeDistance."""
    feed = {
        "trips": make_trips(2),
        "stop_times": make_stop_times(1, 10.0),
        "shapes": make_shapes(1, 0.0, 10.0),  # shape_dist_traveled=0.0
        "stops": make_stops(1),
    }
    notices = validate_trip_and_shape_distance(feed, CTX)
    assert notices == []


def test_trip_distance_exceeds_shape_distance_emits_warning() -> None:
    """Mirrors Java's testTripDistanceExceedsShapeDistanceWarning."""
    feed = {
        "trips": make_trips(2),
        "stop_times": make_stop_times(1, 10.0),
        "shapes": make_shapes(1, 9.0, 0.000001),  # lat/lon very close to 0.0
        "stops": make_stops(1),
    }
    notices = validate_trip_and_shape_distance(feed, CTX)
    assert len(notices) == 1
    n = notices[0]
    assert n.code == "trip_distance_exceeds_shape_distance_below_threshold"
    assert n.severity == Severity.WARNING
    assert n.fields["trip_id"] == "t0"
    assert n.fields["shape_id"] == "s0"
    assert n.fields["geo_distance_to_shape"] <= 11.1


# ---------------------------------------------------------------------------
# Edge-case tests
# ---------------------------------------------------------------------------


def test_trips_absent_returns_empty() -> None:
    feed = {
        "stop_times": make_stop_times(1, 10.0),
        "stops": make_stops(1),
        "shapes": make_shapes(1, 9.0, 10.0),
    }
    assert validate_trip_and_shape_distance(feed, CTX) == []


def test_stop_times_absent_returns_empty() -> None:
    feed = {
        "trips": make_trips(1),
        "stops": make_stops(1),
        "shapes": make_shapes(1, 9.0, 10.0),
    }
    assert validate_trip_and_shape_distance(feed, CTX) == []


def test_stops_absent_returns_empty() -> None:
    feed = {
        "trips": make_trips(1),
        "stop_times": make_stop_times(1, 10.0),
        "shapes": make_shapes(1, 9.0, 10.0),
    }
    assert validate_trip_and_shape_distance(feed, CTX) == []


def test_shapes_absent_returns_empty() -> None:
    feed = {
        "trips": make_trips(1),
        "stop_times": make_stop_times(1, 10.0),
        "stops": make_stops(1),
    }
    assert validate_trip_and_shape_distance(feed, CTX) == []


def test_trips_empty_returns_empty() -> None:
    feed = {
        "trips": pl.DataFrame({"trip_id": [], "shape_id": []}),
        "stop_times": make_stop_times(1, 10.0),
        "stops": make_stops(1),
        "shapes": make_shapes(1, 9.0, 10.0),
    }
    assert validate_trip_and_shape_distance(feed, CTX) == []


def test_trip_with_null_shape_id_skipped() -> None:
    """Trip with null shape_id does not match any shape."""
    feed = {
        "trips": pl.DataFrame({
            "trip_id": ["t0"],
            "shape_id": [None],
        }),
        "stop_times": pl.DataFrame({
            "trip_id": ["t0"],
            "stop_id": ["st0"],
            "stop_sequence": [0],
            "shape_dist_traveled": [10.0],
        }),
        "shapes": make_shapes(1, 9.0, 10.0),
        "stops": make_stops(1),
    }
    assert validate_trip_and_shape_distance(feed, CTX) == []


def test_shape_id_absent_from_shapes_skipped() -> None:
    """Trip references a shape_id not in shapes.txt."""
    feed = {
        "trips": pl.DataFrame({
            "trip_id": ["t0"],
            "shape_id": ["s99"],
        }),
        "stop_times": pl.DataFrame({
            "trip_id": ["t0"],
            "stop_id": ["st0"],
            "stop_sequence": [0],
            "shape_dist_traveled": [10.0],
        }),
        "shapes": make_shapes(1, 9.0, 10.0),  # only has s0
        "stops": make_stops(1),
    }
    assert validate_trip_and_shape_distance(feed, CTX) == []


def test_stop_id_absent_from_stops_skipped() -> None:
    """Last stop time references a stop_id not in stops.txt."""
    feed = {
        "trips": make_trips(1),
        "stop_times": pl.DataFrame({
            "trip_id": ["t0"],
            "stop_id": ["unknown"],
            "stop_sequence": [0],
            "shape_dist_traveled": [10.0],
        }),
        "shapes": make_shapes(1, 9.0, 10.0),
        "stops": make_stops(1),  # only has st0
    }
    assert validate_trip_and_shape_distance(feed, CTX) == []


def test_null_shape_dist_on_stop_time_treated_as_zero() -> None:
    """Null shape_dist_traveled on stop time is treated as 0.0; no notice."""
    feed = {
        "trips": make_trips(1),
        "stop_times": pl.DataFrame({
            "trip_id": ["t0"],
            "stop_id": ["st0"],
            "stop_sequence": [0],
            "shape_dist_traveled": [None],
        }),
        "shapes": make_shapes(1, 5.0, 10.0),  # max_shape_dist=5.0
        "stops": make_stops(1),
    }
    # 0.0 > 5.0 is false; no notice
    assert validate_trip_and_shape_distance(feed, CTX) == []


def test_null_shape_dist_on_shape_treated_as_zero() -> None:
    """Null shape_dist_traveled on shape becomes 0.0; guard fires; no notice."""
    feed = {
        "trips": make_trips(1),
        "stop_times": pl.DataFrame({
            "trip_id": ["t0"],
            "stop_id": ["st0"],
            "stop_sequence": [0],
            "shape_dist_traveled": [10.0],
        }),
        "shapes": pl.DataFrame({
            "shape_id": ["s0"],
            "shape_pt_lat": [10.0],
            "shape_pt_lon": [10.0],
            "shape_pt_sequence": [0],
            "shape_dist_traveled": [None],
        }),
        "stops": make_stops(1),
    }
    # max_shape_dist becomes 0.0; guard max_shape_dist > 0.0 fires; no notice
    assert validate_trip_and_shape_distance(feed, CTX) == []


def test_geo_distance_exactly_11_1_emits_warning() -> None:
    """Boundary condition: geo distance == 11.1 falls into WARNING (strictly >)."""
    # Mock _haversine_meters to return exactly 11.1
    feed = {
        "trips": make_trips(1),
        "stop_times": make_stop_times(1, 10.0),
        "shapes": make_shapes(1, 9.0, 10.0),
        "stops": make_stops(1),
    }
    module = "gtfs_validator.validators.trip_and_shape_distance._haversine_meters"
    with patch(module, return_value=11.1):
        notices = validate_trip_and_shape_distance(feed, CTX)
    assert len(notices) == 1
    assert notices[0].code == "trip_distance_exceeds_shape_distance_below_threshold"
    assert notices[0].severity == Severity.WARNING


def test_multiple_trips_all_violating() -> None:
    """3 trips, all violating — expect 3 ERROR notices."""
    feed = {
        "trips": make_trips(3),
        "stop_times": make_stop_times(3, 10.0),  # shape_dist_traveled: 10, 11, 12
        "shapes": make_shapes(3, 9.0, 10.0),     # shape_dist_traveled: 9, 10, 11; all lat/lon=10.0
        "stops": make_stops(3),                  # lat/lon=0.0
    }
    notices = validate_trip_and_shape_distance(feed, CTX)
    assert len(notices) == 3
    for n in notices:
        assert n.code == "trip_distance_exceeds_shape_distance"
        assert n.severity == Severity.ERROR


def test_stop_time_with_multiple_rows_uses_last_by_stop_sequence() -> None:
    """Verifies correct ordering when stop times are provided out of sequence."""
    feed = {
        "trips": pl.DataFrame({"trip_id": ["t0"], "shape_id": ["s0"]}),
        "stop_times": pl.DataFrame({
            "trip_id": ["t0", "t0"],
            "stop_id": ["st1", "st0"],       # st1 has higher sequence
            "stop_sequence": [1, 0],          # st1=seq1, st0=seq0
            "shape_dist_traveled": [5.0, 8.0],
        }),
        "shapes": pl.DataFrame({
            "shape_id": ["s0"],
            "shape_pt_lat": [10.0],
            "shape_pt_lon": [10.0],
            "shape_pt_sequence": [0],
            "shape_dist_traveled": [6.0],
        }),
        "stops": pl.DataFrame({
            "stop_id": ["st0", "st1"],
            "stop_lat": [0.0, 1.0],
            "stop_lon": [0.0, 1.0],
        }),
    }
    # After sort by stop_sequence, last is stop_sequence=1 (st1, dist=5.0)
    # 5.0 <= 6.0 — no violation
    notices = validate_trip_and_shape_distance(feed, CTX)
    assert notices == []

    # Now make stop_sequence=1 have dist=10.0 (exceeds max_shape_dist=6.0)
    feed2 = {
        "trips": pl.DataFrame({"trip_id": ["t0"], "shape_id": ["s0"]}),
        "stop_times": pl.DataFrame({
            "trip_id": ["t0", "t0"],
            "stop_id": ["st1", "st0"],
            "stop_sequence": [1, 0],
            "shape_dist_traveled": [10.0, 8.0],  # st1 (seq=1) has dist=10.0
        }),
        "shapes": pl.DataFrame({
            "shape_id": ["s0"],
            "shape_pt_lat": [10.0],
            "shape_pt_lon": [10.0],
            "shape_pt_sequence": [0],
            "shape_dist_traveled": [6.0],
        }),
        "stops": pl.DataFrame({
            "stop_id": ["st0", "st1"],
            "stop_lat": [0.0, 1.0],
            "stop_lon": [0.0, 1.0],
        }),
    }
    notices2 = validate_trip_and_shape_distance(feed2, CTX)
    assert len(notices2) == 1
    assert notices2[0].code == "trip_distance_exceeds_shape_distance"
    assert notices2[0].severity == Severity.ERROR


def test_trip_with_exactly_one_stop_time_is_processed() -> None:
    """One trip, one stop time (first == last). Expect 1 ERROR."""
    feed = {
        "trips": make_trips(1),
        "stop_times": make_stop_times(1, 10.0),
        "shapes": make_shapes(1, 9.0, 10.0),
        "stops": make_stops(1),
    }
    notices = validate_trip_and_shape_distance(feed, CTX)
    assert len(notices) == 1
    assert notices[0].code == "trip_distance_exceeds_shape_distance"
    assert notices[0].severity == Severity.ERROR


# ---------------------------------------------------------------------------
# Haversine unit tests
# ---------------------------------------------------------------------------


def test_haversine_meters_unit() -> None:
    # Distance from (0.0, 0.0) to (10.0, 10.0) ~ 1,568,520 m
    dist = _haversine_meters(0.0, 0.0, 10.0, 10.0)
    assert 1_560_000 < dist < 1_580_000

    # Self-distance
    assert _haversine_meters(45.0, 90.0, 45.0, 90.0) == pytest.approx(0.0, abs=1e-6)

    # Near-zero distance (matches Java's 0.000001 lat/lon test case)
    dist_near = _haversine_meters(0.0, 0.0, 0.000001, 0.000001)
    assert dist_near < 1.0  # well below the 11.1 m threshold
