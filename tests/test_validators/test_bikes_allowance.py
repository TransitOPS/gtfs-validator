"""Tests for bikes allowance validator."""

from __future__ import annotations

from datetime import date

import polars as pl

from gtfs_validator.context import ValidationContext
from gtfs_validator.notices import Severity
from gtfs_validator.validators.bikes_allowance import validate_bikes_allowance

CTX = ValidationContext(country_code="US", date_for_validation=date(2024, 6, 1))


def _routes(data: dict) -> pl.DataFrame:  # type: ignore[type-arg]
    """Build a minimal routes DataFrame."""
    return pl.DataFrame(data).cast({"route_type": pl.Int64})


def _trips(data: dict) -> pl.DataFrame:  # type: ignore[type-arg]
    """Build a minimal trips DataFrame."""
    schema_overrides = {"bikes_allowed": pl.Int64}
    return pl.DataFrame(data).cast(schema_overrides)


# ------------------------------------------------------------------
# Test 1: valid ferry trips with bikes_allowed = 1
# ------------------------------------------------------------------


def test_valid_ferry_trips_with_bikes_allowed() -> None:
    feed = {
        "routes": _routes({"csv_row_number": [2], "route_id": ["r1"], "route_type": [4]}),
        "trips": _trips({
            "csv_row_number": [2, 3],
            "trip_id": ["t1", "t2"],
            "route_id": ["r1", "r1"],
            "bikes_allowed": [1, 1],
        }),
    }
    notices = validate_bikes_allowance(feed, CTX)
    assert len(notices) == 0


# ------------------------------------------------------------------
# Test 2: valid ferry trips with bikes_allowed = 2
# ------------------------------------------------------------------


def test_valid_ferry_trips_with_bikes_not_allowed() -> None:
    feed = {
        "routes": _routes({"csv_row_number": [2], "route_id": ["r1"], "route_type": [4]}),
        "trips": _trips({
            "csv_row_number": [2, 3],
            "trip_id": ["t1", "t2"],
            "route_id": ["r1", "r1"],
            "bikes_allowed": [2, 2],
        }),
    }
    notices = validate_bikes_allowance(feed, CTX)
    assert len(notices) == 0


# ------------------------------------------------------------------
# Test 3: non-ferry route with null bikes_allowed -> no notices
# ------------------------------------------------------------------


def test_non_ferry_route_with_missing_bikes_allowed() -> None:
    feed = {
        "routes": _routes({"csv_row_number": [2], "route_id": ["r1"], "route_type": [3]}),
        "trips": _trips({
            "csv_row_number": [2, 3],
            "trip_id": ["t1", "t2"],
            "route_id": ["r1", "r1"],
            "bikes_allowed": [None, None],
        }),
    }
    notices = validate_bikes_allowance(feed, CTX)
    assert len(notices) == 0


# ------------------------------------------------------------------
# Test 4: non-ferry route with unrecognized bikes_allowed -> no notices
# ------------------------------------------------------------------


def test_non_ferry_route_with_unrecognized_bikes_allowed() -> None:
    feed = {
        "routes": _routes({"csv_row_number": [2], "route_id": ["r1"], "route_type": [3]}),
        "trips": _trips({
            "csv_row_number": [2, 3],
            "trip_id": ["t1", "t2"],
            "route_id": ["r1", "r1"],
            "bikes_allowed": [5, 5],
        }),
    }
    notices = validate_bikes_allowance(feed, CTX)
    assert len(notices) == 0


# ------------------------------------------------------------------
# Test 5: ferry trips with null bikes_allowed
# ------------------------------------------------------------------


def test_ferry_trips_with_null_bikes_allowed() -> None:
    feed = {
        "routes": _routes({"csv_row_number": [2], "route_id": ["r1"], "route_type": [4]}),
        "trips": _trips({
            "csv_row_number": [2, 3],
            "trip_id": ["t1", "t2"],
            "route_id": ["r1", "r1"],
            "bikes_allowed": [None, None],
        }),
    }
    notices = validate_bikes_allowance(feed, CTX)
    assert len(notices) == 2
    for n in notices:
        assert n.code == "missing_bike_allowance"
        assert n.severity == Severity.WARNING
        assert n.fields["route_id"] == "r1"
    trip_ids = {n.fields["trip_id"] for n in notices}
    assert trip_ids == {"t1", "t2"}


# ------------------------------------------------------------------
# Test 6: ferry trips with bikes_allowed = 0 (UNKNOWN)
# ------------------------------------------------------------------


def test_ferry_trips_with_unknown_bikes_allowed() -> None:
    feed = {
        "routes": _routes({"csv_row_number": [2], "route_id": ["r1"], "route_type": [4]}),
        "trips": _trips({
            "csv_row_number": [2, 3],
            "trip_id": ["t1", "t2"],
            "route_id": ["r1", "r1"],
            "bikes_allowed": [0, 0],
        }),
    }
    notices = validate_bikes_allowance(feed, CTX)
    assert len(notices) == 2
    for n in notices:
        assert n.code == "missing_bike_allowance"
        assert n.severity == Severity.WARNING


# ------------------------------------------------------------------
# Test 7: ferry trips with unrecognized bikes_allowed value
# ------------------------------------------------------------------


def test_ferry_trips_with_unrecognized_bikes_allowed() -> None:
    feed = {
        "routes": _routes({"csv_row_number": [2], "route_id": ["r1"], "route_type": [4]}),
        "trips": _trips({
            "csv_row_number": [2, 3],
            "trip_id": ["t1", "t2"],
            "route_id": ["r1", "r1"],
            "bikes_allowed": [5, 5],
        }),
    }
    notices = validate_bikes_allowance(feed, CTX)
    assert len(notices) == 2
    for n in notices:
        assert n.code == "missing_bike_allowance"
        assert n.severity == Severity.WARNING


# ------------------------------------------------------------------
# Test 8: missing routes table
# ------------------------------------------------------------------


def test_missing_routes_table() -> None:
    feed = {
        "trips": _trips({
            "csv_row_number": [2],
            "trip_id": ["t1"],
            "route_id": ["r1"],
            "bikes_allowed": [None],
        }),
    }
    notices = validate_bikes_allowance(feed, CTX)
    assert len(notices) == 0


# ------------------------------------------------------------------
# Test 9: missing trips table
# ------------------------------------------------------------------


def test_missing_trips_table() -> None:
    feed = {
        "routes": _routes({"csv_row_number": [2], "route_id": ["r1"], "route_type": [4]}),
    }
    notices = validate_bikes_allowance(feed, CTX)
    assert len(notices) == 0


# ------------------------------------------------------------------
# Test 10: empty routes table
# ------------------------------------------------------------------


def test_empty_routes_table() -> None:
    feed = {
        "routes": pl.DataFrame(
            schema={"csv_row_number": pl.Int64, "route_id": pl.Utf8, "route_type": pl.Int64}
        ),
        "trips": _trips({
            "csv_row_number": [2],
            "trip_id": ["t1"],
            "route_id": ["r1"],
            "bikes_allowed": [None],
        }),
    }
    notices = validate_bikes_allowance(feed, CTX)
    assert len(notices) == 0


# ------------------------------------------------------------------
# Test 11: no ferry routes
# ------------------------------------------------------------------


def test_no_ferry_routes() -> None:
    feed = {
        "routes": _routes({"csv_row_number": [2], "route_id": ["r1"], "route_type": [3]}),
        "trips": _trips({
            "csv_row_number": [2],
            "trip_id": ["t1"],
            "route_id": ["r1"],
            "bikes_allowed": [None],
        }),
    }
    notices = validate_bikes_allowance(feed, CTX)
    assert len(notices) == 0


# ------------------------------------------------------------------
# Test 12: ferry route with no trips
# ------------------------------------------------------------------


def test_ferry_route_with_no_trips() -> None:
    feed = {
        "routes": _routes({"csv_row_number": [2], "route_id": ["r1"], "route_type": [4]}),
        "trips": _trips({
            "csv_row_number": [2],
            "trip_id": ["t1"],
            "route_id": ["r2"],
            "bikes_allowed": [None],
        }),
    }
    notices = validate_bikes_allowance(feed, CTX)
    assert len(notices) == 0


# ------------------------------------------------------------------
# Test 13: mixed valid and invalid ferry trips
# ------------------------------------------------------------------


def test_mixed_valid_and_invalid_ferry_trips() -> None:
    feed = {
        "routes": _routes({"csv_row_number": [2], "route_id": ["r1"], "route_type": [4]}),
        "trips": _trips({
            "csv_row_number": [2, 3, 4],
            "trip_id": ["t1", "t2", "t3"],
            "route_id": ["r1", "r1", "r1"],
            "bikes_allowed": [1, None, 0],
        }),
    }
    notices = validate_bikes_allowance(feed, CTX)
    assert len(notices) == 2
    trip_ids = {n.fields["trip_id"] for n in notices}
    assert trip_ids == {"t2", "t3"}


# ------------------------------------------------------------------
# Test 14: multiple ferry routes mixed with non-ferry
# ------------------------------------------------------------------


def test_multiple_ferry_routes_mixed() -> None:
    feed = {
        "routes": _routes({
            "csv_row_number": [2, 3],
            "route_id": ["r1", "r2"],
            "route_type": [4, 3],
        }),
        "trips": _trips({
            "csv_row_number": [2, 3],
            "trip_id": ["t1", "t2"],
            "route_id": ["r1", "r2"],
            "bikes_allowed": [None, None],
        }),
    }
    notices = validate_bikes_allowance(feed, CTX)
    assert len(notices) == 1
    assert notices[0].fields["trip_id"] == "t1"
    assert notices[0].fields["route_id"] == "r1"
