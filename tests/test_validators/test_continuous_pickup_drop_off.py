"""Tests for continuous pickup/drop-off validator."""

from __future__ import annotations

from datetime import date

import polars as pl

from gtfs_validator.context import ValidationContext
from gtfs_validator.notices import Severity
from gtfs_validator.validators.continuous_pickup_drop_off import (
    validate_continuous_pickup_drop_off,
)

CTX = ValidationContext(country_code="US", date_for_validation=date(2024, 6, 1))


def make_routes(rows: list[dict]) -> pl.DataFrame:  # type: ignore[type-arg]
    """Create a routes DataFrame with csv_row_number, route_id, and optional continuous columns."""
    df = pl.DataFrame(rows)
    casts: dict[str, pl.DataType] = {}
    if "continuous_pickup" in df.columns:
        casts["continuous_pickup"] = pl.Int64
    if "continuous_drop_off" in df.columns:
        casts["continuous_drop_off"] = pl.Int64
    if casts:
        df = df.cast(casts)
    return df


def make_trips(rows: list[dict]) -> pl.DataFrame:  # type: ignore[type-arg]
    """Create a trips DataFrame with trip_id and route_id."""
    return pl.DataFrame(rows)


def make_stop_times(rows: list[dict]) -> pl.DataFrame:  # type: ignore[type-arg]
    """Create a stop_times DataFrame with csv_row_number, trip_id, and optional window columns."""
    return pl.DataFrame(rows)


def make_feed(
    routes: pl.DataFrame,
    trips: pl.DataFrame,
    stop_times: pl.DataFrame,
) -> dict[str, pl.DataFrame]:
    """Assemble feed dict from the three DataFrames."""
    return {"routes": routes, "trips": trips, "stop_times": stop_times}


# ------------------------------------------------------------------
# Test 1: continuous_pickup with window -> notice
# ------------------------------------------------------------------


def test_continuous_pickup_with_window() -> None:
    routes = make_routes([
        {"csv_row_number": 2, "route_id": "r1", "continuous_pickup": 2},
    ])
    trips = make_trips([{"trip_id": "trip1", "route_id": "r1"}])
    stop_times = make_stop_times([{
        "csv_row_number": 3,
        "trip_id": "trip1",
        "start_pickup_drop_off_window": "08:00:00",
        "end_pickup_drop_off_window": "09:00:00",
    }])
    notices = validate_continuous_pickup_drop_off(make_feed(routes, trips, stop_times), CTX)
    assert len(notices) == 1
    n = notices[0]
    assert n.code == "forbidden_continuous_pickup_drop_off"
    assert n.severity == Severity.ERROR
    assert n.fields["route_csv_row_number"] == 2
    assert n.fields["trip_id"] == "trip1"
    assert n.fields["start_pickup_drop_off_window"] == "08:00:00"
    assert n.fields["end_pickup_drop_off_window"] == "09:00:00"


# ------------------------------------------------------------------
# Test 2: no continuous -> no notice
# ------------------------------------------------------------------


def test_no_continuous_no_notice() -> None:
    routes = make_routes([
        {"csv_row_number": 2, "route_id": "r1", "continuous_pickup": 1, "continuous_drop_off": 1},
    ])
    trips = make_trips([{"trip_id": "trip1", "route_id": "r1"}])
    stop_times = make_stop_times([{
        "csv_row_number": 3,
        "trip_id": "trip1",
        "start_pickup_drop_off_window": "08:00:00",
        "end_pickup_drop_off_window": "09:00:00",
    }])
    notices = validate_continuous_pickup_drop_off(make_feed(routes, trips, stop_times), CTX)
    assert len(notices) == 0


# ------------------------------------------------------------------
# Test 3: continuous but no window -> no notice
# ------------------------------------------------------------------


def test_continuous_no_window_no_notice() -> None:
    routes = make_routes([
        {"csv_row_number": 2, "route_id": "r1", "continuous_pickup": 0},
    ])
    trips = make_trips([{"trip_id": "trip1", "route_id": "r1"}])
    stop_times = make_stop_times([{
        "csv_row_number": 3,
        "trip_id": "trip1",
        "start_pickup_drop_off_window": None,
        "end_pickup_drop_off_window": None,
    }])
    notices = validate_continuous_pickup_drop_off(make_feed(routes, trips, stop_times), CTX)
    assert len(notices) == 0


# ------------------------------------------------------------------
# Test 4: continuous_drop_off only
# ------------------------------------------------------------------


def test_continuous_drop_off_only() -> None:
    routes = make_routes([
        {"csv_row_number": 2, "route_id": "r1", "continuous_pickup": 1, "continuous_drop_off": 0},
    ])
    trips = make_trips([{"trip_id": "trip1", "route_id": "r1"}])
    stop_times = make_stop_times([{
        "csv_row_number": 3,
        "trip_id": "trip1",
        "start_pickup_drop_off_window": "10:00:00",
        "end_pickup_drop_off_window": None,
    }])
    notices = validate_continuous_pickup_drop_off(make_feed(routes, trips, stop_times), CTX)
    assert len(notices) == 1
    assert notices[0].code == "forbidden_continuous_pickup_drop_off"


# ------------------------------------------------------------------
# Test 5: multiple trips, one with window
# ------------------------------------------------------------------


def test_multiple_trips_one_with_window() -> None:
    routes = make_routes([
        {"csv_row_number": 2, "route_id": "r1", "continuous_pickup": 3},
    ])
    trips = make_trips([
        {"trip_id": "trip1", "route_id": "r1"},
        {"trip_id": "trip2", "route_id": "r1"},
    ])
    stop_times = make_stop_times([
        {
            "csv_row_number": 3,
            "trip_id": "trip1",
            "start_pickup_drop_off_window": "08:00:00",
            "end_pickup_drop_off_window": "09:00:00",
        },
        {
            "csv_row_number": 4,
            "trip_id": "trip2",
            "start_pickup_drop_off_window": None,
            "end_pickup_drop_off_window": None,
        },
    ])
    notices = validate_continuous_pickup_drop_off(make_feed(routes, trips, stop_times), CTX)
    assert len(notices) == 1


# ------------------------------------------------------------------
# Test 6: multiple stop_times with windows
# ------------------------------------------------------------------


def test_multiple_stop_times_with_windows() -> None:
    routes = make_routes([
        {"csv_row_number": 2, "route_id": "r1", "continuous_pickup": 0},
    ])
    trips = make_trips([{"trip_id": "trip1", "route_id": "r1"}])
    stop_times = make_stop_times([
        {
            "csv_row_number": 3,
            "trip_id": "trip1",
            "start_pickup_drop_off_window": "08:00:00",
            "end_pickup_drop_off_window": "09:00:00",
        },
        {
            "csv_row_number": 4,
            "trip_id": "trip1",
            "start_pickup_drop_off_window": "10:00:00",
            "end_pickup_drop_off_window": "11:00:00",
        },
    ])
    notices = validate_continuous_pickup_drop_off(make_feed(routes, trips, stop_times), CTX)
    assert len(notices) == 2


# ------------------------------------------------------------------
# Test 7: only start_window set
# ------------------------------------------------------------------


def test_only_start_window_set() -> None:
    routes = make_routes([
        {"csv_row_number": 2, "route_id": "r1", "continuous_pickup": 2},
    ])
    trips = make_trips([{"trip_id": "trip1", "route_id": "r1"}])
    stop_times = make_stop_times([{
        "csv_row_number": 3,
        "trip_id": "trip1",
        "start_pickup_drop_off_window": "08:00:00",
        "end_pickup_drop_off_window": None,
    }])
    notices = validate_continuous_pickup_drop_off(make_feed(routes, trips, stop_times), CTX)
    assert len(notices) == 1
    assert notices[0].fields["end_pickup_drop_off_window"] is None


# ------------------------------------------------------------------
# Test 8: only end_window set
# ------------------------------------------------------------------


def test_only_end_window_set() -> None:
    routes = make_routes([
        {"csv_row_number": 2, "route_id": "r1", "continuous_pickup": 2},
    ])
    trips = make_trips([{"trip_id": "trip1", "route_id": "r1"}])
    stop_times = make_stop_times([{
        "csv_row_number": 3,
        "trip_id": "trip1",
        "start_pickup_drop_off_window": None,
        "end_pickup_drop_off_window": "09:00:00",
    }])
    notices = validate_continuous_pickup_drop_off(make_feed(routes, trips, stop_times), CTX)
    assert len(notices) == 1
    assert notices[0].fields["start_pickup_drop_off_window"] is None


# ------------------------------------------------------------------
# Test 9: route with no trips
# ------------------------------------------------------------------


def test_route_with_no_trips() -> None:
    routes = make_routes([
        {"csv_row_number": 2, "route_id": "r1", "continuous_pickup": 0},
    ])
    trips = make_trips([{"trip_id": "trip_other", "route_id": "r_other"}])
    stop_times = make_stop_times([{
        "csv_row_number": 3,
        "trip_id": "trip_other",
        "start_pickup_drop_off_window": "08:00:00",
        "end_pickup_drop_off_window": "09:00:00",
    }])
    notices = validate_continuous_pickup_drop_off(make_feed(routes, trips, stop_times), CTX)
    assert len(notices) == 0


# ------------------------------------------------------------------
# Test 10: trip with no stop_times
# ------------------------------------------------------------------


def test_trip_with_no_stop_times() -> None:
    routes = make_routes([
        {"csv_row_number": 2, "route_id": "r1", "continuous_pickup": 0},
    ])
    trips = make_trips([{"trip_id": "trip1", "route_id": "r1"}])
    stop_times = make_stop_times([{
        "csv_row_number": 3,
        "trip_id": "trip_other",
        "start_pickup_drop_off_window": "08:00:00",
        "end_pickup_drop_off_window": "09:00:00",
    }])
    notices = validate_continuous_pickup_drop_off(make_feed(routes, trips, stop_times), CTX)
    assert len(notices) == 0


# ------------------------------------------------------------------
# Test 11: missing routes table
# ------------------------------------------------------------------


def test_missing_routes_table() -> None:
    feed: dict[str, pl.DataFrame] = {
        "stop_times": make_stop_times([{
            "csv_row_number": 3,
            "trip_id": "trip1",
            "start_pickup_drop_off_window": "08:00:00",
            "end_pickup_drop_off_window": "09:00:00",
        }]),
    }
    notices = validate_continuous_pickup_drop_off(feed, CTX)
    assert len(notices) == 0


# ------------------------------------------------------------------
# Test 12: missing stop_times table
# ------------------------------------------------------------------


def test_missing_stop_times_table() -> None:
    feed: dict[str, pl.DataFrame] = {
        "routes": make_routes([
            {"csv_row_number": 2, "route_id": "r1", "continuous_pickup": 0},
        ]),
    }
    notices = validate_continuous_pickup_drop_off(feed, CTX)
    assert len(notices) == 0


# ------------------------------------------------------------------
# Test 13: missing trips table
# ------------------------------------------------------------------


def test_missing_trips_table() -> None:
    routes = make_routes([
        {"csv_row_number": 2, "route_id": "r1", "continuous_pickup": 0},
    ])
    stop_times = make_stop_times([{
        "csv_row_number": 3,
        "trip_id": "trip1",
        "start_pickup_drop_off_window": "08:00:00",
        "end_pickup_drop_off_window": "09:00:00",
    }])
    feed: dict[str, pl.DataFrame] = {"routes": routes, "stop_times": stop_times}
    notices = validate_continuous_pickup_drop_off(feed, CTX)
    assert len(notices) == 0


# ------------------------------------------------------------------
# Test 14: default continuous_pickup=0 is continuous
# ------------------------------------------------------------------


def test_default_continuous_pickup_zero_is_continuous() -> None:
    routes = make_routes([
        {"csv_row_number": 2, "route_id": "r1", "continuous_pickup": 0},
    ])
    trips = make_trips([{"trip_id": "trip1", "route_id": "r1"}])
    stop_times = make_stop_times([{
        "csv_row_number": 3,
        "trip_id": "trip1",
        "start_pickup_drop_off_window": "08:00:00",
        "end_pickup_drop_off_window": "09:00:00",
    }])
    notices = validate_continuous_pickup_drop_off(make_feed(routes, trips, stop_times), CTX)
    assert len(notices) == 1


# ------------------------------------------------------------------
# Test 15: continuous_pickup all enum values
# ------------------------------------------------------------------


def test_continuous_pickup_all_enum_values() -> None:
    routes = make_routes([
        {"csv_row_number": 2, "route_id": "r0", "continuous_pickup": 0},
        {"csv_row_number": 3, "route_id": "r1", "continuous_pickup": 1},
        {"csv_row_number": 4, "route_id": "r2", "continuous_pickup": 2},
        {"csv_row_number": 5, "route_id": "r3", "continuous_pickup": 3},
    ])
    trips = make_trips([
        {"trip_id": "t0", "route_id": "r0"},
        {"trip_id": "t1", "route_id": "r1"},
        {"trip_id": "t2", "route_id": "r2"},
        {"trip_id": "t3", "route_id": "r3"},
    ])
    stop_times = make_stop_times([
        {"csv_row_number": 10, "trip_id": "t0", "start_pickup_drop_off_window": "08:00:00", "end_pickup_drop_off_window": "09:00:00"},
        {"csv_row_number": 11, "trip_id": "t1", "start_pickup_drop_off_window": "08:00:00", "end_pickup_drop_off_window": "09:00:00"},
        {"csv_row_number": 12, "trip_id": "t2", "start_pickup_drop_off_window": "08:00:00", "end_pickup_drop_off_window": "09:00:00"},
        {"csv_row_number": 13, "trip_id": "t3", "start_pickup_drop_off_window": "08:00:00", "end_pickup_drop_off_window": "09:00:00"},
    ])
    notices = validate_continuous_pickup_drop_off(make_feed(routes, trips, stop_times), CTX)
    assert len(notices) == 3
    trip_ids = {n.fields["trip_id"] for n in notices}
    assert trip_ids == {"t0", "t2", "t3"}


# ------------------------------------------------------------------
# Test 16: skip when no continuous columns
# ------------------------------------------------------------------


def test_skip_when_no_continuous_columns() -> None:
    routes = pl.DataFrame({"csv_row_number": [2], "route_id": ["r1"]})
    trips = make_trips([{"trip_id": "trip1", "route_id": "r1"}])
    stop_times = make_stop_times([{
        "csv_row_number": 3,
        "trip_id": "trip1",
        "start_pickup_drop_off_window": "08:00:00",
        "end_pickup_drop_off_window": "09:00:00",
    }])
    notices = validate_continuous_pickup_drop_off(make_feed(routes, trips, stop_times), CTX)
    assert len(notices) == 0


# ------------------------------------------------------------------
# Test 17: skip when no window columns with drop_off only
# ------------------------------------------------------------------


def test_skip_when_no_window_columns_with_drop_off_only() -> None:
    routes = make_routes([
        {"csv_row_number": 2, "route_id": "r1", "continuous_drop_off": 0},
    ])
    trips = make_trips([{"trip_id": "trip1", "route_id": "r1"}])
    stop_times = pl.DataFrame({
        "csv_row_number": [3],
        "trip_id": ["trip1"],
    })
    notices = validate_continuous_pickup_drop_off(make_feed(routes, trips, stop_times), CTX)
    assert len(notices) == 0


# ------------------------------------------------------------------
# Test 18: runs when continuous_pickup present without window columns
# ------------------------------------------------------------------


def test_runs_when_continuous_pickup_present_without_window_columns() -> None:
    routes = make_routes([
        {"csv_row_number": 2, "route_id": "r1", "continuous_pickup": 2},
    ])
    trips = make_trips([{"trip_id": "trip1", "route_id": "r1"}])
    stop_times = pl.DataFrame({
        "csv_row_number": [3],
        "trip_id": ["trip1"],
    })
    notices = validate_continuous_pickup_drop_off(make_feed(routes, trips, stop_times), CTX)
    assert len(notices) == 0


# ------------------------------------------------------------------
# Test 19: multiple routes mixed
# ------------------------------------------------------------------


def test_multiple_routes_mixed() -> None:
    routes = make_routes([
        {"csv_row_number": 2, "route_id": "r1", "continuous_pickup": 0},
        {"csv_row_number": 3, "route_id": "r2", "continuous_pickup": 1},
    ])
    trips = make_trips([
        {"trip_id": "t1", "route_id": "r1"},
        {"trip_id": "t2", "route_id": "r2"},
    ])
    stop_times = make_stop_times([
        {"csv_row_number": 10, "trip_id": "t1", "start_pickup_drop_off_window": "08:00:00", "end_pickup_drop_off_window": "09:00:00"},
        {"csv_row_number": 11, "trip_id": "t2", "start_pickup_drop_off_window": "08:00:00", "end_pickup_drop_off_window": "09:00:00"},
    ])
    notices = validate_continuous_pickup_drop_off(make_feed(routes, trips, stop_times), CTX)
    assert len(notices) == 1
    assert notices[0].fields["trip_id"] == "t1"
