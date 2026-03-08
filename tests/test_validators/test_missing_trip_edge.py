"""Tests for MissingTripEdgeValidator."""

from datetime import date

import polars as pl

from gtfs_validator.context import ValidationContext
from gtfs_validator.notices import Severity
from gtfs_validator.validators.missing_trip_edge import validate_missing_trip_edge

CTX = ValidationContext(country_code="US", date_for_validation=date(2026, 3, 8))


def make_stop_times(rows: list[dict]) -> pl.DataFrame:
    """Build a stop_times DataFrame from a list of row dicts.

    Each dict must supply:
      trip_id (str), stop_sequence (int), _row_number (int),
      arrival_time (str | None), departure_time (str | None).
    Optional:
      start_pickup_drop_off_window (str | None),
      end_pickup_drop_off_window (str | None).
    """
    base: dict[str, list] = {
        "trip_id": [],
        "stop_sequence": [],
        "_row_number": [],
        "arrival_time": [],
        "departure_time": [],
        "start_pickup_drop_off_window": [],
        "end_pickup_drop_off_window": [],
    }
    for r in rows:
        base["trip_id"].append(r["trip_id"])
        base["stop_sequence"].append(r["stop_sequence"])
        base["_row_number"].append(r["_row_number"])
        base["arrival_time"].append(r.get("arrival_time"))
        base["departure_time"].append(r.get("departure_time"))
        base["start_pickup_drop_off_window"].append(
            r.get("start_pickup_drop_off_window")
        )
        base["end_pickup_drop_off_window"].append(
            r.get("end_pickup_drop_off_window")
        )
    return pl.DataFrame(
        base,
        schema={
            "trip_id": pl.Utf8,
            "stop_sequence": pl.Int64,
            "_row_number": pl.Int64,
            "arrival_time": pl.Utf8,
            "departure_time": pl.Utf8,
            "start_pickup_drop_off_window": pl.Utf8,
            "end_pickup_drop_off_window": pl.Utf8,
        },
    )


def test_first_stop_missing_arrival_time_yields_notice() -> None:
    feed = {
        "stop_times": make_stop_times([
            {"trip_id": "trip id value", "stop_sequence": 1, "_row_number": 2,
             "arrival_time": None, "departure_time": "08:00:00"},
            {"trip_id": "trip id value", "stop_sequence": 2, "_row_number": 4,
             "arrival_time": "08:10:00", "departure_time": "08:10:00"},
            {"trip_id": "trip id value", "stop_sequence": 3, "_row_number": 5,
             "arrival_time": "08:20:00", "departure_time": "08:20:00"},
            {"trip_id": "trip id value", "stop_sequence": 4, "_row_number": 3,
             "arrival_time": "08:30:00", "departure_time": "08:30:00"},
        ])
    }
    notices = validate_missing_trip_edge(feed, CTX)
    assert len(notices) == 1
    n = notices[0]
    assert n.code == "missing_trip_edge"
    assert n.severity == Severity.ERROR
    assert n.fields["csv_row_number"] == 2
    assert n.fields["stop_sequence"] == 1
    assert n.fields["trip_id"] == "trip id value"
    assert n.fields["specified_field"] == "arrival_time"


def test_first_stop_missing_departure_time_yields_notice() -> None:
    feed = {
        "stop_times": make_stop_times([
            {"trip_id": "trip id value", "stop_sequence": 1, "_row_number": 2,
             "arrival_time": "08:00:00", "departure_time": None},
            {"trip_id": "trip id value", "stop_sequence": 2, "_row_number": 4,
             "arrival_time": "08:10:00", "departure_time": "08:10:00"},
            {"trip_id": "trip id value", "stop_sequence": 3, "_row_number": 5,
             "arrival_time": "08:20:00", "departure_time": "08:20:00"},
            {"trip_id": "trip id value", "stop_sequence": 5, "_row_number": 3,
             "arrival_time": "08:30:00", "departure_time": "08:30:00"},
        ])
    }
    notices = validate_missing_trip_edge(feed, CTX)
    assert len(notices) == 1
    n = notices[0]
    assert n.code == "missing_trip_edge"
    assert n.severity == Severity.ERROR
    assert n.fields["stop_sequence"] == 1
    assert n.fields["csv_row_number"] == 2
    assert n.fields["specified_field"] == "departure_time"


def test_last_stop_missing_arrival_time_yields_notice() -> None:
    feed = {
        "stop_times": make_stop_times([
            {"trip_id": "trip id value", "stop_sequence": 1, "_row_number": 2,
             "arrival_time": "08:00:00", "departure_time": "08:00:00"},
            {"trip_id": "trip id value", "stop_sequence": 2, "_row_number": 4,
             "arrival_time": "08:10:00", "departure_time": "08:10:00"},
            {"trip_id": "trip id value", "stop_sequence": 3, "_row_number": 5,
             "arrival_time": "08:20:00", "departure_time": "08:20:00"},
            {"trip_id": "trip id value", "stop_sequence": 5, "_row_number": 10,
             "arrival_time": None, "departure_time": "08:30:00"},
        ])
    }
    notices = validate_missing_trip_edge(feed, CTX)
    assert len(notices) == 1
    n = notices[0]
    assert n.fields["stop_sequence"] == 5
    assert n.fields["csv_row_number"] == 10
    assert n.fields["specified_field"] == "arrival_time"


def test_last_stop_missing_departure_time_yields_notice() -> None:
    feed = {
        "stop_times": make_stop_times([
            {"trip_id": "trip id value", "stop_sequence": 1, "_row_number": 2,
             "arrival_time": "08:00:00", "departure_time": "08:00:00"},
            {"trip_id": "trip id value", "stop_sequence": 2, "_row_number": 4,
             "arrival_time": "08:10:00", "departure_time": "08:10:00"},
            {"trip_id": "trip id value", "stop_sequence": 3, "_row_number": 5,
             "arrival_time": "08:20:00", "departure_time": "08:20:00"},
            {"trip_id": "trip id value", "stop_sequence": 5, "_row_number": 10,
             "arrival_time": "08:30:00", "departure_time": None},
        ])
    }
    notices = validate_missing_trip_edge(feed, CTX)
    assert len(notices) == 1
    n = notices[0]
    assert n.fields["stop_sequence"] == 5
    assert n.fields["csv_row_number"] == 10
    assert n.fields["specified_field"] == "departure_time"


def test_valid_edges_no_notice() -> None:
    feed = {
        "stop_times": make_stop_times([
            {"trip_id": "trip id value", "stop_sequence": 1, "_row_number": 2,
             "arrival_time": "08:00:00", "departure_time": "08:00:00"},
            {"trip_id": "trip id value", "stop_sequence": 2, "_row_number": 3,
             "arrival_time": None, "departure_time": None},
            {"trip_id": "trip id value", "stop_sequence": 3, "_row_number": 4,
             "arrival_time": None, "departure_time": None},
            {"trip_id": "trip id value", "stop_sequence": 4, "_row_number": 10,
             "arrival_time": "08:30:00", "departure_time": "08:30:00"},
        ])
    }
    notices = validate_missing_trip_edge(feed, CTX)
    assert notices == []


def test_pickup_drop_off_window_exempts_edge() -> None:
    feed = {
        "stop_times": make_stop_times([
            {"trip_id": "trip id value", "stop_sequence": 1, "_row_number": 2,
             "arrival_time": None, "departure_time": None,
             "start_pickup_drop_off_window": "08:00:00",
             "end_pickup_drop_off_window": "10:00:00"},
            {"trip_id": "trip id value 2", "stop_sequence": 1, "_row_number": 3,
             "arrival_time": None, "departure_time": None,
             "start_pickup_drop_off_window": "08:00:00",
             "end_pickup_drop_off_window": "10:00:00"},
        ])
    }
    notices = validate_missing_trip_edge(feed, CTX)
    assert notices == []


def test_stop_times_absent_no_notices() -> None:
    notices = validate_missing_trip_edge({}, CTX)
    assert notices == []


def test_stop_times_empty_no_notices() -> None:
    feed = {"stop_times": make_stop_times([])}
    notices = validate_missing_trip_edge(feed, CTX)
    assert notices == []


def test_single_stop_trip_missing_both_times_yields_four_notices() -> None:
    feed = {
        "stop_times": make_stop_times([
            {"trip_id": "t1", "stop_sequence": 1, "_row_number": 2,
             "arrival_time": None, "departure_time": None},
        ])
    }
    notices = validate_missing_trip_edge(feed, CTX)
    assert len(notices) == 4
    assert all(n.code == "missing_trip_edge" for n in notices)
    assert all(n.severity == Severity.ERROR for n in notices)
    arrival_notices = [n for n in notices if n.fields["specified_field"] == "arrival_time"]
    departure_notices = [n for n in notices if n.fields["specified_field"] == "departure_time"]
    assert len(arrival_notices) == 2
    assert len(departure_notices) == 2


def test_middle_stops_missing_times_no_notice() -> None:
    feed = {
        "stop_times": make_stop_times([
            {"trip_id": "t1", "stop_sequence": 1, "_row_number": 2,
             "arrival_time": "08:00:00", "departure_time": "08:00:00"},
            {"trip_id": "t1", "stop_sequence": 2, "_row_number": 3,
             "arrival_time": None, "departure_time": None},
            {"trip_id": "t1", "stop_sequence": 3, "_row_number": 4,
             "arrival_time": "08:30:00", "departure_time": "08:30:00"},
        ])
    }
    notices = validate_missing_trip_edge(feed, CTX)
    assert notices == []


def test_only_start_window_present_exempts_row() -> None:
    feed = {
        "stop_times": make_stop_times([
            {"trip_id": "t1", "stop_sequence": 1, "_row_number": 2,
             "arrival_time": None, "departure_time": None,
             "start_pickup_drop_off_window": "08:00:00",
             "end_pickup_drop_off_window": None},
        ])
    }
    notices = validate_missing_trip_edge(feed, CTX)
    assert notices == []


def test_only_end_window_present_exempts_row() -> None:
    feed = {
        "stop_times": make_stop_times([
            {"trip_id": "t1", "stop_sequence": 1, "_row_number": 2,
             "arrival_time": None, "departure_time": None,
             "start_pickup_drop_off_window": None,
             "end_pickup_drop_off_window": "10:00:00"},
        ])
    }
    notices = validate_missing_trip_edge(feed, CTX)
    assert notices == []


def test_multiple_trips_mixed_validity() -> None:
    feed = {
        "stop_times": make_stop_times([
            # Trip A
            {"trip_id": "A", "stop_sequence": 1, "_row_number": 2,
             "arrival_time": "08:00:00", "departure_time": "08:00:00"},
            {"trip_id": "A", "stop_sequence": 2, "_row_number": 3,
             "arrival_time": None, "departure_time": "09:00:00"},
            # Trip B
            {"trip_id": "B", "stop_sequence": 1, "_row_number": 4,
             "arrival_time": "08:00:00", "departure_time": "08:00:00"},
            {"trip_id": "B", "stop_sequence": 2, "_row_number": 5,
             "arrival_time": "09:00:00", "departure_time": "09:00:00"},
            # Trip C
            {"trip_id": "C", "stop_sequence": 1, "_row_number": 6,
             "arrival_time": "08:00:00", "departure_time": None},
            {"trip_id": "C", "stop_sequence": 2, "_row_number": 7,
             "arrival_time": "09:00:00", "departure_time": "09:00:00"},
        ])
    }
    notices = validate_missing_trip_edge(feed, CTX)
    assert len(notices) == 2
    trip_a_notices = [n for n in notices if n.fields["trip_id"] == "A"]
    trip_b_notices = [n for n in notices if n.fields["trip_id"] == "B"]
    trip_c_notices = [n for n in notices if n.fields["trip_id"] == "C"]
    assert len(trip_a_notices) == 1
    assert trip_a_notices[0].fields["specified_field"] == "arrival_time"
    assert trip_a_notices[0].fields["stop_sequence"] == 2
    assert len(trip_b_notices) == 0
    assert len(trip_c_notices) == 1
    assert trip_c_notices[0].fields["specified_field"] == "departure_time"
    assert trip_c_notices[0].fields["stop_sequence"] == 1


def test_unsorted_input_still_finds_correct_edges() -> None:
    feed = {
        "stop_times": make_stop_times([
            {"trip_id": "t1", "stop_sequence": 3, "_row_number": 2,
             "arrival_time": "09:00:00", "departure_time": "09:00:00"},
            {"trip_id": "t1", "stop_sequence": 1, "_row_number": 3,
             "arrival_time": None, "departure_time": "08:00:00"},
            {"trip_id": "t1", "stop_sequence": 2, "_row_number": 4,
             "arrival_time": "08:30:00", "departure_time": "08:30:00"},
        ])
    }
    notices = validate_missing_trip_edge(feed, CTX)
    assert len(notices) == 1
    assert notices[0].fields["stop_sequence"] == 1
    assert notices[0].fields["specified_field"] == "arrival_time"
