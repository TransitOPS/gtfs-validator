"""Tests for block trips with overlapping stop times validator."""

from __future__ import annotations

from datetime import date

import polars as pl

from gtfs_validator.context import ValidationContext
from gtfs_validator.notices import Severity
from gtfs_validator.validators.block_trips_overlapping import (
    validate_block_trips_overlapping,
)

CTX = ValidationContext(country_code="US", date_for_validation=date(2021, 1, 4))


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _make_calendar(
    service_id: str,
    start_date: date,
    end_date: date,
    days: str = "1111100",
) -> pl.DataFrame:
    day_cols = [
        "monday", "tuesday", "wednesday", "thursday",
        "friday", "saturday", "sunday",
    ]
    data: dict[str, list] = {  # type: ignore[type-arg]
        "service_id": [service_id],
        "start_date": [start_date],
        "end_date": [end_date],
    }
    for i, col in enumerate(day_cols):
        data[col] = [int(days[i])]
    return pl.DataFrame(data)


def _make_trip(
    trip_id: str,
    service_id: str,
    block_id: str | None,
    csv_row_number: int,
) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "trip_id": [trip_id],
            "service_id": [service_id],
            "block_id": [block_id],
            "csv_row_number": [csv_row_number],
        },
        schema={
            "trip_id": pl.Utf8,
            "service_id": pl.Utf8,
            "block_id": pl.Utf8,
            "csv_row_number": pl.Int64,
        },
    )


def _make_stop_times(
    trip_id: str,
    times: list[tuple[int, int]],
) -> pl.DataFrame:
    """Create stop_times rows for a trip.

    times is a list of (arrival_seconds, departure_seconds) per stop.
    """
    n = len(times)
    return pl.DataFrame({
        "trip_id": [trip_id] * n,
        "stop_sequence": list(range(1, n + 1)),
        "arrival_time": [t[0] for t in times],
        "departure_time": [t[1] for t in times],
    })


def _concat_trips(*dfs: pl.DataFrame) -> pl.DataFrame:
    return pl.concat(list(dfs))


def _concat_stop_times(*dfs: pl.DataFrame) -> pl.DataFrame:
    return pl.concat(list(dfs))


def _concat_calendars(*dfs: pl.DataFrame) -> pl.DataFrame:
    return pl.concat(list(dfs))


# Time constants (seconds since midnight)
T_0800 = 8 * 3600   # 28800
T_0830 = 8 * 3600 + 30 * 60  # 30600
T_0900 = 9 * 3600   # 32400
T_0930 = 9 * 3600 + 30 * 60  # 34200
T_1000 = 10 * 3600  # 36000
T_1100 = 11 * 3600  # 39600
T_1200 = 12 * 3600  # 43200
T_1300 = 13 * 3600  # 46800


# ------------------------------------------------------------------
# Test 1: good feed, no overlap
# ------------------------------------------------------------------


def test_good_feed_no_overlap() -> None:
    trips = _concat_trips(
        _make_trip("t0", "WEEK", "b1", 2),
        _make_trip("t1", "WEEK", "b1", 3),
        _make_trip("t2", "WEEK", "b1", 4),
        _make_trip("t3", "SAT", "b1", 5),
        _make_trip("t4", "SAT", "b1", 6),
        _make_trip("t5", "SAT", "b1", 7),
    )
    stop_times = _concat_stop_times(
        _make_stop_times("t0", [(T_0800, T_0800), (T_0900, T_0900)]),
        _make_stop_times("t1", [(T_1000, T_1000), (T_1100, T_1100)]),
        _make_stop_times("t2", [(T_1200, T_1200), (T_1300, T_1300)]),
        _make_stop_times("t3", [(T_0800, T_0800), (T_0900, T_0900)]),
        _make_stop_times("t4", [(T_1000, T_1000), (T_1100, T_1100)]),
        _make_stop_times("t5", [(T_1200, T_1200), (T_1300, T_1300)]),
    )
    calendar = _concat_calendars(
        _make_calendar("WEEK", date(2021, 1, 4), date(2021, 1, 10), "1111100"),
        _make_calendar("SAT", date(2021, 1, 4), date(2021, 1, 10), "0000010"),
    )
    feed = {"trips": trips, "stop_times": stop_times, "calendar": calendar}
    notices = validate_block_trips_overlapping(feed, CTX)
    assert len(notices) == 0


# ------------------------------------------------------------------
# Test 2: overlap with same service_id
# ------------------------------------------------------------------


def test_overlap_same_service_id() -> None:
    trips = _concat_trips(
        _make_trip("t0", "WEEK", "b1", 2),
        _make_trip("t1", "WEEK", "b1", 3),
        _make_trip("t2", "WEEK", "b1", 4),
    )
    stop_times = _concat_stop_times(
        _make_stop_times("t0", [(T_0800, T_0800), (T_0900, T_0900)]),
        _make_stop_times("t1", [(T_0830, T_0830), (T_0930, T_0930)]),
        _make_stop_times("t2", [(T_1200, T_1200), (T_1300, T_1300)]),
    )
    calendar = _make_calendar("WEEK", date(2021, 1, 4), date(2021, 1, 8))
    feed = {"trips": trips, "stop_times": stop_times, "calendar": calendar}
    notices = validate_block_trips_overlapping(feed, CTX)
    assert len(notices) == 1
    n = notices[0]
    assert n.code == "block_trips_with_overlapping_stop_times"
    assert n.severity == Severity.ERROR
    assert n.fields["block_id"] == "b1"
    assert n.fields["intersection"] == "2021-01-04"
    # Check that the pair is t0/t1
    pair_ids = {n.fields["trip_id_a"], n.fields["trip_id_b"]}
    assert pair_ids == {"t0", "t1"}


# ------------------------------------------------------------------
# Test 3: overlap with different service_id, shared dates
# ------------------------------------------------------------------


def test_overlap_different_service_id_shared_dates() -> None:
    trips = _concat_trips(
        _make_trip("t0", "WEEK", "b1", 2),
        _make_trip("t1", "WEEK-ALT", "b1", 3),
    )
    stop_times = _concat_stop_times(
        _make_stop_times("t0", [(T_0800, T_0800), (T_0900, T_0900)]),
        _make_stop_times("t1", [(T_0830, T_0830), (T_0930, T_0930)]),
    )
    calendar = _concat_calendars(
        _make_calendar("WEEK", date(2021, 1, 4), date(2021, 1, 8)),
        _make_calendar("WEEK-ALT", date(2021, 1, 4), date(2021, 1, 8)),
    )
    feed = {"trips": trips, "stop_times": stop_times, "calendar": calendar}
    notices = validate_block_trips_overlapping(feed, CTX)
    assert len(notices) == 1
    assert notices[0].fields["intersection"] == "2021-01-04"


# ------------------------------------------------------------------
# Test 4: overlap with different service_id, no shared dates
# ------------------------------------------------------------------


def test_overlap_different_service_id_no_shared_dates() -> None:
    trips = _concat_trips(
        _make_trip("t0", "WEEK", "b1", 2),
        _make_trip("t1", "SAT", "b1", 3),
    )
    stop_times = _concat_stop_times(
        _make_stop_times("t0", [(T_0800, T_0800), (T_0900, T_0900)]),
        _make_stop_times("t1", [(T_0830, T_0830), (T_0930, T_0930)]),
    )
    calendar = _concat_calendars(
        _make_calendar("WEEK", date(2021, 1, 4), date(2021, 1, 10), "1111100"),
        _make_calendar("SAT", date(2021, 1, 4), date(2021, 1, 10), "0000010"),
    )
    feed = {"trips": trips, "stop_times": stop_times, "calendar": calendar}
    notices = validate_block_trips_overlapping(feed, CTX)
    assert len(notices) == 0


# ------------------------------------------------------------------
# Test 5: trips with 0 or 1 stop
# ------------------------------------------------------------------


def test_trips_with_0_or_1_stop() -> None:
    trips = _concat_trips(
        _make_trip("t0", "WEEK", "b1", 2),
        _make_trip("t1", "WEEK", "b1", 3),
    )
    # t0 has 1 stop, t1 has no stops
    stop_times = _make_stop_times("t0", [(T_0800, T_0800)])
    calendar = _make_calendar("WEEK", date(2021, 1, 4), date(2021, 1, 8))
    feed = {"trips": trips, "stop_times": stop_times, "calendar": calendar}
    notices = validate_block_trips_overlapping(feed, CTX)
    assert len(notices) == 0


# ------------------------------------------------------------------
# Test 6: no block_id
# ------------------------------------------------------------------


def test_no_block_id() -> None:
    trips = _concat_trips(
        _make_trip("t0", "WEEK", None, 2),
        _make_trip("t1", "WEEK", "", 3),
    )
    stop_times = _concat_stop_times(
        _make_stop_times("t0", [(T_0800, T_0800), (T_0900, T_0900)]),
        _make_stop_times("t1", [(T_0830, T_0830), (T_0930, T_0930)]),
    )
    calendar = _make_calendar("WEEK", date(2021, 1, 4), date(2021, 1, 8))
    feed = {"trips": trips, "stop_times": stop_times, "calendar": calendar}
    notices = validate_block_trips_overlapping(feed, CTX)
    assert len(notices) == 0


# ------------------------------------------------------------------
# Test 7: block transfer exception
# ------------------------------------------------------------------


def test_block_transfer_exception() -> None:
    trips = _concat_trips(
        _make_trip("t0", "WEEK", "b1", 2),
        _make_trip("t1", "WEEK", "b1", 3),
    )
    # t0: 08:00-09:00, t1: 09:00-10:00
    # last_arrival of t0 == first_arrival of t1 == 32400
    # last_departure of t0 == first_departure of t1 == 32400
    stop_times = _concat_stop_times(
        _make_stop_times("t0", [(T_0800, T_0800), (T_0900, T_0900)]),
        _make_stop_times("t1", [(T_0900, T_0900), (T_1000, T_1000)]),
    )
    calendar = _make_calendar("WEEK", date(2021, 1, 4), date(2021, 1, 8))
    feed = {"trips": trips, "stop_times": stop_times, "calendar": calendar}
    notices = validate_block_trips_overlapping(feed, CTX)
    assert len(notices) == 0


# ------------------------------------------------------------------
# Test 8: touching intervals, no overlap
# ------------------------------------------------------------------


def test_touching_intervals_no_overlap() -> None:
    trips = _concat_trips(
        _make_trip("t0", "WEEK", "b1", 2),
        _make_trip("t1", "WEEK", "b1", 3),
    )
    # t0: 08:00-09:00, t1: 09:00-10:00
    # last_departure of t0 (09:00) == first_arrival of t1 (09:00)
    # but departure at boundary differs (not a block transfer pattern)
    # However, the condition is last_departure <= first_arrival, which is
    # 32400 <= 32400 = True, so no overlap detected.
    stop_times = _concat_stop_times(
        _make_stop_times("t0", [(T_0800, T_0800), (T_0900, T_0900)]),
        _make_stop_times("t1", [(T_0900, T_0930), (T_1000, T_1000)]),
    )
    calendar = _make_calendar("WEEK", date(2021, 1, 4), date(2021, 1, 8))
    feed = {"trips": trips, "stop_times": stop_times, "calendar": calendar}
    notices = validate_block_trips_overlapping(feed, CTX)
    assert len(notices) == 0


# ------------------------------------------------------------------
# Test 9: missing trips table
# ------------------------------------------------------------------


def test_missing_trips_table() -> None:
    stop_times = _make_stop_times("t0", [(T_0800, T_0800), (T_0900, T_0900)])
    feed = {"stop_times": stop_times}
    notices = validate_block_trips_overlapping(feed, CTX)
    assert notices == []


# ------------------------------------------------------------------
# Test 10: missing stop_times table
# ------------------------------------------------------------------


def test_missing_stop_times_table() -> None:
    trips = _make_trip("t0", "WEEK", "b1", 2)
    feed = {"trips": trips}
    notices = validate_block_trips_overlapping(feed, CTX)
    assert notices == []


# ------------------------------------------------------------------
# Test 11: empty trips
# ------------------------------------------------------------------


def test_empty_trips() -> None:
    trips = pl.DataFrame({
        "trip_id": [],
        "service_id": [],
        "block_id": [],
        "csv_row_number": [],
    }).cast({"csv_row_number": pl.Int64})
    stop_times = _make_stop_times("t0", [(T_0800, T_0800), (T_0900, T_0900)])
    feed = {"trips": trips, "stop_times": stop_times}
    notices = validate_block_trips_overlapping(feed, CTX)
    assert notices == []


# ------------------------------------------------------------------
# Test 12: missing arrival or departure time
# ------------------------------------------------------------------


def test_missing_arrival_or_departure() -> None:
    trips = _concat_trips(
        _make_trip("t0", "WEEK", "b1", 2),
        _make_trip("t1", "WEEK", "b1", 3),
    )
    # t0 has null arrival_time on first stop
    st_t0 = pl.DataFrame({
        "trip_id": ["t0", "t0"],
        "stop_sequence": [1, 2],
        "arrival_time": [None, T_0900],
        "departure_time": [T_0800, T_0900],
    })
    st_t1 = _make_stop_times("t1", [(T_0830, T_0830), (T_0930, T_0930)])
    stop_times = pl.concat([st_t0, st_t1])
    calendar = _make_calendar("WEEK", date(2021, 1, 4), date(2021, 1, 8))
    feed = {"trips": trips, "stop_times": stop_times, "calendar": calendar}
    notices = validate_block_trips_overlapping(feed, CTX)
    assert len(notices) == 0


# ------------------------------------------------------------------
# Test 13: multiple overlapping pairs in block
# ------------------------------------------------------------------


def test_multiple_overlapping_pairs_in_block() -> None:
    trips = _concat_trips(
        _make_trip("t0", "WEEK", "b1", 2),
        _make_trip("t1", "WEEK", "b1", 3),
        _make_trip("t2", "WEEK", "b1", 4),
    )
    # t0: 08:00-10:00, t1: 08:30-09:30, t2: 09:00-11:00
    stop_times = _concat_stop_times(
        _make_stop_times("t0", [(T_0800, T_0800), (T_1000, T_1000)]),
        _make_stop_times("t1", [(T_0830, T_0830), (T_0930, T_0930)]),
        _make_stop_times("t2", [(T_0900, T_0900), (T_1100, T_1100)]),
    )
    calendar = _make_calendar("WEEK", date(2021, 1, 4), date(2021, 1, 8))
    feed = {"trips": trips, "stop_times": stop_times, "calendar": calendar}
    notices = validate_block_trips_overlapping(feed, CTX)
    assert len(notices) == 3
    for n in notices:
        assert n.code == "block_trips_with_overlapping_stop_times"
        assert n.severity == Severity.ERROR


# ------------------------------------------------------------------
# Test 14: no calendar tables
# ------------------------------------------------------------------


def test_no_calendar_tables() -> None:
    trips = _concat_trips(
        _make_trip("t0", "WEEK", "b1", 2),
        _make_trip("t1", "WEEK", "b1", 3),
    )
    stop_times = _concat_stop_times(
        _make_stop_times("t0", [(T_0800, T_0800), (T_0900, T_0900)]),
        _make_stop_times("t1", [(T_0830, T_0830), (T_0930, T_0930)]),
    )
    feed = {"trips": trips, "stop_times": stop_times}
    notices = validate_block_trips_overlapping(feed, CTX)
    assert len(notices) == 0
