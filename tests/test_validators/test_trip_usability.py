"""Tests for validate_trip_usability."""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from gtfs_validator.context import ValidationContext
from gtfs_validator.notices import Severity
from gtfs_validator.validators.trip_usability import validate_trip_usability

CTX = ValidationContext(country_code="US", date_for_validation=date(2024, 1, 1))


def make_trips(
    trip_ids: list[str],
    csv_row_numbers: list[int] | None = None,
) -> dict[str, pl.DataFrame]:
    """Return a feed dict with a minimal trips DataFrame."""
    if csv_row_numbers is None:
        csv_row_numbers = list(range(1, len(trip_ids) + 1))
    return {
        "trips": pl.DataFrame(
            {
                "trip_id": trip_ids,
                "csv_row_number": csv_row_numbers,
            }
        )
    }


def make_stop_times(rows: list[tuple[str, int]]) -> dict[str, pl.DataFrame]:
    """Return a feed dict with a minimal stop_times DataFrame.

    rows = [(trip_id, stop_sequence), ...]
    """
    if not rows:
        return {
            "stop_times": pl.DataFrame(
                {
                    "trip_id": pl.Series([], dtype=pl.String),
                    "stop_sequence": pl.Series([], dtype=pl.Int64),
                }
            )
        }
    return {
        "stop_times": pl.DataFrame(
            {
                "trip_id": [r[0] for r in rows],
                "stop_sequence": [r[1] for r in rows],
            }
        )
    }


def make_feed(**tables: dict[str, pl.DataFrame]) -> dict[str, pl.DataFrame]:
    """Merge multiple table dicts into a single feed dict."""
    feed: dict[str, pl.DataFrame] = {}
    for table_dict in tables.values():
        feed.update(table_dict)
    return feed


# ---------------------------------------------------------------------------
# Tests derived from Java contracts
# ---------------------------------------------------------------------------


def test_two_stop_times_per_trip_no_notice() -> None:
    """Mirrors Java tripServingMoreThanOneStopShouldNotGenerateNotice."""
    trips = make_trips(["t0", "t1"], csv_row_numbers=[1, 3])
    stop_times = make_stop_times([("t0", 2), ("t0", 3), ("t1", 5), ("t1", 9)])
    feed = make_feed(trips=trips, stop_times=stop_times)
    assert validate_trip_usability(feed, CTX) == []


def test_one_stop_time_emits_notice() -> None:
    """Mirrors Java tripServingOneStopShouldGenerateNotice."""
    trips = make_trips(["t0", "t1"], csv_row_numbers=[1, 3])
    stop_times = make_stop_times([("t0", 2), ("t1", 5), ("t1", 9)])
    feed = make_feed(trips=trips, stop_times=stop_times)
    notices = validate_trip_usability(feed, CTX)
    assert len(notices) == 1
    assert notices[0].code == "unusable_trip"
    assert notices[0].severity == Severity.WARNING
    assert notices[0].fields["trip_id"] == "t0"
    assert notices[0].fields["csv_row_number"] == 1


# ---------------------------------------------------------------------------
# Edge-case tests
# ---------------------------------------------------------------------------


def test_trips_absent_returns_empty() -> None:
    stop_times = make_stop_times([("t0", 1)])
    assert validate_trip_usability(stop_times, CTX) == []


def test_trips_empty_returns_empty() -> None:
    trips = make_trips([])
    stop_times = make_stop_times([("t0", 1)])
    feed = make_feed(trips=trips, stop_times=stop_times)
    assert validate_trip_usability(feed, CTX) == []


def test_stop_times_absent_returns_empty() -> None:
    trips = make_trips(["t0", "t1"])
    assert validate_trip_usability(trips, CTX) == []


def test_stop_times_empty_all_trips_emit_notice() -> None:
    trips = make_trips(["t0", "t1"], csv_row_numbers=[2, 4])
    stop_times = make_stop_times([])
    feed = make_feed(trips=trips, stop_times=stop_times)
    notices = validate_trip_usability(feed, CTX)
    assert len(notices) == 2
    assert {n.fields["trip_id"] for n in notices} == {"t0", "t1"}


def test_zero_stop_times_emits_notice() -> None:
    trips = make_trips(["t0"], csv_row_numbers=[1])
    stop_times = make_stop_times([])
    feed = make_feed(trips=trips, stop_times=stop_times)
    notices = validate_trip_usability(feed, CTX)
    assert len(notices) == 1
    assert notices[0].fields["trip_id"] == "t0"


def test_exactly_two_stop_times_no_notice() -> None:
    trips = make_trips(["t0"], csv_row_numbers=[1])
    stop_times = make_stop_times([("t0", 1), ("t0", 2)])
    feed = make_feed(trips=trips, stop_times=stop_times)
    assert validate_trip_usability(feed, CTX) == []


def test_multiple_violating_trips_each_emits_notice() -> None:
    trips = make_trips(["t0", "t1", "t2"], csv_row_numbers=[1, 2, 3])
    stop_times = make_stop_times([("t0", 1), ("t2", 1), ("t2", 2)])
    feed = make_feed(trips=trips, stop_times=stop_times)
    notices = validate_trip_usability(feed, CTX)
    assert len(notices) == 2
    violating_ids = {n.fields["trip_id"] for n in notices}
    assert violating_ids == {"t0", "t1"}


def test_orphaned_stop_times_ignored() -> None:
    """trip_ids in stop_times but not in trips must not affect validation."""
    trips = make_trips(["t0"], csv_row_numbers=[1])
    # "orphan" has 5 stop times but is not in trips; t0 has only 1
    stop_times = make_stop_times(
        [("orphan", 1), ("orphan", 2), ("orphan", 3), ("orphan", 4), ("orphan", 5), ("t0", 1)]
    )
    feed = make_feed(trips=trips, stop_times=stop_times)
    notices = validate_trip_usability(feed, CTX)
    assert len(notices) == 1
    assert notices[0].fields["trip_id"] == "t0"


def test_csv_row_number_preserved() -> None:
    trips = make_trips(["t0"], csv_row_numbers=[42])
    stop_times = make_stop_times([("t0", 1)])
    feed = make_feed(trips=trips, stop_times=stop_times)
    notices = validate_trip_usability(feed, CTX)
    assert len(notices) == 1
    assert notices[0].fields["csv_row_number"] == 42


def test_notice_fields_are_exactly_two_keys() -> None:
    trips = make_trips(["t0"], csv_row_numbers=[1])
    stop_times = make_stop_times([("t0", 1)])
    feed = make_feed(trips=trips, stop_times=stop_times)
    notices = validate_trip_usability(feed, CTX)
    assert len(notices) == 1
    assert set(notices[0].fields.keys()) == {"trip_id", "csv_row_number"}


def test_ctx_is_unused() -> None:
    trips = make_trips(["t0"], csv_row_numbers=[1])
    stop_times = make_stop_times([("t0", 1)])
    feed = make_feed(trips=trips, stop_times=stop_times)
    ctx_other = ValidationContext(country_code="DE", date_for_validation=date(1900, 1, 1))
    assert validate_trip_usability(feed, CTX) == validate_trip_usability(feed, ctx_other)
