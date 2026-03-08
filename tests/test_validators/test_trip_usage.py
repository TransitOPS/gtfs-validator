"""Tests for validate_trip_usage."""

from __future__ import annotations

from datetime import date

import polars as pl

from gtfs_validator.context import ValidationContext
from gtfs_validator.notices import Severity
from gtfs_validator.validators.trip_usage import validate_trip_usage

CTX = ValidationContext(country_code="US", date_for_validation=date(2024, 1, 1))


def make_trips(
    trip_ids: list[str],
    csv_row_numbers: list[int] | None = None,
) -> dict[str, pl.DataFrame]:
    """Create a minimal trips DataFrame."""
    if csv_row_numbers is None:
        csv_row_numbers = list(range(1, len(trip_ids) + 1))
    return {
        "trips": pl.DataFrame(
            {
                "trip_id": trip_ids,
                "route_id": ["route_" + str(i) for i in range(len(trip_ids))],
                "service_id": ["service_" + str(i) for i in range(len(trip_ids))],
                "csv_row_number": csv_row_numbers,
            }
        )
    }


def make_stop_times(trip_ids: list[str]) -> dict[str, pl.DataFrame]:
    """Create a minimal stop_times DataFrame with entries for the given trip_ids."""
    if not trip_ids:
        return {
            "stop_times": pl.DataFrame(
                {
                    "trip_id": pl.Series([], dtype=pl.String),
                    "stop_id": pl.Series([], dtype=pl.String),
                    "stop_sequence": pl.Series([], dtype=pl.Int64),
                    "csv_row_number": pl.Series([], dtype=pl.Int64),
                }
            )
        }
    rows = []
    for i, trip_id in enumerate(trip_ids):
        rows.append({
            "trip_id": trip_id,
            "stop_id": f"stop_{i}",
            "stop_sequence": 1,
            "csv_row_number": i + 1,
        })
    return {
        "stop_times": pl.DataFrame(
            {
                "trip_id": [r["trip_id"] for r in rows],
                "stop_id": [r["stop_id"] for r in rows],
                "stop_sequence": [r["stop_sequence"] for r in rows],
                "csv_row_number": [r["csv_row_number"] for r in rows],
            }
        )
    }


def merge_feed(
    trips_feed: dict[str, pl.DataFrame],
    stop_times_feed: dict[str, pl.DataFrame],
) -> dict[str, pl.DataFrame]:
    """Merge trips and stop_times into a single feed dict."""
    feed = {}
    feed.update(trips_feed)
    feed.update(stop_times_feed)
    return feed


def test_all_trips_used_no_notice() -> None:
    """All trips have stop_times → no notices."""
    trips = make_trips(["t0", "t1"])
    stop_times = make_stop_times(["t0", "t1"])
    feed = merge_feed(trips, stop_times)
    notices = validate_trip_usage(feed, CTX)
    assert notices == []


def test_unused_trip_emits_notice() -> None:
    """One trip without stop_times → one notice for that trip."""
    trips = make_trips(["used_trip", "unused_trip"], [1, 3])
    stop_times = make_stop_times(["used_trip"])
    feed = merge_feed(trips, stop_times)
    notices = validate_trip_usage(feed, CTX)
    assert len(notices) == 1
    n = notices[0]
    assert n.code == "unused_trip"
    assert n.severity == Severity.WARNING
    assert n.fields["trip_id"] == "unused_trip"
    assert n.fields["csv_row_number"] == 3


def test_trips_absent_no_notice() -> None:
    """No trips key in feed → no notices."""
    feed: dict[str, pl.DataFrame] = {
        "stop_times": make_stop_times(["t0"])["stop_times"]
    }
    notices = validate_trip_usage(feed, CTX)
    assert notices == []


def test_trips_empty_no_notice() -> None:
    """Empty trips DataFrame → no notices."""
    feed = {
        "trips": pl.DataFrame(
            {
                "trip_id": pl.Series([], dtype=pl.String),
                "route_id": pl.Series([], dtype=pl.String),
                "service_id": pl.Series([], dtype=pl.String),
                "csv_row_number": pl.Series([], dtype=pl.Int64),
            }
        ),
        "stop_times": make_stop_times(["t0"])["stop_times"],
    }
    notices = validate_trip_usage(feed, CTX)
    assert notices == []


def test_stop_times_absent_no_notice() -> None:
    """No stop_times key → no notices (validator is dependency-gated in pipeline)."""
    trips = make_trips(["t0", "t1", "t2"])
    feed = trips  # no stop_times
    notices = validate_trip_usage(feed, CTX)
    assert notices == []


def test_stop_times_empty_all_trips_unused() -> None:
    """Empty stop_times DataFrame → all trips are unused."""
    trips = make_trips(["t0", "t1"])
    stop_times = make_stop_times([])  # empty
    feed = merge_feed(trips, stop_times)
    notices = validate_trip_usage(feed, CTX)
    assert len(notices) == 2
    trip_ids = {n.fields["trip_id"] for n in notices}
    assert trip_ids == {"t0", "t1"}


def test_multiple_unused_trips() -> None:
    """3 trips, only 1 has stop_times → 2 notices."""
    trips = make_trips(["used", "unused1", "unused2"])
    stop_times = make_stop_times(["used"])
    feed = merge_feed(trips, stop_times)
    notices = validate_trip_usage(feed, CTX)
    assert len(notices) == 2
    trip_ids = {n.fields["trip_id"] for n in notices}
    assert trip_ids == {"unused1", "unused2"}


def test_duplicate_trip_id_deduplicated() -> None:
    """Duplicate trip_id in trips → only one notice per unique trip_id."""
    trips = {
        "trips": pl.DataFrame(
            {
                "trip_id": ["dup_trip", "dup_trip", "unique_trip"],
                "route_id": ["r1", "r2", "r3"],
                "service_id": ["s1", "s2", "s3"],
                "csv_row_number": [2, 4, 6],
            }
        )
    }
    stop_times = make_stop_times(["unique_trip"])
    feed = merge_feed(trips, stop_times)
    notices = validate_trip_usage(feed, CTX)
    assert len(notices) == 1
    assert notices[0].fields["trip_id"] == "dup_trip"
    # First occurrence's csv_row_number is used
    assert notices[0].fields["csv_row_number"] == 2


def test_csv_row_number_preserved() -> None:
    """csv_row_number from the trip row is preserved in the notice."""
    trips = make_trips(["unused"], [42])
    stop_times = make_stop_times([])  # empty
    feed = merge_feed(trips, stop_times)
    notices = validate_trip_usage(feed, CTX)
    assert len(notices) == 1
    assert notices[0].fields["csv_row_number"] == 42


def test_notice_fields_are_complete() -> None:
    """Notice fields contain exactly trip_id and csv_row_number."""
    trips = make_trips(["unused"])
    stop_times = make_stop_times([])  # empty
    feed = merge_feed(trips, stop_times)
    notices = validate_trip_usage(feed, CTX)
    assert len(notices) == 1
    assert set(notices[0].fields.keys()) == {"trip_id", "csv_row_number"}


def test_ctx_unused() -> None:
    """ctx value has no effect on the output."""
    wrong_ctx = ValidationContext(
        country_code="XX",
        date_for_validation=date(1900, 1, 1),
    )
    trips = make_trips(["unused"])
    stop_times = make_stop_times([])  # empty
    feed = merge_feed(trips, stop_times)
    notices = validate_trip_usage(feed, wrong_ctx)
    assert len(notices) == 1
    assert notices[0].fields["trip_id"] == "unused"
