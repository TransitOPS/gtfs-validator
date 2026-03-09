"""Tests for OverlappingFrequencyValidator."""

from datetime import date

import polars as pl

from gtfs_validator.context import ValidationContext
from gtfs_validator.notices import Severity
from gtfs_validator.validators.overlapping_frequency import validate_overlapping_frequency

CTX = ValidationContext(country_code="US", date_for_validation=date(2026, 3, 8))


def make_frequencies(rows: list[dict]) -> pl.DataFrame:
    """Build a minimal frequencies DataFrame from a list of row dicts.

    Each dict must supply: trip_id, start_time, end_time, headway_secs, csv_row_number.
    """
    return pl.DataFrame(
        {
            "trip_id": [r["trip_id"] for r in rows],
            "start_time": [r["start_time"] for r in rows],
            "end_time": [r["end_time"] for r in rows],
            "headway_secs": [r["headway_secs"] for r in rows],
            "csv_row_number": [r["csv_row_number"] for r in rows],
        },
        schema={
            "trip_id": pl.Utf8,
            "start_time": pl.Utf8,
            "end_time": pl.Utf8,
            "headway_secs": pl.Int64,
            "csv_row_number": pl.Int64,
        },
    )


def test_valid_sequential_in_order() -> None:
    feed = {"frequencies": make_frequencies([
        {"trip_id": "t0", "start_time": "05:00:00", "end_time": "07:00:00", "headway_secs": 300, "csv_row_number": 2},
        {"trip_id": "t0", "start_time": "07:00:00", "end_time": "10:00:00", "headway_secs": 300, "csv_row_number": 3},
    ])}
    assert validate_overlapping_frequency(feed, CTX) == []


def test_valid_sequential_reversed() -> None:
    feed = {"frequencies": make_frequencies([
        {"trip_id": "t0", "start_time": "07:00:00", "end_time": "10:00:00", "headway_secs": 300, "csv_row_number": 2},
        {"trip_id": "t0", "start_time": "05:00:00", "end_time": "07:00:00", "headway_secs": 300, "csv_row_number": 3},
    ])}
    assert validate_overlapping_frequency(feed, CTX) == []


def test_valid_with_gap() -> None:
    feed = {"frequencies": make_frequencies([
        {"trip_id": "t0", "start_time": "05:00:00", "end_time": "07:00:00", "headway_secs": 300, "csv_row_number": 2},
        {"trip_id": "t0", "start_time": "08:00:00", "end_time": "10:00:00", "headway_secs": 300, "csv_row_number": 3},
    ])}
    assert validate_overlapping_frequency(feed, CTX) == []


def test_valid_different_trips() -> None:
    feed = {"frequencies": make_frequencies([
        {"trip_id": "t0", "start_time": "05:00:00", "end_time": "07:00:00", "headway_secs": 300, "csv_row_number": 2},
        {"trip_id": "t1", "start_time": "06:00:00", "end_time": "10:00:00", "headway_secs": 300, "csv_row_number": 3},
    ])}
    assert validate_overlapping_frequency(feed, CTX) == []


def test_overlapping_partially() -> None:
    feed = {"frequencies": make_frequencies([
        {"trip_id": "t0", "start_time": "05:00:00", "end_time": "07:00:00", "headway_secs": 300, "csv_row_number": 2},
        {"trip_id": "t0", "start_time": "06:00:00", "end_time": "10:00:00", "headway_secs": 300, "csv_row_number": 3},
    ])}
    notices = validate_overlapping_frequency(feed, CTX)
    assert len(notices) == 1
    n = notices[0]
    assert n.code == "overlapping_frequency"
    assert n.severity == Severity.ERROR
    assert n.fields["prev_csv_row_number"] == 2
    assert n.fields["prev_end_time"] == "07:00:00"
    assert n.fields["curr_csv_row_number"] == 3
    assert n.fields["curr_start_time"] == "06:00:00"
    assert n.fields["trip_id"] == "t0"


def test_overlapping_included_same_start() -> None:
    feed = {"frequencies": make_frequencies([
        {"trip_id": "t0", "start_time": "05:00:00", "end_time": "07:00:00", "headway_secs": 600, "csv_row_number": 2},
        {"trip_id": "t0", "start_time": "05:00:00", "end_time": "06:30:00", "headway_secs": 300, "csv_row_number": 3},
    ])}
    notices = validate_overlapping_frequency(feed, CTX)
    assert len(notices) == 1
    n = notices[0]
    assert n.code == "overlapping_frequency"
    assert n.severity == Severity.ERROR
    assert n.fields["prev_csv_row_number"] == 3
    assert n.fields["prev_end_time"] == "06:30:00"
    assert n.fields["curr_csv_row_number"] == 2
    assert n.fields["curr_start_time"] == "05:00:00"
    assert n.fields["trip_id"] == "t0"


def test_overlapping_included_same_end() -> None:
    feed = {"frequencies": make_frequencies([
        {"trip_id": "t0", "start_time": "05:00:00", "end_time": "07:00:00", "headway_secs": 300, "csv_row_number": 2},
        {"trip_id": "t0", "start_time": "06:30:00", "end_time": "07:00:00", "headway_secs": 300, "csv_row_number": 3},
    ])}
    notices = validate_overlapping_frequency(feed, CTX)
    assert len(notices) == 1
    n = notices[0]
    assert n.code == "overlapping_frequency"
    assert n.severity == Severity.ERROR
    assert n.fields["prev_csv_row_number"] == 2
    assert n.fields["prev_end_time"] == "07:00:00"
    assert n.fields["curr_csv_row_number"] == 3
    assert n.fields["curr_start_time"] == "06:30:00"
    assert n.fields["trip_id"] == "t0"


def test_overlapping_included() -> None:
    feed = {"frequencies": make_frequencies([
        {"trip_id": "t0", "start_time": "07:00:00", "end_time": "12:00:00", "headway_secs": 300, "csv_row_number": 2},
        {"trip_id": "t0", "start_time": "08:00:00", "end_time": "11:00:00", "headway_secs": 300, "csv_row_number": 3},
    ])}
    notices = validate_overlapping_frequency(feed, CTX)
    assert len(notices) == 1
    n = notices[0]
    assert n.code == "overlapping_frequency"
    assert n.severity == Severity.ERROR
    assert n.fields["prev_csv_row_number"] == 2
    assert n.fields["prev_end_time"] == "12:00:00"
    assert n.fields["curr_csv_row_number"] == 3
    assert n.fields["curr_start_time"] == "08:00:00"
    assert n.fields["trip_id"] == "t0"


def test_overlapping_three_intervals() -> None:
    feed = {"frequencies": make_frequencies([
        {"trip_id": "t0", "start_time": "05:00:00", "end_time": "05:25:00", "headway_secs": 600, "csv_row_number": 2},
        {"trip_id": "t0", "start_time": "05:00:00", "end_time": "05:15:00", "headway_secs": 300, "csv_row_number": 3},
        {"trip_id": "t0", "start_time": "05:20:00", "end_time": "05:40:00", "headway_secs": 300, "csv_row_number": 4},
    ])}
    notices = validate_overlapping_frequency(feed, CTX)
    assert len(notices) == 2

    # Pair 1: row3 (prev) vs row2 (curr)
    n1 = next(n for n in notices if n.fields["prev_csv_row_number"] == 3)
    assert n1.code == "overlapping_frequency"
    assert n1.severity == Severity.ERROR
    assert n1.fields["prev_end_time"] == "05:15:00"
    assert n1.fields["curr_csv_row_number"] == 2
    assert n1.fields["curr_start_time"] == "05:00:00"
    assert n1.fields["trip_id"] == "t0"

    # Pair 2: row2 (prev) vs row4 (curr)
    n2 = next(n for n in notices if n.fields["prev_csv_row_number"] == 2)
    assert n2.code == "overlapping_frequency"
    assert n2.severity == Severity.ERROR
    assert n2.fields["prev_end_time"] == "05:25:00"
    assert n2.fields["curr_csv_row_number"] == 4
    assert n2.fields["curr_start_time"] == "05:20:00"
    assert n2.fields["trip_id"] == "t0"


def test_frequencies_absent() -> None:
    assert validate_overlapping_frequency({}, CTX) == []


def test_frequencies_empty() -> None:
    empty = pl.DataFrame(
        schema={
            "trip_id": pl.Utf8,
            "start_time": pl.Utf8,
            "end_time": pl.Utf8,
            "headway_secs": pl.Int64,
            "csv_row_number": pl.Int64,
        }
    )
    assert validate_overlapping_frequency({"frequencies": empty}, CTX) == []


def test_single_entry_per_trip() -> None:
    feed = {"frequencies": make_frequencies([
        {"trip_id": "t0", "start_time": "05:00:00", "end_time": "07:00:00", "headway_secs": 300, "csv_row_number": 2},
    ])}
    assert validate_overlapping_frequency(feed, CTX) == []


def test_null_times_skipped_gracefully() -> None:
    df = pl.DataFrame(
        {
            "trip_id": ["t0", "t0"],
            "start_time": [None, "05:00:00"],
            "end_time": ["07:00:00", "07:00:00"],
            "headway_secs": [300, 300],
            "csv_row_number": [2, 3],
        },
        schema={
            "trip_id": pl.Utf8,
            "start_time": pl.Utf8,
            "end_time": pl.Utf8,
            "headway_secs": pl.Int64,
            "csv_row_number": pl.Int64,
        },
    )
    assert validate_overlapping_frequency({"frequencies": df}, CTX) == []


def test_times_exceeding_24_hours() -> None:
    feed = {"frequencies": make_frequencies([
        {"trip_id": "t0", "start_time": "25:00:00", "end_time": "26:00:00", "headway_secs": 300, "csv_row_number": 2},
        {"trip_id": "t0", "start_time": "25:30:00", "end_time": "27:00:00", "headway_secs": 300, "csv_row_number": 3},
    ])}
    notices = validate_overlapping_frequency(feed, CTX)
    assert len(notices) == 1
    n = notices[0]
    assert n.fields["prev_end_time"] == "26:00:00"
    assert n.fields["curr_start_time"] == "25:30:00"
