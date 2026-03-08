"""Tests for TimeframeOverlapValidator."""

from datetime import date

import polars as pl
import pytest

from gtfs_validator.context import ValidationContext
from gtfs_validator.notices import Severity
from gtfs_validator.validators.timeframe_overlap import validate_timeframe_overlap

CTX = ValidationContext(country_code="US", date_for_validation=date(2024, 1, 1))

TIMEFRAMES_SCHEMA = {
    "timeframe_group_id": pl.Utf8,
    "service_id": pl.Utf8,
    "start_time": pl.Utf8,
    "end_time": pl.Utf8,
    "csv_row_number": pl.Int64,
}


def make_timeframes(rows: list[dict]) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame(schema=TIMEFRAMES_SCHEMA)
    return pl.DataFrame(rows).cast(TIMEFRAMES_SCHEMA)


def test_timeframes_absent_no_notice():
    """No timeframes key in feed returns empty notices."""
    assert validate_timeframe_overlap({}, CTX) == []


def test_timeframes_empty_no_notice():
    """timeframes.txt present but empty returns no notices."""
    feed = {"timeframes": make_timeframes([])}
    assert validate_timeframe_overlap(feed, CTX) == []


def test_single_timeframe_no_notice():
    """A single timeframe entry cannot overlap with itself."""
    feed = {
        "timeframes": make_timeframes([
            {
                "timeframe_group_id": "PEAK",
                "service_id": "WEEKDAY",
                "start_time": "00:00:00",
                "end_time": "24:00:00",
                "csv_row_number": 2,
            }
        ])
    }
    assert validate_timeframe_overlap(feed, CTX) == []


def test_no_overlap_gap_between_intervals_no_notice():
    """Two non-overlapping intervals with a gap emit no notice."""
    feed = {
        "timeframes": make_timeframes([
            {
                "timeframe_group_id": "PEAK",
                "service_id": "WEEKDAY",
                "start_time": "08:00:00",
                "end_time": "09:00:00",
                "csv_row_number": 2,
            },
            {
                "timeframe_group_id": "PEAK",
                "service_id": "WEEKDAY",
                "start_time": "17:00:00",
                "end_time": "18:00:00",
                "csv_row_number": 3,
            },
        ])
    }
    assert validate_timeframe_overlap(feed, CTX) == []


def test_no_overlap_adjacent_intervals_no_notice():
    """Adjacent intervals (curr.start == prev.end) do not overlap."""
    feed = {
        "timeframes": make_timeframes([
            {
                "timeframe_group_id": "PEAK",
                "service_id": "WEEKDAY",
                "start_time": "08:00:00",
                "end_time": "09:00:00",
                "csv_row_number": 2,
            },
            {
                "timeframe_group_id": "PEAK",
                "service_id": "WEEKDAY",
                "start_time": "09:00:00",
                "end_time": "10:00:00",
                "csv_row_number": 3,
            },
        ])
    }
    assert validate_timeframe_overlap(feed, CTX) == []


def test_overlap_emits_notice():
    """Two overlapping intervals emit one notice with correct fields."""
    feed = {
        "timeframes": make_timeframes([
            {
                "timeframe_group_id": "PEAK",
                "service_id": "WEEKDAY",
                "start_time": "08:00:00",
                "end_time": "09:00:00",
                "csv_row_number": 2,
            },
            {
                "timeframe_group_id": "PEAK",
                "service_id": "WEEKDAY",
                "start_time": "08:30:00",
                "end_time": "09:30:00",
                "csv_row_number": 3,
            },
        ])
    }
    notices = validate_timeframe_overlap(feed, CTX)
    assert len(notices) == 1
    n = notices[0]
    assert n.code == "timeframe_overlap"
    assert n.severity == Severity.ERROR
    assert n.fields["prev_csv_row_number"] == 2
    assert n.fields["prev_end_time"] == "09:00:00"
    assert n.fields["curr_csv_row_number"] == 3
    assert n.fields["curr_start_time"] == "08:30:00"
    assert n.fields["timeframe_group_id"] == "PEAK"
    assert n.fields["service_id"] == "WEEKDAY"


def test_different_service_ids_no_notice():
    """Same group_id but different service_id rows are in separate groups."""
    feed = {
        "timeframes": make_timeframes([
            {
                "timeframe_group_id": "PEAK",
                "service_id": "WEEKDAY",
                "start_time": "08:00:00",
                "end_time": "09:00:00",
                "csv_row_number": 2,
            },
            {
                "timeframe_group_id": "PEAK",
                "service_id": "WEEKEND",
                "start_time": "08:00:00",
                "end_time": "09:00:00",
                "csv_row_number": 3,
            },
        ])
    }
    assert validate_timeframe_overlap(feed, CTX) == []


def test_different_group_ids_no_notice():
    """Same service_id but different timeframe_group_id rows are in separate groups."""
    feed = {
        "timeframes": make_timeframes([
            {
                "timeframe_group_id": "PEAK",
                "service_id": "WEEKDAY",
                "start_time": "08:00:00",
                "end_time": "09:00:00",
                "csv_row_number": 2,
            },
            {
                "timeframe_group_id": "NON-PEAK",
                "service_id": "WEEKDAY",
                "start_time": "08:00:00",
                "end_time": "09:00:00",
                "csv_row_number": 3,
            },
        ])
    }
    assert validate_timeframe_overlap(feed, CTX) == []


def test_null_start_time_rows_skipped():
    """Rows with null start_time are filtered before grouping."""
    feed = {
        "timeframes": make_timeframes([
            {
                "timeframe_group_id": "PEAK",
                "service_id": "WEEKDAY",
                "start_time": None,
                "end_time": "09:00:00",
                "csv_row_number": 2,
            },
            {
                "timeframe_group_id": "PEAK",
                "service_id": "WEEKDAY",
                "start_time": "08:30:00",
                "end_time": "09:30:00",
                "csv_row_number": 3,
            },
        ])
    }
    assert validate_timeframe_overlap(feed, CTX) == []


def test_null_end_time_rows_skipped():
    """Rows with null end_time are filtered before grouping."""
    feed = {
        "timeframes": make_timeframes([
            {
                "timeframe_group_id": "PEAK",
                "service_id": "WEEKDAY",
                "start_time": "08:00:00",
                "end_time": None,
                "csv_row_number": 2,
            },
            {
                "timeframe_group_id": "PEAK",
                "service_id": "WEEKDAY",
                "start_time": "08:30:00",
                "end_time": "09:30:00",
                "csv_row_number": 3,
            },
        ])
    }
    assert validate_timeframe_overlap(feed, CTX) == []


def test_equal_start_times_tie_break_by_end_time():
    """Equal start_time rows are sorted by end_time; the shorter interval is prev."""
    feed = {
        "timeframes": make_timeframes([
            {
                "timeframe_group_id": "PEAK",
                "service_id": "WEEKDAY",
                "start_time": "08:00:00",
                "end_time": "09:00:00",
                "csv_row_number": 2,
            },
            {
                "timeframe_group_id": "PEAK",
                "service_id": "WEEKDAY",
                "start_time": "08:00:00",
                "end_time": "10:00:00",
                "csv_row_number": 3,
            },
        ])
    }
    notices = validate_timeframe_overlap(feed, CTX)
    assert len(notices) == 1
    assert notices[0].fields["prev_csv_row_number"] == 2
    assert notices[0].fields["curr_csv_row_number"] == 3


def test_three_rows_consecutive_overlaps():
    """Three overlapping rows produce two consecutive notices."""
    feed = {
        "timeframes": make_timeframes([
            {
                "timeframe_group_id": "PEAK",
                "service_id": "WEEKDAY",
                "start_time": "08:00:00",
                "end_time": "10:00:00",
                "csv_row_number": 2,
            },
            {
                "timeframe_group_id": "PEAK",
                "service_id": "WEEKDAY",
                "start_time": "09:00:00",
                "end_time": "11:00:00",
                "csv_row_number": 3,
            },
            {
                "timeframe_group_id": "PEAK",
                "service_id": "WEEKDAY",
                "start_time": "10:30:00",
                "end_time": "12:00:00",
                "csv_row_number": 4,
            },
        ])
    }
    notices = validate_timeframe_overlap(feed, CTX)
    assert len(notices) == 2


def test_multiple_overlapping_pairs_same_group():
    """Three rows where only the first consecutive pair overlaps."""
    feed = {
        "timeframes": make_timeframes([
            {
                "timeframe_group_id": "PEAK",
                "service_id": "WEEKDAY",
                "start_time": "08:00:00",
                "end_time": "09:30:00",
                "csv_row_number": 2,
            },
            {
                "timeframe_group_id": "PEAK",
                "service_id": "WEEKDAY",
                "start_time": "09:00:00",
                "end_time": "10:00:00",
                "csv_row_number": 3,
            },
            {
                "timeframe_group_id": "PEAK",
                "service_id": "WEEKDAY",
                "start_time": "11:00:00",
                "end_time": "12:00:00",
                "csv_row_number": 4,
            },
        ])
    }
    notices = validate_timeframe_overlap(feed, CTX)
    assert len(notices) == 1
    assert notices[0].fields["prev_csv_row_number"] == 2
    assert notices[0].fields["curr_csv_row_number"] == 3


def test_multiple_independent_groups_each_overlapping():
    """Two independent groups each contributing one overlap notice."""
    feed = {
        "timeframes": make_timeframes([
            {
                "timeframe_group_id": "PEAK",
                "service_id": "WEEKDAY",
                "start_time": "08:00:00",
                "end_time": "09:30:00",
                "csv_row_number": 2,
            },
            {
                "timeframe_group_id": "PEAK",
                "service_id": "WEEKDAY",
                "start_time": "09:00:00",
                "end_time": "10:00:00",
                "csv_row_number": 3,
            },
            {
                "timeframe_group_id": "OFF-PEAK",
                "service_id": "WEEKDAY",
                "start_time": "14:00:00",
                "end_time": "16:00:00",
                "csv_row_number": 4,
            },
            {
                "timeframe_group_id": "OFF-PEAK",
                "service_id": "WEEKDAY",
                "start_time": "15:00:00",
                "end_time": "17:00:00",
                "csv_row_number": 5,
            },
        ])
    }
    notices = validate_timeframe_overlap(feed, CTX)
    assert len(notices) == 2
    group_ids = {n.fields["timeframe_group_id"] for n in notices}
    assert "PEAK" in group_ids
    assert "OFF-PEAK" in group_ids


def test_times_past_midnight_overlap():
    """Times >= 24:00:00 are handled correctly via integer-seconds conversion."""
    feed = {
        "timeframes": make_timeframes([
            {
                "timeframe_group_id": "NIGHT",
                "service_id": "WEEKDAY",
                "start_time": "24:00:00",
                "end_time": "25:00:00",
                "csv_row_number": 2,
            },
            {
                "timeframe_group_id": "NIGHT",
                "service_id": "WEEKDAY",
                "start_time": "24:30:00",
                "end_time": "26:00:00",
                "csv_row_number": 3,
            },
        ])
    }
    notices = validate_timeframe_overlap(feed, CTX)
    assert len(notices) == 1
    assert notices[0].fields["prev_csv_row_number"] == 2
    assert notices[0].fields["curr_csv_row_number"] == 3
    assert notices[0].fields["prev_end_time"] == "25:00:00"
    assert notices[0].fields["curr_start_time"] == "24:30:00"
