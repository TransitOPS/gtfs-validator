"""Tests for validate_timeframe_start_and_end_time."""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from gtfs_validator.context import ValidationContext
from gtfs_validator.notices import Severity
from gtfs_validator.validators.timeframe_start_and_end_time import (
    validate_timeframe_start_and_end_time,
)

CTX = ValidationContext(country_code="US", date_for_validation=date(2024, 1, 1))


def _feed(rows: list[dict]) -> dict[str, pl.DataFrame]:
    return {
        "timeframes": pl.DataFrame(
            rows,
            schema={"start_time": pl.Utf8, "end_time": pl.Utf8, "csv_row_number": pl.Int64},
        )
    }


def test_explicit_full_day_interval() -> None:
    feed = _feed([{"start_time": "00:00:00", "end_time": "24:00:00", "csv_row_number": 1}])
    assert validate_timeframe_start_and_end_time(feed, CTX) == []


def test_implicit_full_day_interval() -> None:
    feed = _feed([{"start_time": None, "end_time": None, "csv_row_number": 1}])
    assert validate_timeframe_start_and_end_time(feed, CTX) == []


def test_beyond_twenty_four_hours_end_time() -> None:
    feed = _feed([{"start_time": "00:00:00", "end_time": "24:00:01", "csv_row_number": 2}])
    notices = validate_timeframe_start_and_end_time(feed, CTX)
    assert len(notices) == 1
    n = notices[0]
    assert n.code == "timeframe_start_or_end_time_greater_than_twenty_four_hours"
    assert n.severity == Severity.ERROR
    assert n.fields == {"csv_row_number": 2, "field_name": "end_time", "time": "24:00:01"}


def test_only_start_time_specified() -> None:
    feed = _feed([{"start_time": "00:00:00", "end_time": None, "csv_row_number": 2}])
    notices = validate_timeframe_start_and_end_time(feed, CTX)
    assert len(notices) == 1
    n = notices[0]
    assert n.code == "timeframe_only_start_or_end_time_specified"
    assert n.severity == Severity.ERROR
    assert n.fields == {"csv_row_number": 2}


def test_only_end_time_specified() -> None:
    feed = _feed([{"start_time": None, "end_time": "10:00:00", "csv_row_number": 2}])
    notices = validate_timeframe_start_and_end_time(feed, CTX)
    assert len(notices) == 1
    n = notices[0]
    assert n.code == "timeframe_only_start_or_end_time_specified"
    assert n.severity == Severity.ERROR
    assert n.fields == {"csv_row_number": 2}


def test_timeframes_absent() -> None:
    assert validate_timeframe_start_and_end_time({}, CTX) == []


def test_timeframes_empty() -> None:
    feed = {
        "timeframes": pl.DataFrame(
            {"start_time": [], "end_time": [], "csv_row_number": []},
            schema={"start_time": pl.Utf8, "end_time": pl.Utf8, "csv_row_number": pl.Int64},
        )
    }
    assert validate_timeframe_start_and_end_time(feed, CTX) == []


def test_both_times_exceed_twenty_four_hours() -> None:
    feed = _feed([{"start_time": "25:00:00", "end_time": "26:00:00", "csv_row_number": 3}])
    notices = validate_timeframe_start_and_end_time(feed, CTX)
    assert len(notices) == 2
    assert notices[0].code == "timeframe_start_or_end_time_greater_than_twenty_four_hours"
    assert notices[0].fields == {"csv_row_number": 3, "field_name": "start_time", "time": "25:00:00"}
    assert notices[1].code == "timeframe_start_or_end_time_greater_than_twenty_four_hours"
    assert notices[1].fields == {"csv_row_number": 3, "field_name": "end_time", "time": "26:00:00"}


def test_xor_and_start_exceeds_limit() -> None:
    feed = _feed([{"start_time": "25:00:00", "end_time": None, "csv_row_number": 4}])
    notices = validate_timeframe_start_and_end_time(feed, CTX)
    assert len(notices) == 2
    assert notices[0].code == "timeframe_only_start_or_end_time_specified"
    assert notices[0].fields == {"csv_row_number": 4}
    assert notices[1].code == "timeframe_start_or_end_time_greater_than_twenty_four_hours"
    assert notices[1].fields == {"csv_row_number": 4, "field_name": "start_time", "time": "25:00:00"}


def test_xor_and_end_exceeds_limit() -> None:
    feed = _feed([{"start_time": None, "end_time": "25:00:00", "csv_row_number": 5}])
    notices = validate_timeframe_start_and_end_time(feed, CTX)
    assert len(notices) == 2
    assert notices[0].code == "timeframe_only_start_or_end_time_specified"
    assert notices[0].fields == {"csv_row_number": 5}
    assert notices[1].code == "timeframe_start_or_end_time_greater_than_twenty_four_hours"
    assert notices[1].fields == {"csv_row_number": 5, "field_name": "end_time", "time": "25:00:00"}


def test_boundary_exact_twenty_four_hours() -> None:
    feed = _feed([{"start_time": "00:00:00", "end_time": "24:00:00", "csv_row_number": 6}])
    assert validate_timeframe_start_and_end_time(feed, CTX) == []


def test_boundary_one_second_over() -> None:
    feed = _feed([{"start_time": "00:00:00", "end_time": "24:00:01", "csv_row_number": 7}])
    notices = validate_timeframe_start_and_end_time(feed, CTX)
    assert len(notices) == 1
    assert notices[0].code == "timeframe_start_or_end_time_greater_than_twenty_four_hours"
    assert notices[0].fields == {"csv_row_number": 7, "field_name": "end_time", "time": "24:00:01"}


def test_multiple_rows_mixed() -> None:
    feed = _feed([
        {"start_time": "00:00:00", "end_time": "12:00:00", "csv_row_number": 1},
        {"start_time": "12:00:00", "end_time": None, "csv_row_number": 2},
        {"start_time": "00:00:00", "end_time": "25:00:00", "csv_row_number": 3},
    ])
    notices = validate_timeframe_start_and_end_time(feed, CTX)
    assert len(notices) == 2
    assert notices[0].code == "timeframe_only_start_or_end_time_specified"
    assert notices[0].fields == {"csv_row_number": 2}
    assert notices[1].code == "timeframe_start_or_end_time_greater_than_twenty_four_hours"
    assert notices[1].fields == {"csv_row_number": 3, "field_name": "end_time", "time": "25:00:00"}


def test_single_digit_hour_time_string() -> None:
    feed = _feed([{"start_time": "9:00:00", "end_time": "10:00:00", "csv_row_number": 8}])
    assert validate_timeframe_start_and_end_time(feed, CTX) == []


def test_start_time_exceeds_limit_end_within_bounds() -> None:
    feed = _feed([{"start_time": "25:00:00", "end_time": "26:00:00", "csv_row_number": 9}])
    notices = validate_timeframe_start_and_end_time(feed, CTX)
    assert len(notices) == 2
    assert notices[0].code == "timeframe_start_or_end_time_greater_than_twenty_four_hours"
    assert notices[0].fields["field_name"] == "start_time"
    assert notices[1].code == "timeframe_start_or_end_time_greater_than_twenty_four_hours"
    assert notices[1].fields["field_name"] == "end_time"
