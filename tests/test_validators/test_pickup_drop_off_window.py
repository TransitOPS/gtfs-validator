"""Tests for PickupDropOffWindowValidator."""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from gtfs_validator.context import ValidationContext
from gtfs_validator.notices import Severity
from gtfs_validator.validators.pickup_drop_off_window import validate_pickup_drop_off_window

_COLS = {
    "csv_row_number": pl.Int64,
    "arrival_time": pl.Int64,
    "departure_time": pl.Int64,
    "start_pickup_drop_off_window": pl.Int64,
    "end_pickup_drop_off_window": pl.Int64,
}

_CTX = ValidationContext(country_code="US", date_for_validation=date(2024, 1, 1))


def make_stop_times(**overrides: object) -> pl.DataFrame:
    """Build a 1-row DataFrame with all five columns, all values defaulting to None."""
    defaults: dict[str, object] = {
        "csv_row_number": None,
        "arrival_time": None,
        "departure_time": None,
        "start_pickup_drop_off_window": None,
        "end_pickup_drop_off_window": None,
    }
    defaults.update(overrides)
    return pl.DataFrame(
        {col: [defaults[col]] for col in _COLS},
        schema=_COLS,
    )


def make_feed(stop_times_df: pl.DataFrame) -> dict[str, pl.DataFrame]:
    return {"stop_times": stop_times_df}


# ---------------------------------------------------------------------------
# Check A: forbidden_arrival_or_departure_time
# ---------------------------------------------------------------------------


def test_forbidden_arrival_or_departure_time_generates_notice() -> None:
    df = make_stop_times(
        csv_row_number=1,
        arrival_time=0,
        departure_time=1,
        start_pickup_drop_off_window=2,
        end_pickup_drop_off_window=3,
    )
    notices = validate_pickup_drop_off_window(make_feed(df), _CTX)
    assert len(notices) == 1
    n = notices[0]
    assert n.code == "forbidden_arrival_or_departure_time"
    assert n.severity == Severity.ERROR
    assert n.fields["csv_row_number"] == 1
    assert n.fields["arrival_time"] == 0
    assert n.fields["departure_time"] == 1
    assert n.fields["start_pickup_drop_off_window"] == 2
    assert n.fields["end_pickup_drop_off_window"] == 3


def test_only_departure_time_triggers_check_a() -> None:
    df = make_stop_times(
        csv_row_number=6,
        departure_time=60,
        start_pickup_drop_off_window=120,
        end_pickup_drop_off_window=180,
    )
    notices = validate_pickup_drop_off_window(make_feed(df), _CTX)
    assert len(notices) == 1
    n = notices[0]
    assert n.code == "forbidden_arrival_or_departure_time"
    assert n.fields["arrival_time"] is None
    assert n.fields["departure_time"] == 60


def test_only_arrival_time_triggers_check_a() -> None:
    df = make_stop_times(
        csv_row_number=7,
        arrival_time=30,
        start_pickup_drop_off_window=120,
        end_pickup_drop_off_window=180,
    )
    notices = validate_pickup_drop_off_window(make_feed(df), _CTX)
    assert len(notices) == 1
    n = notices[0]
    assert n.code == "forbidden_arrival_or_departure_time"
    assert n.fields["arrival_time"] == 30
    assert n.fields["departure_time"] is None


# ---------------------------------------------------------------------------
# Check B: missing_pickup_or_drop_off_window
# ---------------------------------------------------------------------------


def test_missing_start_window_generates_notice() -> None:
    df = make_stop_times(
        csv_row_number=1,
        end_pickup_drop_off_window=3,
    )
    notices = validate_pickup_drop_off_window(make_feed(df), _CTX)
    assert len(notices) == 1
    n = notices[0]
    assert n.code == "missing_pickup_or_drop_off_window"
    assert n.severity == Severity.ERROR
    assert n.fields["csv_row_number"] == 1
    assert n.fields["start_pickup_drop_off_window"] is None
    assert n.fields["end_pickup_drop_off_window"] == 3


def test_missing_end_window_generates_notice() -> None:
    df = make_stop_times(
        csv_row_number=1,
        start_pickup_drop_off_window=3,
    )
    notices = validate_pickup_drop_off_window(make_feed(df), _CTX)
    assert len(notices) == 1
    n = notices[0]
    assert n.code == "missing_pickup_or_drop_off_window"
    assert n.severity == Severity.ERROR
    assert n.fields["csv_row_number"] == 1
    assert n.fields["start_pickup_drop_off_window"] == 3
    assert n.fields["end_pickup_drop_off_window"] is None


# ---------------------------------------------------------------------------
# Check C: invalid_pickup_drop_off_window
# ---------------------------------------------------------------------------


def test_end_before_start_generates_invalid_window_notice() -> None:
    df = make_stop_times(
        csv_row_number=1,
        start_pickup_drop_off_window=3,
        end_pickup_drop_off_window=2,
    )
    notices = validate_pickup_drop_off_window(make_feed(df), _CTX)
    assert len(notices) == 1
    n = notices[0]
    assert n.code == "invalid_pickup_drop_off_window"
    assert n.severity == Severity.ERROR
    assert n.fields["csv_row_number"] == 1
    assert n.fields["start_pickup_drop_off_window"] == 3
    assert n.fields["end_pickup_drop_off_window"] == 2


def test_end_equal_to_start_generates_invalid_window_notice() -> None:
    df = make_stop_times(
        csv_row_number=2,
        start_pickup_drop_off_window=3600,
        end_pickup_drop_off_window=3600,
    )
    notices = validate_pickup_drop_off_window(make_feed(df), _CTX)
    assert len(notices) == 1
    n = notices[0]
    assert n.code == "invalid_pickup_drop_off_window"
    assert n.fields["start_pickup_drop_off_window"] == 3600
    assert n.fields["end_pickup_drop_off_window"] == 3600


# ---------------------------------------------------------------------------
# Valid / no-notice cases
# ---------------------------------------------------------------------------


def test_valid_window_no_notices() -> None:
    df = make_stop_times(
        csv_row_number=3,
        start_pickup_drop_off_window=3600,
        end_pickup_drop_off_window=7200,
    )
    notices = validate_pickup_drop_off_window(make_feed(df), _CTX)
    assert notices == []


def test_both_windows_null_row_skipped() -> None:
    df = make_stop_times(
        csv_row_number=8,
        arrival_time=0,
        departure_time=1,
    )
    notices = validate_pickup_drop_off_window(make_feed(df), _CTX)
    assert notices == []


# ---------------------------------------------------------------------------
# Multi-check interactions
# ---------------------------------------------------------------------------


def test_check_a_and_check_c_both_fire() -> None:
    df = make_stop_times(
        csv_row_number=4,
        arrival_time=100,
        start_pickup_drop_off_window=500,
        end_pickup_drop_off_window=200,
    )
    notices = validate_pickup_drop_off_window(make_feed(df), _CTX)
    assert len(notices) == 2
    codes = {n.code for n in notices}
    assert "forbidden_arrival_or_departure_time" in codes
    assert "invalid_pickup_drop_off_window" in codes
    for n in notices:
        assert n.fields["csv_row_number"] == 4


def test_check_a_and_check_b_both_fire() -> None:
    df = make_stop_times(
        csv_row_number=5,
        arrival_time=0,
        departure_time=1,
        end_pickup_drop_off_window=3,
    )
    notices = validate_pickup_drop_off_window(make_feed(df), _CTX)
    assert len(notices) == 2
    codes = {n.code for n in notices}
    assert "forbidden_arrival_or_departure_time" in codes
    assert "missing_pickup_or_drop_off_window" in codes
    for n in notices:
        assert n.fields["csv_row_number"] == 5


# ---------------------------------------------------------------------------
# Guard / short-circuit cases
# ---------------------------------------------------------------------------


def test_stop_times_absent_skips_validation() -> None:
    notices = validate_pickup_drop_off_window({}, _CTX)
    assert notices == []


def test_no_window_columns_in_header_skips_validation() -> None:
    df = pl.DataFrame(
        {"csv_row_number": [1], "arrival_time": [0], "departure_time": [1]},
        schema={"csv_row_number": pl.Int64, "arrival_time": pl.Int64, "departure_time": pl.Int64},
    )
    notices = validate_pickup_drop_off_window(make_feed(df), _CTX)
    assert notices == []


def test_only_start_column_in_header_no_notices() -> None:
    """When end column is absent, start-only rows should not fire any notice."""
    df = pl.DataFrame(
        {
            "csv_row_number": [9],
            "arrival_time": [None],
            "departure_time": [None],
            "start_pickup_drop_off_window": [3600],
        },
        schema={
            "csv_row_number": pl.Int64,
            "arrival_time": pl.Int64,
            "departure_time": pl.Int64,
            "start_pickup_drop_off_window": pl.Int64,
        },
    )
    notices = validate_pickup_drop_off_window(make_feed(df), _CTX)
    assert notices == []


def test_only_end_column_in_header_no_notices() -> None:
    """When start column is absent, end-only rows should not fire any notice."""
    df = pl.DataFrame(
        {
            "csv_row_number": [10],
            "end_pickup_drop_off_window": [3600],
        },
        schema={
            "csv_row_number": pl.Int64,
            "end_pickup_drop_off_window": pl.Int64,
        },
    )
    notices = validate_pickup_drop_off_window(make_feed(df), _CTX)
    assert notices == []


def test_arrival_time_column_absent_check_a_does_not_fire() -> None:
    df = pl.DataFrame(
        {
            "csv_row_number": [11],
            "departure_time": [None],
            "start_pickup_drop_off_window": [3600],
            "end_pickup_drop_off_window": [7200],
        },
        schema={
            "csv_row_number": pl.Int64,
            "departure_time": pl.Int64,
            "start_pickup_drop_off_window": pl.Int64,
            "end_pickup_drop_off_window": pl.Int64,
        },
    )
    notices = validate_pickup_drop_off_window(make_feed(df), _CTX)
    assert notices == []


def test_departure_time_column_absent_check_a_does_not_fire() -> None:
    df = pl.DataFrame(
        {
            "csv_row_number": [12],
            "arrival_time": [None],
            "start_pickup_drop_off_window": [3600],
            "end_pickup_drop_off_window": [7200],
        },
        schema={
            "csv_row_number": pl.Int64,
            "arrival_time": pl.Int64,
            "start_pickup_drop_off_window": pl.Int64,
            "end_pickup_drop_off_window": pl.Int64,
        },
    )
    notices = validate_pickup_drop_off_window(make_feed(df), _CTX)
    assert notices == []


def test_empty_stop_times_no_notices() -> None:
    df = pl.DataFrame(schema=_COLS)
    notices = validate_pickup_drop_off_window(make_feed(df), _CTX)
    assert notices == []


def test_multiple_rows_each_independently_evaluated() -> None:
    df = pl.DataFrame(
        {
            "csv_row_number": [1, 2, 3],
            "arrival_time": [None, None, None],
            "departure_time": [None, None, None],
            "start_pickup_drop_off_window": [3600, 7200, None],
            "end_pickup_drop_off_window": [7200, 3600, None],
        },
        schema=_COLS,
    )
    notices = validate_pickup_drop_off_window(make_feed(df), _CTX)
    assert len(notices) == 1
    assert notices[0].code == "invalid_pickup_drop_off_window"
    assert notices[0].fields["csv_row_number"] == 2
