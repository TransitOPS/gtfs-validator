"""Tests for PickupDropOffTypeValidator."""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from gtfs_validator.context import ValidationContext
from gtfs_validator.notices import Severity
from gtfs_validator.validators.pickup_drop_off_type import validate_pickup_drop_off_type

_COLS = {
    "csv_row_number": pl.Int64,
    "pickup_type": pl.Int64,
    "drop_off_type": pl.Int64,
    "start_pickup_drop_off_window": pl.Int64,
    "end_pickup_drop_off_window": pl.Int64,
}

_CTX = ValidationContext(country_code="US", date_for_validation=date(2024, 1, 1))


def make_stop_times(**overrides: object) -> pl.DataFrame:
    """Construct a 1-row DataFrame with all columns present and sensible defaults (nulls)."""
    defaults: dict[str, object] = {
        "csv_row_number": 1,
        "pickup_type": None,
        "drop_off_type": None,
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


def test_forbidden_drop_off_type_should_generate_notice() -> None:
    df = make_stop_times(
        csv_row_number=1,
        pickup_type=1,  # NOT_AVAILABLE — should not trigger pickup check
        drop_off_type=0,  # REGULAR — should trigger drop-off check
        start_pickup_drop_off_window=2,
        end_pickup_drop_off_window=3,
    )
    notices = validate_pickup_drop_off_type(make_feed(df), _CTX)
    assert len(notices) == 1
    n = notices[0]
    assert n.code == "forbidden_drop_off_type"
    assert n.severity == Severity.ERROR
    assert n.fields["csv_row_number"] == 1
    assert n.fields["start_pickup_drop_off_window"] == 2
    assert n.fields["end_pickup_drop_off_window"] == 3


def test_allowed_drop_off_type_should_not_generate_notice() -> None:
    df = make_stop_times(
        csv_row_number=2,
        drop_off_type=1,  # NOT_AVAILABLE
        start_pickup_drop_off_window=None,
        end_pickup_drop_off_window=None,
    )
    notices = validate_pickup_drop_off_type(make_feed(df), _CTX)
    assert notices == []


def test_forbidden_pickup_type_should_generate_notice() -> None:
    df = make_stop_times(
        csv_row_number=3,
        pickup_type=0,  # REGULAR
        drop_off_type=1,  # NOT_AVAILABLE
        start_pickup_drop_off_window=28800,
        end_pickup_drop_off_window=32400,
    )
    notices = validate_pickup_drop_off_type(make_feed(df), _CTX)
    assert len(notices) == 1
    n = notices[0]
    assert n.code == "forbidden_pickup_type"
    assert n.severity == Severity.ERROR
    assert n.fields["csv_row_number"] == 3
    assert n.fields["start_pickup_drop_off_window"] == 28800
    assert n.fields["end_pickup_drop_off_window"] == 32400


def test_allowed_pickup_type_should_not_generate_notice() -> None:
    df = make_stop_times(
        csv_row_number=4,
        pickup_type=1,  # NOT_AVAILABLE
        start_pickup_drop_off_window=None,
        end_pickup_drop_off_window=None,
    )
    notices = validate_pickup_drop_off_type(make_feed(df), _CTX)
    assert notices == []


def test_pickup_type_3_with_window_generates_notice() -> None:
    df = make_stop_times(
        csv_row_number=5,
        pickup_type=3,  # ON_REQUEST_TO_DRIVER
        drop_off_type=1,
        start_pickup_drop_off_window=3600,
        end_pickup_drop_off_window=None,
    )
    notices = validate_pickup_drop_off_type(make_feed(df), _CTX)
    assert len(notices) == 1
    n = notices[0]
    assert n.code == "forbidden_pickup_type"
    assert n.fields["csv_row_number"] == 5
    assert n.fields["start_pickup_drop_off_window"] == 3600
    assert n.fields["end_pickup_drop_off_window"] is None


def test_both_subchecks_fire_on_same_row() -> None:
    df = make_stop_times(
        csv_row_number=6,
        pickup_type=0,  # REGULAR
        drop_off_type=0,  # REGULAR
        start_pickup_drop_off_window=3600,
        end_pickup_drop_off_window=7200,
    )
    notices = validate_pickup_drop_off_type(make_feed(df), _CTX)
    assert len(notices) == 2
    codes = {n.code for n in notices}
    assert codes == {"forbidden_pickup_type", "forbidden_drop_off_type"}
    for n in notices:
        assert n.severity == Severity.ERROR
        assert n.fields["csv_row_number"] == 6
        assert n.fields["start_pickup_drop_off_window"] == 3600
        assert n.fields["end_pickup_drop_off_window"] == 7200


def test_pickup_type_3_and_drop_off_type_0_generates_two_notices() -> None:
    df = make_stop_times(
        csv_row_number=7,
        pickup_type=3,  # ON_REQUEST_TO_DRIVER
        drop_off_type=0,  # REGULAR
        start_pickup_drop_off_window=1800,
        end_pickup_drop_off_window=3600,
    )
    notices = validate_pickup_drop_off_type(make_feed(df), _CTX)
    assert len(notices) == 2
    codes = {n.code for n in notices}
    assert "forbidden_pickup_type" in codes
    assert "forbidden_drop_off_type" in codes
    for n in notices:
        assert n.fields["csv_row_number"] == 7


def test_stop_times_absent_skips_validation() -> None:
    notices = validate_pickup_drop_off_type({}, _CTX)
    assert notices == []


def test_no_window_columns_in_header_skips_validation() -> None:
    df = pl.DataFrame(
        {"csv_row_number": [1], "pickup_type": [0], "drop_off_type": [0]},
        schema={"csv_row_number": pl.Int64, "pickup_type": pl.Int64, "drop_off_type": pl.Int64},
    )
    notices = validate_pickup_drop_off_type(make_feed(df), _CTX)
    assert notices == []


def test_only_start_window_column_present() -> None:
    df = pl.DataFrame(
        {
            "csv_row_number": [8],
            "pickup_type": [0],
            "drop_off_type": [1],
            "start_pickup_drop_off_window": [7200],
        },
        schema={
            "csv_row_number": pl.Int64,
            "pickup_type": pl.Int64,
            "drop_off_type": pl.Int64,
            "start_pickup_drop_off_window": pl.Int64,
        },
    )
    notices = validate_pickup_drop_off_type(make_feed(df), _CTX)
    assert len(notices) == 1
    n = notices[0]
    assert n.code == "forbidden_pickup_type"
    assert n.fields["start_pickup_drop_off_window"] == 7200
    assert n.fields["end_pickup_drop_off_window"] is None


def test_only_end_window_column_present() -> None:
    df = pl.DataFrame(
        {
            "csv_row_number": [9],
            "pickup_type": [0],
            "drop_off_type": [1],
            "end_pickup_drop_off_window": [10800],
        },
        schema={
            "csv_row_number": pl.Int64,
            "pickup_type": pl.Int64,
            "drop_off_type": pl.Int64,
            "end_pickup_drop_off_window": pl.Int64,
        },
    )
    notices = validate_pickup_drop_off_type(make_feed(df), _CTX)
    assert len(notices) == 1
    n = notices[0]
    assert n.code == "forbidden_pickup_type"
    assert n.fields["start_pickup_drop_off_window"] is None
    assert n.fields["end_pickup_drop_off_window"] == 10800


def test_pickup_type_column_absent_skips_subcheck_a() -> None:
    df = pl.DataFrame(
        {
            "csv_row_number": [10],
            "drop_off_type": [1],
            "start_pickup_drop_off_window": [3600],
            "end_pickup_drop_off_window": [7200],
        },
        schema={
            "csv_row_number": pl.Int64,
            "drop_off_type": pl.Int64,
            "start_pickup_drop_off_window": pl.Int64,
            "end_pickup_drop_off_window": pl.Int64,
        },
    )
    notices = validate_pickup_drop_off_type(make_feed(df), _CTX)
    assert notices == []


def test_drop_off_type_column_absent_skips_subcheck_b() -> None:
    df = pl.DataFrame(
        {
            "csv_row_number": [11],
            "pickup_type": [2],  # MUST_PHONE — not in forbidden set {0, 3}
            "start_pickup_drop_off_window": [3600],
            "end_pickup_drop_off_window": [7200],
        },
        schema={
            "csv_row_number": pl.Int64,
            "pickup_type": pl.Int64,
            "start_pickup_drop_off_window": pl.Int64,
            "end_pickup_drop_off_window": pl.Int64,
        },
    )
    notices = validate_pickup_drop_off_type(make_feed(df), _CTX)
    assert notices == []


def test_both_window_columns_null_no_notices() -> None:
    df = make_stop_times(
        csv_row_number=12,
        pickup_type=0,
        drop_off_type=0,
        start_pickup_drop_off_window=None,
        end_pickup_drop_off_window=None,
    )
    notices = validate_pickup_drop_off_type(make_feed(df), _CTX)
    assert notices == []


def test_null_pickup_type_does_not_trigger_subcheck_a() -> None:
    df = make_stop_times(
        csv_row_number=13,
        pickup_type=None,
        drop_off_type=1,
        start_pickup_drop_off_window=3600,
        end_pickup_drop_off_window=7200,
    )
    notices = validate_pickup_drop_off_type(make_feed(df), _CTX)
    assert notices == []


def test_null_drop_off_type_does_not_trigger_subcheck_b() -> None:
    df = make_stop_times(
        csv_row_number=14,
        pickup_type=2,  # MUST_PHONE
        drop_off_type=None,
        start_pickup_drop_off_window=3600,
        end_pickup_drop_off_window=7200,
    )
    notices = validate_pickup_drop_off_type(make_feed(df), _CTX)
    assert notices == []


def test_empty_stop_times_no_notices() -> None:
    df = pl.DataFrame(schema=_COLS)
    notices = validate_pickup_drop_off_type(make_feed(df), _CTX)
    assert notices == []


def test_pickup_type_1_with_window_does_not_fire() -> None:
    df = make_stop_times(
        pickup_type=1,  # NOT_AVAILABLE
        drop_off_type=2,  # MUST_PHONE
        start_pickup_drop_off_window=3600,
        end_pickup_drop_off_window=7200,
    )
    notices = validate_pickup_drop_off_type(make_feed(df), _CTX)
    assert notices == []


def test_pickup_type_2_with_window_does_not_fire() -> None:
    df = make_stop_times(
        pickup_type=2,  # MUST_PHONE
        drop_off_type=3,  # ON_REQUEST_TO_DRIVER (not forbidden for drop-off)
        start_pickup_drop_off_window=3600,
        end_pickup_drop_off_window=7200,
    )
    notices = validate_pickup_drop_off_type(make_feed(df), _CTX)
    assert notices == []
