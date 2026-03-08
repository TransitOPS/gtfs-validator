"""Tests for PickupBookingRuleIdValidator."""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from gtfs_validator.context import ValidationContext
from gtfs_validator.notices import Severity
from gtfs_validator.validators.pickup_booking_rule_id import validate_pickup_booking_rule_id

_COLS: dict[str, type] = {
    "csv_row_number": pl.Int64,
    "pickup_type": pl.Int64,
    "drop_off_type": pl.Int64,
    "start_pickup_drop_off_window": pl.Int64,
    "end_pickup_drop_off_window": pl.Int64,
    "pickup_booking_rule_id": pl.Utf8,
    "drop_off_booking_rule_id": pl.Utf8,
}

_CTX = ValidationContext(country_code="US", date_for_validation=date(2024, 1, 1))

_BOOKING_RULES_DF = pl.DataFrame({"booking_rule_id": ["rule1"]})


def make_stop_times(**overrides: object) -> pl.DataFrame:
    """Build a 1-row stop_times DataFrame with all columns, applying overrides."""
    defaults: dict[str, object] = {
        "csv_row_number": 1,
        "pickup_type": None,
        "drop_off_type": None,
        "start_pickup_drop_off_window": None,
        "end_pickup_drop_off_window": None,
        "pickup_booking_rule_id": None,
        "drop_off_booking_rule_id": None,
    }
    defaults.update(overrides)

    return pl.DataFrame(
        {col: [defaults[col]] for col in _COLS},
        schema={col: dtype for col, dtype in _COLS.items()},
    )


def make_feed(stop_times_df: pl.DataFrame) -> dict[str, object]:
    """Build a feed dict with stop_times and a minimal booking_rules table."""
    return {
        "stop_times": stop_times_df,
        "booking_rules": _BOOKING_RULES_DF,
    }


def test_missing_booking_rule_id_should_generate_notice() -> None:
    """Mirrors missingBookingRuleIdShouldGenerateNotice."""
    df = make_stop_times(
        csv_row_number=1,
        pickup_type=2,
        start_pickup_drop_off_window=18614,
        pickup_booking_rule_id=None,
        drop_off_type=None,
    )
    feed = make_feed(df)
    notices = validate_pickup_booking_rule_id(feed, _CTX)  # type: ignore[arg-type]

    assert len(notices) == 1
    n = notices[0]
    assert n.code == "missing_pickup_drop_off_booking_rule_id"
    assert n.severity == Severity.WARNING
    assert n.fields["csv_row_number"] == 1
    assert n.fields["pickup_type"] == 2
    assert n.fields["drop_off_type"] is None


def test_existing_booking_rule_id_should_not_generate_notice() -> None:
    """Mirrors existingBookingRuleIdShouldNotGenerateNotice."""
    df = make_stop_times(
        csv_row_number=2,
        pickup_type=2,
        start_pickup_drop_off_window=18614,
        pickup_booking_rule_id="booking_rule_id",
    )
    feed = make_feed(df)
    notices = validate_pickup_booking_rule_id(feed, _CTX)  # type: ignore[arg-type]

    assert notices == []


def test_pickup_type_not_must_phone_should_not_generate_notice() -> None:
    """Mirrors pickUpTypeNotMustPhoneShouldNotGenerateNotice."""
    df = make_stop_times(
        csv_row_number=3,
        pickup_type=1,
        start_pickup_drop_off_window=None,
        end_pickup_drop_off_window=None,
        pickup_booking_rule_id=None,
        drop_off_type=None,
    )
    feed = make_feed(df)
    notices = validate_pickup_booking_rule_id(feed, _CTX)  # type: ignore[arg-type]

    assert notices == []


def test_booking_rules_absent_skips_validation() -> None:
    """When booking_rules is absent, validation is skipped entirely."""
    df = make_stop_times(
        pickup_type=2,
        start_pickup_drop_off_window=18614,
        pickup_booking_rule_id=None,
    )
    feed = {"stop_times": df}
    notices = validate_pickup_booking_rule_id(feed, _CTX)  # type: ignore[arg-type]

    assert notices == []


def test_stop_times_absent_skips_validation() -> None:
    """When stop_times is absent, validation is skipped entirely."""
    feed = {"booking_rules": _BOOKING_RULES_DF}
    notices = validate_pickup_booking_rule_id(feed, _CTX)  # type: ignore[arg-type]

    assert notices == []


def test_neither_pickup_nor_dropoff_column_skips_validation() -> None:
    """When neither pickup_type nor drop_off_type column is present, skip."""
    df = pl.DataFrame(
        {
            "csv_row_number": [1],
            "start_pickup_drop_off_window": [18614],
            "end_pickup_drop_off_window": [36000],
        },
        schema={
            "csv_row_number": pl.Int64,
            "start_pickup_drop_off_window": pl.Int64,
            "end_pickup_drop_off_window": pl.Int64,
        },
    )
    feed = make_feed(df)
    notices = validate_pickup_booking_rule_id(feed, _CTX)  # type: ignore[arg-type]

    assert notices == []


def test_pickup_type_2_without_window_does_not_fire() -> None:
    """Window is required co-condition; without it, sub-check A must not fire."""
    df = make_stop_times(
        pickup_type=2,
        start_pickup_drop_off_window=None,
        pickup_booking_rule_id=None,
    )
    feed = make_feed(df)
    notices = validate_pickup_booking_rule_id(feed, _CTX)  # type: ignore[arg-type]

    assert notices == []


def test_drop_off_type_2_without_window_does_not_fire() -> None:
    """Window is required co-condition; without it, sub-check B must not fire."""
    df = make_stop_times(
        drop_off_type=2,
        end_pickup_drop_off_window=None,
        drop_off_booking_rule_id=None,
    )
    feed = make_feed(df)
    notices = validate_pickup_booking_rule_id(feed, _CTX)  # type: ignore[arg-type]

    assert notices == []


def test_both_subchecks_trigger_on_same_row() -> None:
    """When both pickup_type=2 and drop_off_type=2 with no booking IDs, 2 notices."""
    df = make_stop_times(
        csv_row_number=5,
        pickup_type=2,
        drop_off_type=2,
        start_pickup_drop_off_window=18614,
        end_pickup_drop_off_window=36000,
        pickup_booking_rule_id=None,
        drop_off_booking_rule_id=None,
    )
    feed = make_feed(df)
    notices = validate_pickup_booking_rule_id(feed, _CTX)  # type: ignore[arg-type]

    assert len(notices) == 2
    for n in notices:
        assert n.code == "missing_pickup_drop_off_booking_rule_id"
        assert n.severity == Severity.WARNING
        assert n.fields["csv_row_number"] == 5
        assert n.fields["pickup_type"] == 2
        assert n.fields["drop_off_type"] == 2


def test_drop_off_missing_booking_rule_generates_notice() -> None:
    """drop_off_type=2 with window and no drop_off_booking_rule_id fires sub-check B."""
    df = make_stop_times(
        csv_row_number=7,
        pickup_type=0,
        drop_off_type=2,
        end_pickup_drop_off_window=36000,
        drop_off_booking_rule_id=None,
        start_pickup_drop_off_window=None,
        pickup_booking_rule_id="some_rule",
    )
    feed = make_feed(df)
    notices = validate_pickup_booking_rule_id(feed, _CTX)  # type: ignore[arg-type]

    assert len(notices) == 1
    n = notices[0]
    assert n.fields["csv_row_number"] == 7
    assert n.fields["drop_off_type"] == 2
    assert n.fields["pickup_type"] == 0


def test_empty_stop_times_no_notices() -> None:
    """Empty stop_times DataFrame yields no notices."""
    df = pl.DataFrame(
        {col: [] for col in _COLS},
        schema={col: dtype for col, dtype in _COLS.items()},
    )
    feed = make_feed(df)
    notices = validate_pickup_booking_rule_id(feed, _CTX)  # type: ignore[arg-type]

    assert notices == []


def test_pickup_type_column_absent_skips_subcheck_a() -> None:
    """When pickup_type column is absent, sub-check A is skipped."""
    # Build a DataFrame without pickup_type; drop_off_type=0 (not 2) so B doesn't fire
    df = pl.DataFrame(
        {
            "csv_row_number": [1],
            "drop_off_type": [0],
            "start_pickup_drop_off_window": [18614],
            "end_pickup_drop_off_window": [36000],
            "pickup_booking_rule_id": [None],
            "drop_off_booking_rule_id": [None],
        },
        schema={
            "csv_row_number": pl.Int64,
            "drop_off_type": pl.Int64,
            "start_pickup_drop_off_window": pl.Int64,
            "end_pickup_drop_off_window": pl.Int64,
            "pickup_booking_rule_id": pl.Utf8,
            "drop_off_booking_rule_id": pl.Utf8,
        },
    )
    feed = make_feed(df)
    notices = validate_pickup_booking_rule_id(feed, _CTX)  # type: ignore[arg-type]

    assert notices == []
