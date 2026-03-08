"""Tests for booking_rules_entity validator."""

from __future__ import annotations

from datetime import date

import polars as pl

from gtfs_validator.context import ValidationContext
from gtfs_validator.notices import Severity
from gtfs_validator.validators.booking_rules_entity import validate_booking_rules_entity

CTX = ValidationContext(country_code="US", date_for_validation=date(2024, 6, 1))

# All columns in booking_rules relevant to this validator.
_COLS = {
    "csv_row_number": pl.Int64,
    "booking_rule_id": pl.Utf8,
    "booking_type": pl.Int64,
    "prior_notice_duration_min": pl.Int64,
    "prior_notice_duration_max": pl.Int64,
    "prior_notice_last_day": pl.Int64,
    "prior_notice_last_time": pl.Utf8,
    "prior_notice_start_day": pl.Int64,
    "prior_notice_start_time": pl.Utf8,
    "prior_notice_service_id": pl.Utf8,
}


def make_booking_rules(**overrides: object) -> pl.DataFrame:
    """Create a 1-row booking_rules DataFrame with sensible defaults.

    Defaults: booking_rule_id="br1", booking_type=None, all prior-notice
    fields None, csv_row_number=2. Caller overrides specific fields.
    """
    defaults: dict[str, list[object]] = {
        "csv_row_number": [2],
        "booking_rule_id": ["br1"],
        "booking_type": [None],
        "prior_notice_duration_min": [None],
        "prior_notice_duration_max": [None],
        "prior_notice_last_day": [None],
        "prior_notice_last_time": [None],
        "prior_notice_start_day": [None],
        "prior_notice_start_time": [None],
        "prior_notice_service_id": [None],
    }
    for k, v in overrides.items():
        defaults[k] = [v]
    return pl.DataFrame(defaults, schema=_COLS)


# ------------------------------------------------------------------
# Test 1: REALTIME with forbidden fields
# ------------------------------------------------------------------


def test_realtime_with_forbidden_fields() -> None:
    df = make_booking_rules(
        booking_type=0,
        prior_notice_duration_min=30,
        prior_notice_last_day=2,
    )
    feed = {"booking_rules": df}
    notices = validate_booking_rules_entity(feed, CTX)
    codes = [n.code for n in notices]
    assert "forbidden_real_time_booking_field_value" in codes
    n = next(n for n in notices if n.code == "forbidden_real_time_booking_field_value")
    assert n.severity == Severity.ERROR
    assert "prior_notice_duration_min" in n.fields["field_names"]
    assert "prior_notice_last_day" in n.fields["field_names"]


# ------------------------------------------------------------------
# Test 2: REALTIME clean — no notices
# ------------------------------------------------------------------


def test_realtime_clean() -> None:
    df = make_booking_rules(booking_type=0)
    feed = {"booking_rules": df}
    notices = validate_booking_rules_entity(feed, CTX)
    assert len(notices) == 0


# ------------------------------------------------------------------
# Test 3: SAMEDAY valid
# ------------------------------------------------------------------


def test_sameday_valid() -> None:
    df = make_booking_rules(booking_type=1, prior_notice_duration_min=1)
    feed = {"booking_rules": df}
    notices = validate_booking_rules_entity(feed, CTX)
    assert len(notices) == 0


# ------------------------------------------------------------------
# Test 4: SAMEDAY forbidden last_day
# ------------------------------------------------------------------


def test_sameday_forbidden_last_day() -> None:
    df = make_booking_rules(
        booking_type=1,
        prior_notice_last_day=2,
        prior_notice_duration_min=1,
    )
    feed = {"booking_rules": df}
    notices = validate_booking_rules_entity(feed, CTX)
    codes = [n.code for n in notices]
    assert "forbidden_same_day_booking_field_value" in codes
    n = next(n for n in notices if n.code == "forbidden_same_day_booking_field_value")
    assert n.fields["field_names"] == "prior_notice_last_day"


# ------------------------------------------------------------------
# Test 5: SAMEDAY missing duration_min
# ------------------------------------------------------------------


def test_sameday_missing_duration_min() -> None:
    df = make_booking_rules(booking_type=1)
    feed = {"booking_rules": df}
    notices = validate_booking_rules_entity(feed, CTX)
    codes = [n.code for n in notices]
    assert "missing_prior_notice_duration_min" in codes
    assert len([c for c in codes if c == "missing_prior_notice_duration_min"]) == 1


# ------------------------------------------------------------------
# Test 6: REALTIME multiple forbidden
# ------------------------------------------------------------------


def test_realtime_multiple_forbidden() -> None:
    df = make_booking_rules(
        booking_type=0,
        prior_notice_start_day=1,
        prior_notice_start_time="08:00:00",
        prior_notice_service_id="s1",
    )
    feed = {"booking_rules": df}
    notices = validate_booking_rules_entity(feed, CTX)
    forbidden = [n for n in notices if n.code == "forbidden_real_time_booking_field_value"]
    assert len(forbidden) == 1
    field_names = forbidden[0].fields["field_names"]
    assert "prior_notice_start_day" in field_names
    assert "prior_notice_start_time" in field_names
    assert "prior_notice_service_id" in field_names


# ------------------------------------------------------------------
# Test 7: PRIORDAY forbidden duration fields
# ------------------------------------------------------------------


def test_priorday_forbidden_duration_fields() -> None:
    df = make_booking_rules(
        booking_type=2,
        prior_notice_last_day=1,
        prior_notice_last_time="08:00:00",
        prior_notice_duration_min=30,
        prior_notice_duration_max=60,
    )
    feed = {"booking_rules": df}
    notices = validate_booking_rules_entity(feed, CTX)
    forbidden = [n for n in notices if n.code == "forbidden_prior_day_booking_field_value"]
    assert len(forbidden) == 1
    field_names = forbidden[0].fields["field_names"]
    assert "prior_notice_duration_min" in field_names
    assert "prior_notice_duration_max" in field_names


# ------------------------------------------------------------------
# Test 8: duration min greater than max
# ------------------------------------------------------------------


def test_duration_min_greater_than_max() -> None:
    df = make_booking_rules(
        booking_type=1,
        prior_notice_duration_min=60,
        prior_notice_duration_max=30,
    )
    feed = {"booking_rules": df}
    notices = validate_booking_rules_entity(feed, CTX)
    inv = [n for n in notices if n.code == "invalid_prior_notice_duration_min"]
    assert len(inv) == 1
    assert inv[0].fields["prior_notice_duration_min"] == 60
    assert inv[0].fields["prior_notice_duration_max"] == 30


# ------------------------------------------------------------------
# Test 9: start_day forbidden with duration_max
# ------------------------------------------------------------------


def test_start_day_forbidden_with_duration_max() -> None:
    df = make_booking_rules(
        booking_type=1,
        prior_notice_duration_min=1,
        prior_notice_duration_max=30,
        prior_notice_start_day=5,
        prior_notice_start_time="08:00:00",
    )
    feed = {"booking_rules": df}
    notices = validate_booking_rules_entity(feed, CTX)
    fsd = [n for n in notices if n.code == "forbidden_prior_notice_start_day"]
    assert len(fsd) >= 1
    assert fsd[0].fields["prior_notice_start_day"] == 5
    assert fsd[0].fields["prior_notice_duration_max"] == 30


# ------------------------------------------------------------------
# Test 10: last_day after start_day
# ------------------------------------------------------------------


def test_last_day_after_start_day() -> None:
    df = make_booking_rules(
        prior_notice_last_day=5,
        prior_notice_start_day=3,
        prior_notice_start_time="08:00:00",
    )
    feed = {"booking_rules": df}
    notices = validate_booking_rules_entity(feed, CTX)
    lda = [n for n in notices if n.code == "prior_notice_last_day_after_start_day"]
    assert len(lda) == 1
    assert lda[0].fields["prior_notice_last_day"] == 5
    assert lda[0].fields["prior_notice_start_day"] == 3
    # booking_rule_id intentionally NOT present in this notice
    assert "booking_rule_id" not in lda[0].fields


# ------------------------------------------------------------------
# Test 11: PRIORDAY missing both last fields
# ------------------------------------------------------------------


def test_priorday_missing_both_last_fields() -> None:
    df = make_booking_rules(booking_type=2)
    feed = {"booking_rules": df}
    notices = validate_booking_rules_entity(feed, CTX)
    codes = [n.code for n in notices]
    assert "missing_prior_notice_last_day" in codes
    assert "missing_prior_notice_last_time" in codes


# ------------------------------------------------------------------
# Test 12: PRIORDAY missing last_time only
# ------------------------------------------------------------------


def test_priorday_missing_last_time_only() -> None:
    df = make_booking_rules(booking_type=2, prior_notice_last_day=2)
    feed = {"booking_rules": df}
    notices = validate_booking_rules_entity(feed, CTX)
    codes = [n.code for n in notices]
    assert "missing_prior_notice_last_time" in codes
    assert "missing_prior_notice_last_day" not in codes


# ------------------------------------------------------------------
# Test 13: PRIORDAY missing last_day only
# ------------------------------------------------------------------


def test_priorday_missing_last_day_only() -> None:
    df = make_booking_rules(booking_type=2, prior_notice_last_time="08:00:00")
    feed = {"booking_rules": df}
    notices = validate_booking_rules_entity(feed, CTX)
    codes = [n.code for n in notices]
    assert "missing_prior_notice_last_day" in codes
    assert "missing_prior_notice_last_time" not in codes


# ------------------------------------------------------------------
# Test 14: start_time without start_day
# ------------------------------------------------------------------


def test_start_time_without_start_day() -> None:
    df = make_booking_rules(
        booking_type=2,
        prior_notice_start_time="08:00:00",
        prior_notice_last_day=1,
        prior_notice_last_time="09:00:00",
    )
    feed = {"booking_rules": df}
    notices = validate_booking_rules_entity(feed, CTX)
    fst = [n for n in notices if n.code == "forbidden_prior_notice_start_time"]
    assert len(fst) == 1


# ------------------------------------------------------------------
# Test 15: start_day without start_time
# ------------------------------------------------------------------


def test_start_day_without_start_time() -> None:
    df = make_booking_rules(
        booking_type=2,
        prior_notice_start_day=3,
        prior_notice_last_day=1,
        prior_notice_last_time="09:00:00",
    )
    feed = {"booking_rules": df}
    notices = validate_booking_rules_entity(feed, CTX)
    mst = [n for n in notices if n.code == "missing_prior_notice_start_time"]
    assert len(mst) == 1


# ------------------------------------------------------------------
# Test 16: missing booking_rules table
# ------------------------------------------------------------------


def test_missing_booking_rules_table() -> None:
    feed: dict[str, pl.DataFrame] = {}
    notices = validate_booking_rules_entity(feed, CTX)
    assert len(notices) == 0


# ------------------------------------------------------------------
# Test 17: empty booking_rules table
# ------------------------------------------------------------------


def test_empty_booking_rules_table() -> None:
    feed = {"booking_rules": pl.DataFrame(schema=_COLS)}
    notices = validate_booking_rules_entity(feed, CTX)
    assert len(notices) == 0


# ------------------------------------------------------------------
# Test 18: unrecognized booking_type
# ------------------------------------------------------------------


def test_unrecognized_booking_type() -> None:
    df = make_booking_rules(
        booking_type=99,
        prior_notice_duration_min=60,
        prior_notice_duration_max=30,
    )
    feed = {"booking_rules": df}
    notices = validate_booking_rules_entity(feed, CTX)
    codes = [n.code for n in notices]
    # No type-specific notices
    assert "forbidden_real_time_booking_field_value" not in codes
    assert "forbidden_same_day_booking_field_value" not in codes
    assert "forbidden_prior_day_booking_field_value" not in codes
    # Type-independent check still fires
    assert "invalid_prior_notice_duration_min" in codes


# ------------------------------------------------------------------
# Test 19: multiple notices from a single row
# ------------------------------------------------------------------


def test_multiple_notices_single_row() -> None:
    df = make_booking_rules(
        booking_type=2,
        prior_notice_duration_min=30,
        prior_notice_duration_max=60,
    )
    feed = {"booking_rules": df}
    notices = validate_booking_rules_entity(feed, CTX)
    codes = [n.code for n in notices]
    assert "forbidden_prior_day_booking_field_value" in codes
    assert "missing_prior_notice_last_day" in codes
    assert "missing_prior_notice_last_time" in codes
    assert len(codes) == 3


# ------------------------------------------------------------------
# Test 20: start_day and start_time both present — no co-dep notices
# ------------------------------------------------------------------


def test_start_day_and_start_time_both_present() -> None:
    df = make_booking_rules(
        booking_type=2,
        prior_notice_start_day=3,
        prior_notice_start_time="08:00:00",
        prior_notice_last_day=1,
        prior_notice_last_time="09:00:00",
    )
    feed = {"booking_rules": df}
    notices = validate_booking_rules_entity(feed, CTX)
    codes = [n.code for n in notices]
    assert "forbidden_prior_notice_start_time" not in codes
    assert "missing_prior_notice_start_time" not in codes
