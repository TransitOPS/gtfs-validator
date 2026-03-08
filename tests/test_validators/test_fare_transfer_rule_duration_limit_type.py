"""Tests for validate_fare_transfer_rule_duration_limit_type."""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from gtfs_validator.context import ValidationContext
from gtfs_validator.notices import Severity
from gtfs_validator.validators.fare_transfer_rule_duration_limit_type import (
    validate_fare_transfer_rule_duration_limit_type,
)

CTX = ValidationContext(country_code="US", date_for_validation=date(2024, 1, 1))


def make_fare_transfer_rules(rows: list[dict]) -> pl.DataFrame:
    """Build a fare_transfer_rules DataFrame from a list of dicts.

    Keys: duration_limit (int | None), duration_limit_type (int | None),
    csv_row_number (int). Other columns are omitted unless explicitly needed.
    """
    return pl.DataFrame(
        {
            "duration_limit": pl.Series(
                [r.get("duration_limit") for r in rows], dtype=pl.Int64
            ),
            "duration_limit_type": pl.Series(
                [r.get("duration_limit_type") for r in rows], dtype=pl.Int64
            ),
            "csv_row_number": [r["csv_row_number"] for r in rows],
        }
    )


def test_duration_limit_without_type_generates_notice() -> None:
    """Row with duration_limit set but duration_limit_type absent emits an ERROR."""
    feed = {
        "fare_transfer_rules": make_fare_transfer_rules(
            [{"duration_limit": 120, "duration_limit_type": None, "csv_row_number": 2}]
        )
    }
    notices = validate_fare_transfer_rule_duration_limit_type(feed, CTX)
    assert len(notices) == 1
    assert notices[0].code == "fare_transfer_rule_duration_limit_without_type"
    assert notices[0].severity == Severity.ERROR
    assert notices[0].fields["csv_row_number"] == 2


def test_duration_limit_type_without_duration_limit_generates_notice() -> None:
    """Row with duration_limit_type set but duration_limit absent emits an ERROR."""
    feed = {
        "fare_transfer_rules": make_fare_transfer_rules(
            [{"duration_limit": None, "duration_limit_type": 3, "csv_row_number": 2}]
        )
    }
    notices = validate_fare_transfer_rule_duration_limit_type(feed, CTX)
    assert len(notices) == 1
    assert notices[0].code == "fare_transfer_rule_duration_limit_type_without_duration_limit"
    assert notices[0].severity == Severity.ERROR
    assert notices[0].fields["csv_row_number"] == 2


def test_both_present_no_notice() -> None:
    """Row with both duration_limit and duration_limit_type set produces no notices."""
    feed = {
        "fare_transfer_rules": make_fare_transfer_rules(
            [{"duration_limit": 120, "duration_limit_type": 3, "csv_row_number": 2}]
        )
    }
    notices = validate_fare_transfer_rule_duration_limit_type(feed, CTX)
    assert notices == []


def test_both_absent_no_notice() -> None:
    """Row with both duration_limit and duration_limit_type absent produces no notices."""
    feed = {
        "fare_transfer_rules": make_fare_transfer_rules(
            [{"duration_limit": None, "duration_limit_type": None, "csv_row_number": 2}]
        )
    }
    notices = validate_fare_transfer_rule_duration_limit_type(feed, CTX)
    assert notices == []


def test_fare_transfer_rules_absent_skips() -> None:
    """Missing fare_transfer_rules key returns empty list immediately."""
    notices = validate_fare_transfer_rule_duration_limit_type({}, CTX)
    assert notices == []


def test_fare_transfer_rules_empty_skips() -> None:
    """Empty fare_transfer_rules DataFrame returns empty list immediately."""
    empty_df = pl.DataFrame(
        {
            "duration_limit": pl.Series([], dtype=pl.Int64),
            "duration_limit_type": pl.Series([], dtype=pl.Int64),
            "csv_row_number": pl.Series([], dtype=pl.Int64),
        }
    )
    feed = {"fare_transfer_rules": empty_df}
    notices = validate_fare_transfer_rule_duration_limit_type(feed, CTX)
    assert notices == []


def test_multiple_violating_rows_each_emits_notice() -> None:
    """Each violating row gets its own notice; different codes per violation type."""
    feed = {
        "fare_transfer_rules": make_fare_transfer_rules(
            [
                {"duration_limit": 60, "duration_limit_type": None, "csv_row_number": 3},
                {"duration_limit": None, "duration_limit_type": 1, "csv_row_number": 5},
            ]
        )
    }
    notices = validate_fare_transfer_rule_duration_limit_type(feed, CTX)
    assert len(notices) == 2
    assert notices[0].code == "fare_transfer_rule_duration_limit_without_type"
    assert notices[0].fields["csv_row_number"] == 3
    assert notices[1].code == "fare_transfer_rule_duration_limit_type_without_duration_limit"
    assert notices[1].fields["csv_row_number"] == 5


def test_duration_limit_type_departure_to_arrival_without_limit() -> None:
    """duration_limit_type=0 (DEPARTURE_TO_ARRIVAL) without duration_limit emits ERROR."""
    feed = {
        "fare_transfer_rules": make_fare_transfer_rules(
            [{"duration_limit": None, "duration_limit_type": 0, "csv_row_number": 4}]
        )
    }
    notices = validate_fare_transfer_rule_duration_limit_type(feed, CTX)
    assert len(notices) == 1
    assert notices[0].code == "fare_transfer_rule_duration_limit_type_without_duration_limit"
    assert notices[0].fields["csv_row_number"] == 4


def test_duration_limit_type_departure_to_departure_without_limit() -> None:
    """duration_limit_type=1 (DEPARTURE_TO_DEPARTURE) without duration_limit emits ERROR."""
    feed = {
        "fare_transfer_rules": make_fare_transfer_rules(
            [{"duration_limit": None, "duration_limit_type": 1, "csv_row_number": 7}]
        )
    }
    notices = validate_fare_transfer_rule_duration_limit_type(feed, CTX)
    assert len(notices) == 1
    assert notices[0].code == "fare_transfer_rule_duration_limit_type_without_duration_limit"
    assert notices[0].fields["csv_row_number"] == 7


def test_notice_fields_exact_values_limit_without_type() -> None:
    """All three notice properties (code, severity, fields) match exactly."""
    feed = {
        "fare_transfer_rules": make_fare_transfer_rules(
            [{"duration_limit": 300, "duration_limit_type": None, "csv_row_number": 9}]
        )
    }
    notices = validate_fare_transfer_rule_duration_limit_type(feed, CTX)
    assert len(notices) == 1
    notice = notices[0]
    assert notice.code == "fare_transfer_rule_duration_limit_without_type"
    assert notice.severity == Severity.ERROR
    assert notice.fields == {"csv_row_number": 9}


def test_notice_fields_exact_values_type_without_limit() -> None:
    """All three notice properties (code, severity, fields) match exactly."""
    feed = {
        "fare_transfer_rules": make_fare_transfer_rules(
            [{"duration_limit": None, "duration_limit_type": 2, "csv_row_number": 11}]
        )
    }
    notices = validate_fare_transfer_rule_duration_limit_type(feed, CTX)
    assert len(notices) == 1
    notice = notices[0]
    assert notice.code == "fare_transfer_rule_duration_limit_type_without_duration_limit"
    assert notice.severity == Severity.ERROR
    assert notice.fields == {"csv_row_number": 11}


def test_mixed_rows_only_violating_rows_emit_notices() -> None:
    """Valid rows produce no notices; only the two invalid rows each emit one notice."""
    feed = {
        "fare_transfer_rules": make_fare_transfer_rules(
            [
                {"duration_limit": 120, "duration_limit_type": 0, "csv_row_number": 1},
                {"duration_limit": None, "duration_limit_type": None, "csv_row_number": 2},
                {"duration_limit": 60, "duration_limit_type": None, "csv_row_number": 3},
                {"duration_limit": None, "duration_limit_type": 1, "csv_row_number": 4},
            ]
        )
    }
    notices = validate_fare_transfer_rule_duration_limit_type(feed, CTX)
    assert len(notices) == 2
    codes = [n.code for n in notices]
    row_numbers = [n.fields["csv_row_number"] for n in notices]
    assert "fare_transfer_rule_duration_limit_without_type" in codes
    assert "fare_transfer_rule_duration_limit_type_without_duration_limit" in codes
    assert 3 in row_numbers
    assert 4 in row_numbers
