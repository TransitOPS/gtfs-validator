"""Tests for validate_fare_transfer_rule_transfer_count."""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from gtfs_validator.context import ValidationContext
from gtfs_validator.notices import Severity
from gtfs_validator.validators.fare_transfer_rule_transfer_count import (
    validate_fare_transfer_rule_transfer_count,
)

CTX = ValidationContext(country_code="US", date_for_validation=date(2024, 1, 1))


def make_fare_transfer_rules(rows: list[dict]) -> pl.DataFrame:
    """Build a fare_transfer_rules DataFrame from a list of dicts.

    Keys: from_leg_group_id (str | None), to_leg_group_id (str | None),
    transfer_count (int | None), csv_row_number (int).
    Other columns are omitted unless explicitly needed.
    """
    return pl.DataFrame(
        {
            "from_leg_group_id": pl.Series(
                [r.get("from_leg_group_id") for r in rows], dtype=pl.Utf8
            ),
            "to_leg_group_id": pl.Series(
                [r.get("to_leg_group_id") for r in rows], dtype=pl.Utf8
            ),
            "transfer_count": pl.Series(
                [r.get("transfer_count") for r in rows], dtype=pl.Int64
            ),
            "csv_row_number": [r["csv_row_number"] for r in rows],
        }
    )


def test_valid_transfer_count_one() -> None:
    """Self-loop with transfer_count=1 is valid — no notices."""
    feed = {
        "fare_transfer_rules": make_fare_transfer_rules(
            [{"from_leg_group_id": "a", "to_leg_group_id": "a", "transfer_count": 1, "csv_row_number": 2}]
        )
    }
    assert validate_fare_transfer_rule_transfer_count(feed, CTX) == []


def test_valid_transfer_count_negative_one() -> None:
    """Self-loop with transfer_count=-1 (unlimited) is valid — no notices."""
    feed = {
        "fare_transfer_rules": make_fare_transfer_rules(
            [{"from_leg_group_id": "a", "to_leg_group_id": "a", "transfer_count": -1, "csv_row_number": 2}]
        )
    }
    assert validate_fare_transfer_rule_transfer_count(feed, CTX) == []


def test_invalid_transfer_count_zero() -> None:
    """Self-loop with transfer_count=0 emits invalid_transfer_count error."""
    feed = {
        "fare_transfer_rules": make_fare_transfer_rules(
            [{"from_leg_group_id": "a", "to_leg_group_id": "a", "transfer_count": 0, "csv_row_number": 2}]
        )
    }
    notices = validate_fare_transfer_rule_transfer_count(feed, CTX)
    assert len(notices) == 1
    assert notices[0].code == "fare_transfer_rule_invalid_transfer_count"
    assert notices[0].severity == Severity.ERROR
    assert notices[0].fields["csv_row_number"] == 2
    assert notices[0].fields["transfer_count"] == 0


def test_invalid_transfer_count_negative_two() -> None:
    """Self-loop with transfer_count=-2 emits invalid_transfer_count error."""
    feed = {
        "fare_transfer_rules": make_fare_transfer_rules(
            [{"from_leg_group_id": "a", "to_leg_group_id": "a", "transfer_count": -2, "csv_row_number": 2}]
        )
    }
    notices = validate_fare_transfer_rule_transfer_count(feed, CTX)
    assert len(notices) == 1
    assert notices[0].code == "fare_transfer_rule_invalid_transfer_count"
    assert notices[0].severity == Severity.ERROR
    assert notices[0].fields["csv_row_number"] == 2
    assert notices[0].fields["transfer_count"] == -2


def test_missing_required_transfer_count() -> None:
    """Self-loop missing transfer_count emits without_transfer_count error."""
    feed = {
        "fare_transfer_rules": make_fare_transfer_rules(
            [{"from_leg_group_id": "a", "to_leg_group_id": "a", "transfer_count": None, "csv_row_number": 2}]
        )
    }
    notices = validate_fare_transfer_rule_transfer_count(feed, CTX)
    assert len(notices) == 1
    assert notices[0].code == "fare_transfer_rule_without_transfer_count"
    assert notices[0].severity == Severity.ERROR
    assert notices[0].fields["csv_row_number"] == 2


def test_valid_unspecified_transfer_count() -> None:
    """Non-loop row without transfer_count is valid — no notices."""
    feed = {
        "fare_transfer_rules": make_fare_transfer_rules(
            [{"from_leg_group_id": "a", "to_leg_group_id": "b", "transfer_count": None, "csv_row_number": 2}]
        )
    }
    assert validate_fare_transfer_rule_transfer_count(feed, CTX) == []


def test_forbidden_transfer_count() -> None:
    """Non-loop row with transfer_count present emits with_forbidden_transfer_count error."""
    feed = {
        "fare_transfer_rules": make_fare_transfer_rules(
            [{"from_leg_group_id": "a", "to_leg_group_id": "b", "transfer_count": 1, "csv_row_number": 2}]
        )
    }
    notices = validate_fare_transfer_rule_transfer_count(feed, CTX)
    assert len(notices) == 1
    assert notices[0].code == "fare_transfer_rule_with_forbidden_transfer_count"
    assert notices[0].severity == Severity.ERROR
    assert notices[0].fields["csv_row_number"] == 2


def test_fare_transfer_rules_absent_skips() -> None:
    """Missing fare_transfer_rules key returns empty list."""
    assert validate_fare_transfer_rule_transfer_count({}, CTX) == []


def test_fare_transfer_rules_empty_skips() -> None:
    """Empty fare_transfer_rules DataFrame returns empty list."""
    empty_df = make_fare_transfer_rules([])
    feed = {"fare_transfer_rules": empty_df}
    assert validate_fare_transfer_rule_transfer_count(feed, CTX) == []


def test_from_leg_group_id_null_transfer_count_present() -> None:
    """Null from_leg_group_id with transfer_count present emits forbidden error."""
    feed = {
        "fare_transfer_rules": make_fare_transfer_rules(
            [{"from_leg_group_id": None, "to_leg_group_id": "a", "transfer_count": 1, "csv_row_number": 3}]
        )
    }
    notices = validate_fare_transfer_rule_transfer_count(feed, CTX)
    assert len(notices) == 1
    assert notices[0].code == "fare_transfer_rule_with_forbidden_transfer_count"
    assert notices[0].fields["csv_row_number"] == 3


def test_both_leg_group_ids_null_transfer_count_present() -> None:
    """Both leg group IDs null with transfer_count present emits forbidden error."""
    feed = {
        "fare_transfer_rules": make_fare_transfer_rules(
            [{"from_leg_group_id": None, "to_leg_group_id": None, "transfer_count": 1, "csv_row_number": 4}]
        )
    }
    notices = validate_fare_transfer_rule_transfer_count(feed, CTX)
    assert len(notices) == 1
    assert notices[0].code == "fare_transfer_rule_with_forbidden_transfer_count"
    assert notices[0].fields["csv_row_number"] == 4


def test_both_leg_group_ids_null_transfer_count_absent() -> None:
    """Both leg group IDs null and transfer_count absent — no notices."""
    feed = {
        "fare_transfer_rules": make_fare_transfer_rules(
            [{"from_leg_group_id": None, "to_leg_group_id": None, "transfer_count": None, "csv_row_number": 5}]
        )
    }
    assert validate_fare_transfer_rule_transfer_count(feed, CTX) == []


def test_multiple_violating_rows_mixed_codes() -> None:
    """Two violating rows with different codes each produce one notice."""
    feed = {
        "fare_transfer_rules": make_fare_transfer_rules(
            [
                {"from_leg_group_id": "a", "to_leg_group_id": "a", "transfer_count": None, "csv_row_number": 2},
                {"from_leg_group_id": "a", "to_leg_group_id": "b", "transfer_count": 1, "csv_row_number": 3},
            ]
        )
    }
    notices = validate_fare_transfer_rule_transfer_count(feed, CTX)
    assert len(notices) == 2
    codes = [n.code for n in notices]
    assert "fare_transfer_rule_without_transfer_count" in codes
    assert "fare_transfer_rule_with_forbidden_transfer_count" in codes
    without = next(n for n in notices if n.code == "fare_transfer_rule_without_transfer_count")
    forbidden = next(n for n in notices if n.code == "fare_transfer_rule_with_forbidden_transfer_count")
    assert without.fields["csv_row_number"] == 2
    assert forbidden.fields["csv_row_number"] == 3


def test_valid_transfer_count_two() -> None:
    """Self-loop with transfer_count=2 is valid — no notices."""
    feed = {
        "fare_transfer_rules": make_fare_transfer_rules(
            [{"from_leg_group_id": "x", "to_leg_group_id": "x", "transfer_count": 2, "csv_row_number": 2}]
        )
    }
    assert validate_fare_transfer_rule_transfer_count(feed, CTX) == []


def test_invalid_transfer_count_negative_three() -> None:
    """Self-loop with transfer_count=-3 emits invalid_transfer_count error."""
    feed = {
        "fare_transfer_rules": make_fare_transfer_rules(
            [{"from_leg_group_id": "x", "to_leg_group_id": "x", "transfer_count": -3, "csv_row_number": 2}]
        )
    }
    notices = validate_fare_transfer_rule_transfer_count(feed, CTX)
    assert len(notices) == 1
    assert notices[0].code == "fare_transfer_rule_invalid_transfer_count"
    assert notices[0].severity == Severity.ERROR
    assert notices[0].fields["csv_row_number"] == 2
    assert notices[0].fields["transfer_count"] == -3


def test_notice_fields_exact_values_invalid_count() -> None:
    """Verify exact fields for invalid_transfer_count notice."""
    feed = {
        "fare_transfer_rules": make_fare_transfer_rules(
            [{"from_leg_group_id": "a", "to_leg_group_id": "a", "transfer_count": 0, "csv_row_number": 7}]
        )
    }
    notices = validate_fare_transfer_rule_transfer_count(feed, CTX)
    assert len(notices) == 1
    assert notices[0].code == "fare_transfer_rule_invalid_transfer_count"
    assert notices[0].severity == Severity.ERROR
    assert notices[0].fields == {"csv_row_number": 7, "transfer_count": 0}


def test_notice_fields_exact_values_without_count() -> None:
    """Verify exact fields for without_transfer_count notice."""
    feed = {
        "fare_transfer_rules": make_fare_transfer_rules(
            [{"from_leg_group_id": "a", "to_leg_group_id": "a", "transfer_count": None, "csv_row_number": 9}]
        )
    }
    notices = validate_fare_transfer_rule_transfer_count(feed, CTX)
    assert len(notices) == 1
    assert notices[0].code == "fare_transfer_rule_without_transfer_count"
    assert notices[0].severity == Severity.ERROR
    assert notices[0].fields == {"csv_row_number": 9}


def test_notice_fields_exact_values_forbidden_count() -> None:
    """Verify exact fields for with_forbidden_transfer_count notice."""
    feed = {
        "fare_transfer_rules": make_fare_transfer_rules(
            [{"from_leg_group_id": "a", "to_leg_group_id": "b", "transfer_count": 1, "csv_row_number": 11}]
        )
    }
    notices = validate_fare_transfer_rule_transfer_count(feed, CTX)
    assert len(notices) == 1
    assert notices[0].code == "fare_transfer_rule_with_forbidden_transfer_count"
    assert notices[0].severity == Severity.ERROR
    assert notices[0].fields == {"csv_row_number": 11}


def test_mixed_rows_only_violating_rows_emit_notices() -> None:
    """Four rows — only the two violating rows produce notices."""
    feed = {
        "fare_transfer_rules": make_fare_transfer_rules(
            [
                {"from_leg_group_id": "a", "to_leg_group_id": "a", "transfer_count": 1, "csv_row_number": 1},
                {"from_leg_group_id": "a", "to_leg_group_id": "a", "transfer_count": None, "csv_row_number": 2},
                {"from_leg_group_id": "a", "to_leg_group_id": "b", "transfer_count": None, "csv_row_number": 3},
                {"from_leg_group_id": "a", "to_leg_group_id": "b", "transfer_count": 2, "csv_row_number": 4},
            ]
        )
    }
    notices = validate_fare_transfer_rule_transfer_count(feed, CTX)
    assert len(notices) == 2
    codes = [n.code for n in notices]
    assert "fare_transfer_rule_without_transfer_count" in codes
    assert "fare_transfer_rule_with_forbidden_transfer_count" in codes
    without = next(n for n in notices if n.code == "fare_transfer_rule_without_transfer_count")
    forbidden = next(n for n in notices if n.code == "fare_transfer_rule_with_forbidden_transfer_count")
    assert without.fields["csv_row_number"] == 2
    assert forbidden.fields["csv_row_number"] == 4
