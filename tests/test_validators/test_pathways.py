"""Tests for pathway validators."""

from __future__ import annotations

from datetime import date

import polars as pl

from gtfs_validator.context import ValidationContext
from gtfs_validator.notices import Severity
from gtfs_validator.validators.pathways import validate_bidirectional_exit_gate

CTX = ValidationContext(country_code="US", date_for_validation=date(2024, 6, 1))


# ------------------------------------------------------------------
# Test 1: bidirectional exit gate generates notice
# ------------------------------------------------------------------


def test_bidirectional_exit_gate_generates_notice() -> None:
    feed = {
        "pathways": pl.DataFrame({
            "csv_row_number": [1],
            "pathway_mode": [7],
            "is_bidirectional": [1],
        })
    }
    notices = validate_bidirectional_exit_gate(feed, CTX)
    assert len(notices) == 1
    n = notices[0]
    assert n.code == "bidirectional_exit_gate"
    assert n.severity == Severity.ERROR
    assert n.fields == {
        "csv_row_number": 1,
        "pathway_mode": 7,
        "is_bidirectional": 1,
    }


# ------------------------------------------------------------------
# Test 2: unidirectional exit gate -> no notice
# ------------------------------------------------------------------


def test_unidirectional_exit_gate_no_notice() -> None:
    feed = {
        "pathways": pl.DataFrame({
            "csv_row_number": [1],
            "pathway_mode": [7],
            "is_bidirectional": [0],
        })
    }
    notices = validate_bidirectional_exit_gate(feed, CTX)
    assert len(notices) == 0


# ------------------------------------------------------------------
# Test 3: bidirectional non-exit-gate modes -> no notice
# ------------------------------------------------------------------


def test_bidirectional_non_exit_gate_no_notice() -> None:
    feed = {
        "pathways": pl.DataFrame({
            "csv_row_number": list(range(1, 7)),
            "pathway_mode": [1, 2, 3, 4, 5, 6],
            "is_bidirectional": [1, 1, 1, 1, 1, 1],
        })
    }
    notices = validate_bidirectional_exit_gate(feed, CTX)
    assert len(notices) == 0


# ------------------------------------------------------------------
# Test 4: missing pathways table
# ------------------------------------------------------------------


def test_missing_pathways_table() -> None:
    feed: dict[str, pl.DataFrame] = {}
    notices = validate_bidirectional_exit_gate(feed, CTX)
    assert len(notices) == 0


# ------------------------------------------------------------------
# Test 5: empty DataFrame
# ------------------------------------------------------------------


def test_empty_dataframe() -> None:
    feed = {
        "pathways": pl.DataFrame(
            schema={
                "csv_row_number": pl.Int64,
                "pathway_mode": pl.Int64,
                "is_bidirectional": pl.Int64,
            }
        )
    }
    notices = validate_bidirectional_exit_gate(feed, CTX)
    assert len(notices) == 0


# ------------------------------------------------------------------
# Test 6: multiple rows mixed — only violating row produces notice
# ------------------------------------------------------------------


def test_multiple_rows_mixed() -> None:
    feed = {
        "pathways": pl.DataFrame({
            "csv_row_number": [1, 2, 3],
            "pathway_mode": [7, 7, 6],
            "is_bidirectional": [1, 0, 1],
        })
    }
    notices = validate_bidirectional_exit_gate(feed, CTX)
    assert len(notices) == 1
    assert notices[0].fields["csv_row_number"] == 1


# ------------------------------------------------------------------
# Test 7: multiple violations
# ------------------------------------------------------------------


def test_multiple_violations() -> None:
    feed = {
        "pathways": pl.DataFrame({
            "csv_row_number": [2, 5],
            "pathway_mode": [7, 7],
            "is_bidirectional": [1, 1],
        })
    }
    notices = validate_bidirectional_exit_gate(feed, CTX)
    assert len(notices) == 2
    row_numbers = {n.fields["csv_row_number"] for n in notices}
    assert row_numbers == {2, 5}
