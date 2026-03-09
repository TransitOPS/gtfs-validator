"""Tests for validate_transfer_stop_ids_conditional."""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from gtfs_validator.context import ValidationContext
from gtfs_validator.notices import Severity
from gtfs_validator.validators.transfer_stop_ids_conditional import (
    validate_transfer_stop_ids_conditional,
)


def _ctx() -> ValidationContext:
    return ValidationContext(country_code="US", date_for_validation=date(2024, 1, 1))


def _make_transfers(rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "transfer_type": [r.get("transfer_type") for r in rows],
            "from_stop_id": [r.get("from_stop_id") for r in rows],
            "to_stop_id": [r.get("to_stop_id") for r in rows],
            "csv_row_number": [r.get("csv_row_number", 1) for r in rows],
        }
    )


@pytest.mark.parametrize("transfer_type", [4, 5])
def test_in_seat_transfer_types_exempt_no_notices(transfer_type: int) -> None:
    """In-seat transfer rows (type 4 or 5) are fully exempt even when both IDs absent."""
    feed = {
        "transfers": _make_transfers(
            [{"transfer_type": transfer_type, "from_stop_id": None, "to_stop_id": None}]
        )
    }
    result = validate_transfer_stop_ids_conditional(feed, _ctx())
    assert result == []


@pytest.mark.parametrize("transfer_type", [0, 1, 2, 3])
def test_non_in_seat_both_stop_ids_missing_two_notices(transfer_type: int) -> None:
    """Non-in-seat transfer with both stop IDs missing produces two notices."""
    feed = {
        "transfers": _make_transfers(
            [{"transfer_type": transfer_type, "from_stop_id": None, "to_stop_id": None}]
        )
    }
    result = validate_transfer_stop_ids_conditional(feed, _ctx())
    assert len(result) == 2
    field_names = {n.fields["field_name"] for n in result}
    assert field_names == {"from_stop_id", "to_stop_id"}
    for notice in result:
        assert notice.code == "missing_required_field"
        assert notice.severity == Severity.ERROR
        assert notice.fields["filename"] == "transfers.txt"


@pytest.mark.parametrize("transfer_type", [0, 1, 2, 3, 4, 5])
def test_non_in_seat_both_stop_ids_present_no_notices(transfer_type: int) -> None:
    """When both stop IDs are present, no notices emitted for any transfer type."""
    feed = {
        "transfers": _make_transfers(
            [
                {
                    "transfer_type": transfer_type,
                    "from_stop_id": "stop1",
                    "to_stop_id": "stop2",
                }
            ]
        )
    }
    result = validate_transfer_stop_ids_conditional(feed, _ctx())
    assert result == []


def test_null_transfer_type_row_skipped() -> None:
    """Row with null transfer_type is skipped entirely even with null stop IDs."""
    feed = {
        "transfers": _make_transfers(
            [{"transfer_type": None, "from_stop_id": None, "to_stop_id": None}]
        )
    }
    result = validate_transfer_stop_ids_conditional(feed, _ctx())
    assert result == []


def test_only_from_stop_id_missing_one_notice() -> None:
    """Only from_stop_id missing produces exactly one notice."""
    feed = {
        "transfers": _make_transfers(
            [{"transfer_type": 0, "from_stop_id": None, "to_stop_id": "stop2"}]
        )
    }
    result = validate_transfer_stop_ids_conditional(feed, _ctx())
    assert len(result) == 1
    assert result[0].fields["field_name"] == "from_stop_id"


def test_only_to_stop_id_missing_one_notice() -> None:
    """Only to_stop_id missing produces exactly one notice."""
    feed = {
        "transfers": _make_transfers(
            [{"transfer_type": 0, "from_stop_id": "stop1", "to_stop_id": None}]
        )
    }
    result = validate_transfer_stop_ids_conditional(feed, _ctx())
    assert len(result) == 1
    assert result[0].fields["field_name"] == "to_stop_id"


def test_transfers_absent_returns_empty() -> None:
    """Missing transfers table returns empty list."""
    result = validate_transfer_stop_ids_conditional({}, _ctx())
    assert result == []


def test_transfers_empty_returns_empty() -> None:
    """Empty transfers DataFrame returns empty list."""
    empty_df = pl.DataFrame(
        {
            "transfer_type": pl.Series([], dtype=pl.Int64),
            "from_stop_id": pl.Series([], dtype=pl.Utf8),
            "to_stop_id": pl.Series([], dtype=pl.Utf8),
            "csv_row_number": pl.Series([], dtype=pl.Int64),
        }
    )
    feed = {"transfers": empty_df}
    result = validate_transfer_stop_ids_conditional(feed, _ctx())
    assert result == []


def test_transfer_type_column_absent_returns_empty() -> None:
    """Missing transfer_type column triggers guard; no notices even with null stop IDs."""
    df = pl.DataFrame(
        {
            "from_stop_id": [None],
            "to_stop_id": [None],
            "csv_row_number": [1],
        }
    )
    feed = {"transfers": df}
    result = validate_transfer_stop_ids_conditional(feed, _ctx())
    assert result == []


def test_from_stop_id_column_absent_returns_empty() -> None:
    """Missing from_stop_id column triggers guard."""
    df = pl.DataFrame(
        {
            "transfer_type": [0],
            "to_stop_id": [None],
            "csv_row_number": [1],
        }
    )
    feed = {"transfers": df}
    result = validate_transfer_stop_ids_conditional(feed, _ctx())
    assert result == []


def test_to_stop_id_column_absent_returns_empty() -> None:
    """Missing to_stop_id column triggers guard."""
    df = pl.DataFrame(
        {
            "transfer_type": [0],
            "from_stop_id": [None],
            "csv_row_number": [1],
        }
    )
    feed = {"transfers": df}
    result = validate_transfer_stop_ids_conditional(feed, _ctx())
    assert result == []


def test_csv_row_number_carried_into_notice() -> None:
    """csv_row_number value is correctly carried into notice fields."""
    feed = {
        "transfers": _make_transfers(
            [
                {
                    "transfer_type": 1,
                    "from_stop_id": None,
                    "to_stop_id": None,
                    "csv_row_number": 42,
                }
            ]
        )
    }
    result = validate_transfer_stop_ids_conditional(feed, _ctx())
    assert len(result) == 2
    for notice in result:
        assert notice.fields["csv_row_number"] == 42


def test_multiple_rows_mixed_results() -> None:
    """Four rows with mixed conditions produce exactly two notices for the problematic row."""
    feed = {
        "transfers": _make_transfers(
            [
                {
                    "transfer_type": 0,
                    "from_stop_id": "stop1",
                    "to_stop_id": "stop2",
                    "csv_row_number": 1,
                },  # clean row
                {
                    "transfer_type": 4,
                    "from_stop_id": None,
                    "to_stop_id": None,
                    "csv_row_number": 2,
                },  # in-seat, exempt
                {
                    "transfer_type": None,
                    "from_stop_id": None,
                    "to_stop_id": None,
                    "csv_row_number": 3,
                },  # null type, skipped
                {
                    "transfer_type": 2,
                    "from_stop_id": None,
                    "to_stop_id": None,
                    "csv_row_number": 4,
                },  # non-in-seat, both missing
            ]
        )
    }
    result = validate_transfer_stop_ids_conditional(feed, _ctx())
    assert len(result) == 2
    row_numbers = {n.fields["csv_row_number"] for n in result}
    assert row_numbers == {4}
    field_names = [n.fields["field_name"] for n in result]
    assert "from_stop_id" in field_names
    assert "to_stop_id" in field_names


def test_notice_fields_complete() -> None:
    """All notice fields are populated correctly."""
    feed = {
        "transfers": _make_transfers(
            [
                {
                    "transfer_type": 3,
                    "from_stop_id": None,
                    "to_stop_id": "stop2",
                    "csv_row_number": 7,
                }
            ]
        )
    }
    result = validate_transfer_stop_ids_conditional(feed, _ctx())
    assert len(result) == 1
    notice = result[0]
    assert notice.code == "missing_required_field"
    assert notice.severity == Severity.ERROR
    assert notice.fields["filename"] == "transfers.txt"
    assert notice.fields["csv_row_number"] == 7
    assert notice.fields["field_name"] == "from_stop_id"
