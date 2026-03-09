"""Tests for StopTimesTripBlockOrderValidator."""

from datetime import date

import polars as pl
import pytest

from gtfs_validator.context import ValidationContext
from gtfs_validator.notices import Severity
from gtfs_validator.validators.stop_times_trip_block_order import (
    validate_stop_times_trip_block_order,
)

CTX = ValidationContext(country_code="US", date_for_validation=date(2024, 1, 1))


def make_stop_times(rows: list[dict]) -> pl.DataFrame:
    """Build a stop_times DataFrame from a list of row dicts.

    Provides default dtypes: trip_id Utf8, stop_sequence Int64,
    csv_row_number Int64. Rows are added in list order; csv_row_number
    must be set explicitly by the caller to reflect intended file order.
    """
    return pl.DataFrame(
        rows,
        schema={
            "trip_id": pl.Utf8,
            "stop_sequence": pl.Int64,
            "csv_row_number": pl.Int64,
        },
    )


def test_non_contiguous_trip_block_emits_notice() -> None:
    """Non-contiguous trip block emits one notice for the affected trip."""
    st = make_stop_times([
        {"trip_id": "0914", "stop_sequence": 1, "csv_row_number": 1},
        {"trip_id": "0914", "stop_sequence": 2, "csv_row_number": 2},
        {"trip_id": "0915", "stop_sequence": 1, "csv_row_number": 3},
        {"trip_id": "0914", "stop_sequence": 3, "csv_row_number": 4},
    ])
    feed = {"stop_times": st}
    notices = validate_stop_times_trip_block_order(feed, CTX)

    # Exactly one notice for trip 0914
    assert len(notices) == 1
    notice = notices[0]
    assert notice.code == "unsorted_stop_times"
    assert notice.severity == Severity.INFO
    assert notice.fields["trip_id"] == "0914"
    assert notice.fields["start_csv_row_number"] == 1
    assert notice.fields["end_csv_row_number"] == 4

    # Trip 0915 must not appear
    trip_ids = {n.fields["trip_id"] for n in notices}
    assert "0915" not in trip_ids


def test_non_increasing_stop_sequence_emits_notice() -> None:
    """Non-increasing stop_sequence triggers one notice."""
    st = make_stop_times([
        {"trip_id": "0916", "stop_sequence": 1, "csv_row_number": 1},
        {"trip_id": "0916", "stop_sequence": 3, "csv_row_number": 2},
        {"trip_id": "0916", "stop_sequence": 2, "csv_row_number": 3},
    ])
    feed = {"stop_times": st}
    notices = validate_stop_times_trip_block_order(feed, CTX)

    assert len(notices) == 1
    notice = notices[0]
    assert notice.code == "unsorted_stop_times"
    assert notice.severity == Severity.INFO
    assert notice.fields["trip_id"] == "0916"
    assert notice.fields["start_csv_row_number"] == 1
    assert notice.fields["end_csv_row_number"] == 3


def test_stop_times_absent_returns_empty() -> None:
    """Missing stop_times table returns no notices."""
    notices = validate_stop_times_trip_block_order({}, CTX)
    assert notices == []


def test_empty_stop_times_returns_empty() -> None:
    """Empty stop_times DataFrame returns no notices."""
    st = pl.DataFrame(
        schema={
            "trip_id": pl.Utf8,
            "stop_sequence": pl.Int64,
            "csv_row_number": pl.Int64,
        }
    )
    notices = validate_stop_times_trip_block_order({"stop_times": st}, CTX)
    assert notices == []


def test_null_trip_id_rows_are_skipped() -> None:
    """Rows with null trip_id are skipped and produce no notices."""
    st = make_stop_times([
        {"trip_id": None, "stop_sequence": 1, "csv_row_number": 1},
        {"trip_id": None, "stop_sequence": 2, "csv_row_number": 2},
    ])
    notices = validate_stop_times_trip_block_order({"stop_times": st}, CTX)
    assert notices == []


def test_single_row_trip_no_notice() -> None:
    """Single-row trip is always valid."""
    st = make_stop_times([
        {"trip_id": "T1", "stop_sequence": 5, "csv_row_number": 1},
    ])
    notices = validate_stop_times_trip_block_order({"stop_times": st}, CTX)
    assert notices == []


def test_contiguous_increasing_sequence_no_notice() -> None:
    """Perfectly contiguous, increasing sequence produces no notices."""
    st = make_stop_times([
        {"trip_id": "T1", "stop_sequence": 1, "csv_row_number": 1},
        {"trip_id": "T1", "stop_sequence": 2, "csv_row_number": 2},
        {"trip_id": "T1", "stop_sequence": 3, "csv_row_number": 3},
    ])
    notices = validate_stop_times_trip_block_order({"stop_times": st}, CTX)
    assert notices == []


def test_duplicate_stop_sequence_triggers_notice() -> None:
    """Duplicate stop_sequence values trigger a notice (1 <= 1)."""
    st = make_stop_times([
        {"trip_id": "T1", "stop_sequence": 1, "csv_row_number": 1},
        {"trip_id": "T1", "stop_sequence": 1, "csv_row_number": 2},
    ])
    notices = validate_stop_times_trip_block_order({"stop_times": st}, CTX)
    assert len(notices) == 1
    assert notices[0].fields["trip_id"] == "T1"
    assert notices[0].fields["start_csv_row_number"] == 1
    assert notices[0].fields["end_csv_row_number"] == 2


def test_both_violations_on_same_trip_emits_one_notice() -> None:
    """Trip with both non-contiguous and non-increasing sequence emits one notice."""
    # T1 rows at csv_row_numbers 1, 3, 4 (non-contiguous due to T2 at 2)
    # T1 sequence: 3, 1, 2 — also non-increasing
    st = make_stop_times([
        {"trip_id": "T1", "stop_sequence": 3, "csv_row_number": 1},
        {"trip_id": "T2", "stop_sequence": 1, "csv_row_number": 2},
        {"trip_id": "T1", "stop_sequence": 1, "csv_row_number": 3},
        {"trip_id": "T1", "stop_sequence": 2, "csv_row_number": 4},
    ])
    notices = validate_stop_times_trip_block_order({"stop_times": st}, CTX)
    trip_ids = [n.fields["trip_id"] for n in notices]
    assert trip_ids.count("T1") == 1


def test_non_contiguous_but_sorted_emits_notice() -> None:
    """Non-contiguous trip with sorted sequence still emits a notice."""
    # T1 at rows 1 and 3; T2 at row 2. T1 span=3, count=2.
    st = make_stop_times([
        {"trip_id": "T1", "stop_sequence": 1, "csv_row_number": 1},
        {"trip_id": "T2", "stop_sequence": 1, "csv_row_number": 2},
        {"trip_id": "T1", "stop_sequence": 2, "csv_row_number": 3},
    ])
    notices = validate_stop_times_trip_block_order({"stop_times": st}, CTX)
    assert len(notices) == 1
    assert notices[0].fields["trip_id"] == "T1"


def test_contiguous_but_unsorted_sequence_emits_notice() -> None:
    """Contiguous trip with non-increasing sequence emits a notice."""
    st = make_stop_times([
        {"trip_id": "T1", "stop_sequence": 3, "csv_row_number": 1},
        {"trip_id": "T1", "stop_sequence": 1, "csv_row_number": 2},
    ])
    notices = validate_stop_times_trip_block_order({"stop_times": st}, CTX)
    assert len(notices) == 1
    assert notices[0].fields["trip_id"] == "T1"


def test_multiple_violating_trips_each_emit_one_notice() -> None:
    """Multiple violating trips each get exactly one notice; valid trip is excluded."""
    # T1: non-contiguous (rows 1, 3; T2 at row 2)
    # T2: unsorted sequence (5 then 2), contiguous rows 2 and 4 with T1 at 3
    # T3: valid (contiguous rows 5-6, sorted)
    st = make_stop_times([
        {"trip_id": "T1", "stop_sequence": 1, "csv_row_number": 1},
        {"trip_id": "T2", "stop_sequence": 5, "csv_row_number": 2},
        {"trip_id": "T1", "stop_sequence": 2, "csv_row_number": 3},
        {"trip_id": "T2", "stop_sequence": 2, "csv_row_number": 4},
        {"trip_id": "T3", "stop_sequence": 1, "csv_row_number": 5},
        {"trip_id": "T3", "stop_sequence": 2, "csv_row_number": 6},
    ])
    notices = validate_stop_times_trip_block_order({"stop_times": st}, CTX)
    trip_ids = {n.fields["trip_id"] for n in notices}
    assert "T1" in trip_ids
    assert "T2" in trip_ids
    assert "T3" not in trip_ids
    assert len(notices) == 2


def test_all_trips_valid_no_notices() -> None:
    """Three clean trips produce no notices."""
    st = make_stop_times([
        {"trip_id": "A", "stop_sequence": 1, "csv_row_number": 1},
        {"trip_id": "A", "stop_sequence": 2, "csv_row_number": 2},
        {"trip_id": "B", "stop_sequence": 1, "csv_row_number": 3},
        {"trip_id": "B", "stop_sequence": 2, "csv_row_number": 4},
        {"trip_id": "C", "stop_sequence": 1, "csv_row_number": 5},
        {"trip_id": "C", "stop_sequence": 2, "csv_row_number": 6},
    ])
    notices = validate_stop_times_trip_block_order({"stop_times": st}, CTX)
    assert notices == []


def test_notice_fields_are_complete() -> None:
    """Notice fields dict contains exactly the required keys."""
    st = make_stop_times([
        {"trip_id": "T1", "stop_sequence": 3, "csv_row_number": 1},
        {"trip_id": "T1", "stop_sequence": 1, "csv_row_number": 2},
    ])
    notices = validate_stop_times_trip_block_order({"stop_times": st}, CTX)
    assert len(notices) == 1
    assert set(notices[0].fields.keys()) == {
        "trip_id",
        "start_csv_row_number",
        "end_csv_row_number",
    }


def test_only_null_and_valid_trips_mixed() -> None:
    """Null trip_id rows are skipped; valid trip with no issues produces no notices."""
    st = make_stop_times([
        {"trip_id": None, "stop_sequence": 1, "csv_row_number": 1},
        {"trip_id": None, "stop_sequence": 2, "csv_row_number": 2},
        {"trip_id": "T1", "stop_sequence": 1, "csv_row_number": 3},
        {"trip_id": "T1", "stop_sequence": 2, "csv_row_number": 4},
    ])
    notices = validate_stop_times_trip_block_order({"stop_times": st}, CTX)
    assert notices == []
