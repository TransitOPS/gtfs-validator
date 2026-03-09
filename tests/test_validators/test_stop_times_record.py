"""Tests for StopTimesRecordValidator."""

from datetime import date

import polars as pl
import pytest

from gtfs_validator.context import ValidationContext
from gtfs_validator.notices import Severity
from gtfs_validator.validators.stop_times_record import validate_stop_times_record

CTX = ValidationContext(country_code="US", date_for_validation=date(2026, 3, 8))

_BASE_SCHEMA = {
    "csv_row_number": pl.Int64,
    "trip_id": pl.Utf8,
    "start_pickup_drop_off_window": pl.Utf8,
    "end_pickup_drop_off_window": pl.Utf8,
    "pickup_type": pl.Int64,
    "drop_off_type": pl.Int64,
}


def make_stop_times(
    rows: list[dict],
    include_location_group: bool = True,
    include_location_id: bool = True,
) -> pl.DataFrame:
    """Build a stop_times DataFrame from a list of row dicts.

    Optional columns ``location_group_id`` and ``location_id`` are included by
    default; pass ``False`` to omit them from the schema entirely.
    """
    schema = dict(_BASE_SCHEMA)
    if include_location_group:
        schema["location_group_id"] = pl.Utf8
    if include_location_id:
        schema["location_id"] = pl.Utf8

    if not rows:
        return pl.DataFrame(schema=schema)

    # Build columns dict from rows, filling missing keys with None
    cols: dict[str, list] = {col: [] for col in schema}
    for row in rows:
        for col in schema:
            cols[col].append(row.get(col))

    return pl.DataFrame(cols, schema=schema)


# ---------------------------------------------------------------------------
# Guard tests
# ---------------------------------------------------------------------------


def test_stop_times_absent_returns_empty() -> None:
    """Guard 1: missing stop_times key yields no notices."""
    result = validate_stop_times_record({}, CTX)
    assert result == []


@pytest.mark.parametrize(
    "missing_col",
    [
        "start_pickup_drop_off_window",
        "end_pickup_drop_off_window",
        "pickup_type",
        "drop_off_type",
    ],
)
def test_missing_required_column_returns_empty(missing_col: str) -> None:
    """Guard 2: if any of the four required columns is absent, skip entirely."""
    schema = dict(_BASE_SCHEMA)
    del schema[missing_col]
    st = pl.DataFrame(schema=schema)
    result = validate_stop_times_record({"stop_times": st}, CTX)
    assert result == []


def test_empty_stop_times_returns_empty() -> None:
    """Guard 3: empty stop_times table yields no notices."""
    st = make_stop_times([])
    result = validate_stop_times_record({"stop_times": st}, CTX)
    assert result == []


# ---------------------------------------------------------------------------
# No-notice cases
# ---------------------------------------------------------------------------


def test_regular_pickup_type_no_notice() -> None:
    """pickup_type=0 (REGULAR) does not trigger the notice even with window times set."""
    st = make_stop_times([
        {
            "csv_row_number": 1,
            "trip_id": "trip1",
            "location_group_id": "locationGroupId1",
            "location_id": "locationId1",
            "pickup_type": 0,
            "drop_off_type": 0,
            "start_pickup_drop_off_window": "08:00:00",
            "end_pickup_drop_off_window": "09:00:00",
        }
    ])
    result = validate_stop_times_record({"stop_times": st}, CTX)
    assert result == []


def test_must_phone_two_records_same_trip_no_notice() -> None:
    """Two rows with MUST_PHONE and same trip_id: count == 2, no notice."""
    st = make_stop_times([
        {
            "csv_row_number": 1,
            "trip_id": "trip3",
            "location_group_id": "lg1",
            "location_id": "li1",
            "pickup_type": 2,
            "drop_off_type": 2,
            "start_pickup_drop_off_window": "08:00:00",
            "end_pickup_drop_off_window": "09:00:00",
        },
        {
            "csv_row_number": 2,
            "trip_id": "trip3",
            "location_group_id": "lg2",
            "location_id": "li2",
            "pickup_type": 2,
            "drop_off_type": 2,
            "start_pickup_drop_off_window": "09:00:00",
            "end_pickup_drop_off_window": "10:00:00",
        },
    ])
    result = validate_stop_times_record({"stop_times": st}, CTX)
    assert result == []


def test_must_phone_drop_off_only_no_notice() -> None:
    """pickup_type=2 but drop_off_type=0: conjunction fails, no notice."""
    st = make_stop_times([
        {
            "csv_row_number": 1,
            "trip_id": "trip4",
            "pickup_type": 2,
            "drop_off_type": 0,
            "start_pickup_drop_off_window": "08:00:00",
            "end_pickup_drop_off_window": "09:00:00",
        }
    ])
    result = validate_stop_times_record({"stop_times": st}, CTX)
    assert result == []


def test_must_phone_without_window_times_no_notice() -> None:
    """Both window times are None: condition fails, no notice emitted."""
    st = make_stop_times([
        {
            "csv_row_number": 1,
            "trip_id": "trip5",
            "pickup_type": 2,
            "drop_off_type": 2,
            "start_pickup_drop_off_window": None,
            "end_pickup_drop_off_window": None,
        }
    ])
    result = validate_stop_times_record({"stop_times": st}, CTX)
    assert result == []


# ---------------------------------------------------------------------------
# Notice-emitting cases
# ---------------------------------------------------------------------------


def test_must_phone_singleton_trip_emits_notice() -> None:
    """All four conditions met + singleton trip → exactly one notice."""
    st = make_stop_times([
        {
            "csv_row_number": 1,
            "trip_id": "trip2",
            "location_group_id": "locationGroupId1",
            "location_id": "locationId1",
            "pickup_type": 2,
            "drop_off_type": 2,
            "start_pickup_drop_off_window": "08:00:00",
            "end_pickup_drop_off_window": "09:00:00",
        }
    ])
    result = validate_stop_times_record({"stop_times": st}, CTX)
    assert len(result) == 1
    notice = result[0]
    assert notice.code == "missing_stop_times_record"
    assert notice.severity == Severity.ERROR
    assert notice.fields["csv_row_number"] == 1
    assert notice.fields["trip_id"] == "trip2"
    assert notice.fields["location_group_id"] == "locationGroupId1"
    assert notice.fields["location_id"] == "locationId1"


def test_location_group_id_and_location_id_null_in_notice() -> None:
    """Columns present but values None → notice payload contains None."""
    st = make_stop_times([
        {
            "csv_row_number": 1,
            "trip_id": "trip6",
            "location_group_id": None,
            "location_id": None,
            "pickup_type": 2,
            "drop_off_type": 2,
            "start_pickup_drop_off_window": "08:00:00",
            "end_pickup_drop_off_window": "09:00:00",
        }
    ])
    result = validate_stop_times_record({"stop_times": st}, CTX)
    assert len(result) == 1
    assert result[0].fields["location_group_id"] is None
    assert result[0].fields["location_id"] is None


def test_location_columns_absent_from_schema() -> None:
    """Columns not present at all → notice payload uses None (no KeyError)."""
    st = make_stop_times(
        [
            {
                "csv_row_number": 1,
                "trip_id": "trip7",
                "pickup_type": 2,
                "drop_off_type": 2,
                "start_pickup_drop_off_window": "08:00:00",
                "end_pickup_drop_off_window": "09:00:00",
            }
        ],
        include_location_group=False,
        include_location_id=False,
    )
    result = validate_stop_times_record({"stop_times": st}, CTX)
    assert len(result) == 1
    assert result[0].fields["location_group_id"] is None
    assert result[0].fields["location_id"] is None


def test_multiple_trips_each_singleton_emits_one_notice_each() -> None:
    """Three singleton MUST_PHONE trips → three notices (one per trip)."""
    st = make_stop_times([
        {
            "csv_row_number": 1,
            "trip_id": "tripA",
            "pickup_type": 2,
            "drop_off_type": 2,
            "start_pickup_drop_off_window": "08:00:00",
            "end_pickup_drop_off_window": "09:00:00",
        },
        {
            "csv_row_number": 2,
            "trip_id": "tripB",
            "pickup_type": 2,
            "drop_off_type": 2,
            "start_pickup_drop_off_window": "09:00:00",
            "end_pickup_drop_off_window": "10:00:00",
        },
        {
            "csv_row_number": 3,
            "trip_id": "tripC",
            "pickup_type": 2,
            "drop_off_type": 2,
            "start_pickup_drop_off_window": "10:00:00",
            "end_pickup_drop_off_window": "11:00:00",
        },
    ])
    result = validate_stop_times_record({"stop_times": st}, CTX)
    assert len(result) == 3
    trip_ids = {n.fields["trip_id"] for n in result}
    assert trip_ids == {"tripA", "tripB", "tripC"}
    for notice in result:
        assert notice.code == "missing_stop_times_record"
        assert notice.severity == Severity.ERROR


def test_mixed_trips_only_singleton_emits_notice() -> None:
    """Only the qualifying singleton MUST_PHONE trip emits a notice; others are silent."""
    st = make_stop_times([
        # tripOK: two records → count == 2, no notice
        {
            "csv_row_number": 1,
            "trip_id": "tripOK",
            "pickup_type": 2,
            "drop_off_type": 2,
            "start_pickup_drop_off_window": "08:00:00",
            "end_pickup_drop_off_window": "09:00:00",
        },
        {
            "csv_row_number": 2,
            "trip_id": "tripOK",
            "pickup_type": 2,
            "drop_off_type": 2,
            "start_pickup_drop_off_window": "09:00:00",
            "end_pickup_drop_off_window": "10:00:00",
        },
        # tripBad: singleton MUST_PHONE → notice
        {
            "csv_row_number": 3,
            "trip_id": "tripBad",
            "pickup_type": 2,
            "drop_off_type": 2,
            "start_pickup_drop_off_window": "08:00:00",
            "end_pickup_drop_off_window": "09:00:00",
        },
        # tripNone: pickup_type=0 → conjunction fails
        {
            "csv_row_number": 4,
            "trip_id": "tripNone",
            "pickup_type": 0,
            "drop_off_type": 0,
            "start_pickup_drop_off_window": "08:00:00",
            "end_pickup_drop_off_window": "09:00:00",
        },
    ])
    result = validate_stop_times_record({"stop_times": st}, CTX)
    assert len(result) == 1
    assert result[0].fields["trip_id"] == "tripBad"
