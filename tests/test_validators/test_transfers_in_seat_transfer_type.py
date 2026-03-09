"""Tests for validate_transfers_in_seat_transfer_type."""

from datetime import date

import polars as pl
import pytest

from gtfs_validator.context import ValidationContext
from gtfs_validator.notices import Severity
from gtfs_validator.validators.transfers_in_seat_transfer_type import (
    validate_transfers_in_seat_transfer_type,
)


def _ctx() -> ValidationContext:
    return ValidationContext(country_code="US", date_for_validation=date(2024, 1, 1))


def _make_transfers(rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "transfer_type": [r.get("transfer_type") for r in rows],
            "from_stop_id": [r.get("from_stop_id") for r in rows],
            "to_stop_id": [r.get("to_stop_id") for r in rows],
            "from_trip_id": [r.get("from_trip_id") for r in rows],
            "to_trip_id": [r.get("to_trip_id") for r in rows],
            "csv_row_number": [r.get("csv_row_number", 2) for r in rows],
        }
    )


def _make_stops(rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "stop_id": [r["stop_id"] for r in rows],
            "location_type": [r.get("location_type", 0) for r in rows],
        }
    )


def _make_stop_times(rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "trip_id": [r["trip_id"] for r in rows],
            "stop_id": [r["stop_id"] for r in rows],
            "stop_sequence": [r["stop_sequence"] for r in rows],
        }
    )


def test_valid_in_seat_transfer_no_notices() -> None:
    """Happy path: correct stop positions, no station stops."""
    # s0 is last stop in t0; s1 is first stop in t1
    stops = _make_stops([
        {"stop_id": "s0", "location_type": 0},
        {"stop_id": "s1", "location_type": 0},
        {"stop_id": "s_other", "location_type": 0},
    ])
    stop_times = _make_stop_times([
        {"trip_id": "t0", "stop_id": "s_other", "stop_sequence": 1},
        {"trip_id": "t0", "stop_id": "s0", "stop_sequence": 2},
        {"trip_id": "t1", "stop_id": "s1", "stop_sequence": 1},
        {"trip_id": "t1", "stop_id": "s_other", "stop_sequence": 2},
    ])
    transfers = _make_transfers([
        {
            "transfer_type": 4,
            "from_stop_id": "s0",
            "to_stop_id": "s1",
            "from_trip_id": "t0",
            "to_trip_id": "t1",
        }
    ])
    feed = {"transfers": transfers, "stops": stops, "stop_times": stop_times}
    result = validate_transfers_in_seat_transfer_type(feed, _ctx())
    assert result == []


def test_in_seat_transfer_to_station_emits_error() -> None:
    """to_stop_id references a STATION (location_type=1) — one error notice."""
    stops = _make_stops([
        {"stop_id": "s0", "location_type": 0},
        {"stop_id": "s1", "location_type": 1},  # STATION
        {"stop_id": "s_other", "location_type": 0},
    ])
    stop_times = _make_stop_times([
        {"trip_id": "t0", "stop_id": "s_other", "stop_sequence": 1},
        {"trip_id": "t0", "stop_id": "s0", "stop_sequence": 2},
        {"trip_id": "t1", "stop_id": "s1", "stop_sequence": 1},
        {"trip_id": "t1", "stop_id": "s_other", "stop_sequence": 2},
    ])
    transfers = _make_transfers([
        {
            "transfer_type": 4,
            "from_stop_id": "s0",
            "to_stop_id": "s1",
            "from_trip_id": "t0",
            "to_trip_id": "t1",
        }
    ])
    feed = {"transfers": transfers, "stops": stops, "stop_times": stop_times}
    result = validate_transfers_in_seat_transfer_type(feed, _ctx())

    assert len(result) == 1
    notice = result[0]
    assert notice.code == "transfer_with_invalid_stop_location_type"
    assert notice.severity == Severity.ERROR
    assert notice.fields["stop_id_field_name"] == "to_stop_id"
    assert notice.fields["stop_id"] == "s1"
    assert notice.fields["location_type_value"] == 1
    assert notice.fields["location_type_name"] == "STATION"


def test_suspicious_mid_trip_in_seat_both_directions() -> None:
    """Both from_stop_id and to_stop_id are at wrong positions — two warnings."""
    stops = _make_stops([
        {"stop_id": "s0", "location_type": 0},
        {"stop_id": "s1", "location_type": 0},
        {"stop_id": "s_other", "location_type": 0},
    ])
    # s0 is FIRST in t0 (not last); s1 is LAST in t1 (not first)
    stop_times = _make_stop_times([
        {"trip_id": "t0", "stop_id": "s0", "stop_sequence": 1},
        {"trip_id": "t0", "stop_id": "s_other", "stop_sequence": 2},
        {"trip_id": "t1", "stop_id": "s_other", "stop_sequence": 1},
        {"trip_id": "t1", "stop_id": "s1", "stop_sequence": 2},
    ])
    transfers = _make_transfers([
        {
            "transfer_type": 4,
            "from_stop_id": "s0",
            "to_stop_id": "s1",
            "from_trip_id": "t0",
            "to_trip_id": "t1",
        }
    ])
    feed = {"transfers": transfers, "stops": stops, "stop_times": stop_times}
    result = validate_transfers_in_seat_transfer_type(feed, _ctx())

    assert len(result) == 2
    assert all(n.code == "transfer_with_suspicious_mid_trip_in_seat" for n in result)
    assert all(n.severity == Severity.WARNING for n in result)

    from_notice = result[0]
    assert from_notice.fields["trip_id_field_name"] == "from_trip_id"
    assert from_notice.fields["trip_id"] == "t0"
    assert from_notice.fields["stop_id_field_name"] == "from_stop_id"
    assert from_notice.fields["stop_id"] == "s0"

    to_notice = result[1]
    assert to_notice.fields["trip_id_field_name"] == "to_trip_id"
    assert to_notice.fields["trip_id"] == "t1"
    assert to_notice.fields["stop_id_field_name"] == "to_stop_id"
    assert to_notice.fields["stop_id"] == "s1"


def test_missing_from_trip_id_emits_missing_required_field() -> None:
    """from_trip_id=None on in-seat row emits missing_required_field."""
    transfers = _make_transfers([
        {
            "transfer_type": 4,
            "from_trip_id": None,
            "to_trip_id": "t1",
            "from_stop_id": "s0",
            "to_stop_id": "s1",
            "csv_row_number": 5,
        }
    ])
    feed = {"transfers": transfers}
    result = validate_transfers_in_seat_transfer_type(feed, _ctx())

    assert len(result) == 1
    notice = result[0]
    assert notice.code == "missing_required_field"
    assert notice.severity == Severity.ERROR
    assert notice.fields["field_name"] == "from_trip_id"
    assert notice.fields["csv_row_number"] == 5
    assert notice.fields["filename"] == "transfers.txt"


def test_missing_both_trip_ids_emits_two_notices() -> None:
    """Both trip IDs missing — two missing_required_field notices."""
    transfers = _make_transfers([
        {
            "transfer_type": 5,
            "from_trip_id": None,
            "to_trip_id": None,
            "from_stop_id": "s0",
            "to_stop_id": "s1",
        }
    ])
    feed = {"transfers": transfers}
    result = validate_transfers_in_seat_transfer_type(feed, _ctx())

    assert len(result) == 2
    assert all(n.code == "missing_required_field" for n in result)
    field_names = [n.fields["field_name"] for n in result]
    assert "from_trip_id" in field_names
    assert "to_trip_id" in field_names


def test_null_transfer_type_row_skipped() -> None:
    """transfer_type=None is excluded by the in-seat filter."""
    transfers = _make_transfers([
        {
            "transfer_type": None,
            "from_trip_id": None,
            "to_trip_id": None,
            "from_stop_id": "s0",
            "to_stop_id": "s1",
        }
    ])
    feed = {"transfers": transfers}
    result = validate_transfers_in_seat_transfer_type(feed, _ctx())
    assert result == []


@pytest.mark.parametrize("transfer_type", [0, 1, 2, 3])
def test_non_in_seat_transfer_type_skipped(transfer_type: int) -> None:
    """Non-in-seat transfer types (0-3) are not processed."""
    transfers = _make_transfers([
        {
            "transfer_type": transfer_type,
            "from_trip_id": None,
            "to_trip_id": None,
            "from_stop_id": "s0",
            "to_stop_id": "s1",
        }
    ])
    feed = {"transfers": transfers}
    result = validate_transfers_in_seat_transfer_type(feed, _ctx())
    assert result == []


def test_stop_not_in_stops_table_skips_location_and_position_checks() -> None:
    """Unknown stop IDs silently skip stop-related checks."""
    stop_times = _make_stop_times([])
    transfers = _make_transfers([
        {
            "transfer_type": 4,
            "from_stop_id": "unknown",
            "to_stop_id": "unknown",
            "from_trip_id": "t0",
            "to_trip_id": "t1",
        }
    ])
    feed = {
        "transfers": transfers,
        "stops": _make_stops([]),
        "stop_times": stop_times,
    }
    result = validate_transfers_in_seat_transfer_type(feed, _ctx())
    assert result == []


def test_stops_table_absent_skips_stop_checks() -> None:
    """No stops key in feed — stop checks silently skipped."""
    transfers = _make_transfers([
        {
            "transfer_type": 4,
            "from_stop_id": "s0",
            "to_stop_id": "s1",
            "from_trip_id": "t0",
            "to_trip_id": "t1",
        }
    ])
    # No "stops" in feed
    feed = {"transfers": transfers}
    result = validate_transfers_in_seat_transfer_type(feed, _ctx())
    # No stop-related notices (no missing_required_field because trip IDs are present)
    assert result == []


def test_stop_times_absent_skips_position_check() -> None:
    """No stop_times key in feed — position check silently skipped."""
    stops = _make_stops([
        {"stop_id": "s0", "location_type": 0},
        {"stop_id": "s1", "location_type": 0},
    ])
    transfers = _make_transfers([
        {
            "transfer_type": 4,
            "from_stop_id": "s0",
            "to_stop_id": "s1",
            "from_trip_id": "t0",
            "to_trip_id": "t1",
        }
    ])
    # No "stop_times" in feed
    feed = {"transfers": transfers, "stops": stops}
    result = validate_transfers_in_seat_transfer_type(feed, _ctx())
    assert result == []


def test_stop_in_stop_times_but_not_at_terminal_position() -> None:
    """from_stop_id is mid-trip (not last) — one warning for FROM direction."""
    stops = _make_stops([
        {"stop_id": "s0", "location_type": 0},
        {"stop_id": "s1", "location_type": 0},
        {"stop_id": "s2", "location_type": 0},
        {"stop_id": "s_ok", "location_type": 0},
    ])
    # t0: [s0 seq=1, s1 seq=2, s2 seq=3] — from_stop_id=s1 is mid-trip, not last
    # t1: [s_ok seq=1, s2 seq=2] — to_stop_id=s_ok is first (valid)
    stop_times = _make_stop_times([
        {"trip_id": "t0", "stop_id": "s0", "stop_sequence": 1},
        {"trip_id": "t0", "stop_id": "s1", "stop_sequence": 2},
        {"trip_id": "t0", "stop_id": "s2", "stop_sequence": 3},
        {"trip_id": "t1", "stop_id": "s_ok", "stop_sequence": 1},
        {"trip_id": "t1", "stop_id": "s2", "stop_sequence": 2},
    ])
    transfers = _make_transfers([
        {
            "transfer_type": 4,
            "from_stop_id": "s1",
            "to_stop_id": "s_ok",
            "from_trip_id": "t0",
            "to_trip_id": "t1",
        }
    ])
    feed = {"transfers": transfers, "stops": stops, "stop_times": stop_times}
    result = validate_transfers_in_seat_transfer_type(feed, _ctx())

    assert len(result) == 1
    notice = result[0]
    assert notice.code == "transfer_with_suspicious_mid_trip_in_seat"
    assert notice.fields["stop_id_field_name"] == "from_stop_id"


def test_stop_not_in_trip_stop_times_skips_position_check() -> None:
    """Stop appears in stops table but not in trip stop-times — position check deferred."""
    stops = _make_stops([
        {"stop_id": "s0", "location_type": 0},
        {"stop_id": "s1", "location_type": 0},
    ])
    # t0 has stop-times but s0 is NOT among them
    stop_times = _make_stop_times([
        {"trip_id": "t0", "stop_id": "s_other", "stop_sequence": 1},
    ])
    transfers = _make_transfers([
        {
            "transfer_type": 4,
            "from_stop_id": "s0",
            "to_stop_id": "s1",
            "from_trip_id": "t0",
            "to_trip_id": "t1",
        }
    ])
    feed = {"transfers": transfers, "stops": stops, "stop_times": stop_times}
    result = validate_transfers_in_seat_transfer_type(feed, _ctx())
    # No position notice — deferred to TransfersTripReferenceValidator
    position_notices = [n for n in result if n.code == "transfer_with_suspicious_mid_trip_in_seat"]
    assert position_notices == []


def test_transfers_absent_returns_empty() -> None:
    """No transfers key in feed returns empty list."""
    result = validate_transfers_in_seat_transfer_type({}, _ctx())
    assert result == []


def test_transfers_empty_returns_empty() -> None:
    """Empty transfers DataFrame returns empty list."""
    transfers = pl.DataFrame(
        {
            "transfer_type": pl.Series([], dtype=pl.Int64),
            "from_stop_id": pl.Series([], dtype=pl.Utf8),
            "to_stop_id": pl.Series([], dtype=pl.Utf8),
            "from_trip_id": pl.Series([], dtype=pl.Utf8),
            "to_trip_id": pl.Series([], dtype=pl.Utf8),
            "csv_row_number": pl.Series([], dtype=pl.Int64),
        }
    )
    feed = {"transfers": transfers}
    result = validate_transfers_in_seat_transfer_type(feed, _ctx())
    assert result == []


def test_transfer_type_column_absent_returns_empty() -> None:
    """transfers DataFrame without transfer_type column returns empty list."""
    transfers = pl.DataFrame(
        {
            "from_stop_id": ["s0"],
            "to_stop_id": ["s1"],
        }
    )
    feed = {"transfers": transfers}
    result = validate_transfers_in_seat_transfer_type(feed, _ctx())
    assert result == []


def test_csv_row_number_carried_into_notices() -> None:
    """csv_row_number is correctly carried into notice fields."""
    transfers = _make_transfers([
        {
            "transfer_type": 4,
            "from_trip_id": None,
            "to_trip_id": None,
            "from_stop_id": "s0",
            "to_stop_id": "s1",
            "csv_row_number": 99,
        }
    ])
    feed = {"transfers": transfers}
    result = validate_transfers_in_seat_transfer_type(feed, _ctx())

    assert len(result) == 2
    assert all(n.fields["csv_row_number"] == 99 for n in result)


def test_notice_ordering_from_before_to() -> None:
    """FROM direction notice appears before TO direction notice."""
    transfers = _make_transfers([
        {
            "transfer_type": 4,
            "from_trip_id": None,
            "to_trip_id": None,
            "from_stop_id": "s0",
            "to_stop_id": "s1",
        }
    ])
    feed = {"transfers": transfers}
    result = validate_transfers_in_seat_transfer_type(feed, _ctx())

    assert len(result) == 2
    assert result[0].fields["field_name"] == "from_trip_id"
    assert result[1].fields["field_name"] == "to_trip_id"


def test_transfer_type_5_also_triggers_checks() -> None:
    """transfer_type=5 also activates in-seat checks — valid case returns no notices."""
    stops = _make_stops([
        {"stop_id": "s0", "location_type": 0},
        {"stop_id": "s1", "location_type": 0},
        {"stop_id": "s_other", "location_type": 0},
    ])
    stop_times = _make_stop_times([
        {"trip_id": "t0", "stop_id": "s_other", "stop_sequence": 1},
        {"trip_id": "t0", "stop_id": "s0", "stop_sequence": 2},
        {"trip_id": "t1", "stop_id": "s1", "stop_sequence": 1},
        {"trip_id": "t1", "stop_id": "s_other", "stop_sequence": 2},
    ])
    transfers = _make_transfers([
        {
            "transfer_type": 5,
            "from_stop_id": "s0",
            "to_stop_id": "s1",
            "from_trip_id": "t0",
            "to_trip_id": "t1",
        }
    ])
    feed = {"transfers": transfers, "stops": stops, "stop_times": stop_times}
    result = validate_transfers_in_seat_transfer_type(feed, _ctx())
    assert result == []


def test_combined_station_and_missing_trip_id_same_row() -> None:
    """One row: missing from_trip_id AND to_stop_id is a STATION — two notices."""
    stops = _make_stops([
        {"stop_id": "s0", "location_type": 0},
        {"stop_id": "s1", "location_type": 1},  # STATION
        {"stop_id": "s_other", "location_type": 0},
    ])
    stop_times = _make_stop_times([
        {"trip_id": "t0", "stop_id": "s_other", "stop_sequence": 1},
        {"trip_id": "t0", "stop_id": "s0", "stop_sequence": 2},
        {"trip_id": "t1", "stop_id": "s1", "stop_sequence": 1},
        {"trip_id": "t1", "stop_id": "s_other", "stop_sequence": 2},
    ])
    transfers = _make_transfers([
        {
            "transfer_type": 4,
            "from_stop_id": "s0",
            "to_stop_id": "s1",
            "from_trip_id": None,  # missing
            "to_trip_id": "t1",
        }
    ])
    feed = {"transfers": transfers, "stops": stops, "stop_times": stop_times}
    result = validate_transfers_in_seat_transfer_type(feed, _ctx())

    assert len(result) == 2
    codes = [n.code for n in result]
    assert "missing_required_field" in codes
    assert "transfer_with_invalid_stop_location_type" in codes

    missing_notice = next(n for n in result if n.code == "missing_required_field")
    assert missing_notice.fields["field_name"] == "from_trip_id"

    station_notice = next(n for n in result if n.code == "transfer_with_invalid_stop_location_type")
    assert station_notice.fields["stop_id_field_name"] == "to_stop_id"
