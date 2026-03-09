"""Tests for TransfersStopTypeValidator (spec 075)."""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from gtfs_validator.context import ValidationContext
from gtfs_validator.notices import Severity
from gtfs_validator.validators.transfers_stop_type import validate_transfers_stop_type


def _ctx() -> ValidationContext:
    return ValidationContext(country_code="US", date_for_validation=date(2024, 1, 1))


def _make_transfers(rows: list[dict], include_transfer_type: bool = False) -> pl.DataFrame:
    data: dict[str, list] = {
        "from_stop_id": [r.get("from_stop_id") for r in rows],
        "to_stop_id": [r.get("to_stop_id") for r in rows],
        "csv_row_number": [r.get("csv_row_number", 2) for r in rows],
    }
    if include_transfer_type:
        data["transfer_type"] = [r.get("transfer_type") for r in rows]
    return pl.DataFrame(data)


def _make_stops(rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "stop_id": [r["stop_id"] for r in rows],
            "location_type": [r.get("location_type", 0) for r in rows],
        }
    )


def test_stop_to_station_transfer_no_notices() -> None:
    """STOP -> STATION is valid; no notices expected."""
    stops = _make_stops([{"stop_id": "s0", "location_type": 0}, {"stop_id": "s1", "location_type": 1}])
    transfers = _make_transfers([{"from_stop_id": "s0", "to_stop_id": "s1", "csv_row_number": 2}])
    feed = {"stops": stops, "transfers": transfers}
    notices = validate_transfers_stop_type(feed, _ctx())
    assert notices == []


def test_entrance_to_generic_node_transfer_two_notices() -> None:
    """ENTRANCE (2) and GENERIC_NODE (3) both produce notices — one per direction."""
    stops = _make_stops([{"stop_id": "s0", "location_type": 2}, {"stop_id": "s1", "location_type": 3}])
    transfers = _make_transfers([{"from_stop_id": "s0", "to_stop_id": "s1", "csv_row_number": 2}])
    feed = {"stops": stops, "transfers": transfers}
    notices = validate_transfers_stop_type(feed, _ctx())
    assert len(notices) == 2

    from_notice = notices[0]
    assert from_notice.code == "transfer_with_invalid_stop_location_type"
    assert from_notice.severity == Severity.ERROR
    assert from_notice.fields["csv_row_number"] == 2
    assert from_notice.fields["stop_id_field_name"] == "from_stop_id"
    assert from_notice.fields["stop_id"] == "s0"
    assert from_notice.fields["location_type_value"] == 2
    assert from_notice.fields["location_type_name"] == "ENTRANCE"

    to_notice = notices[1]
    assert to_notice.fields["stop_id_field_name"] == "to_stop_id"
    assert to_notice.fields["stop_id"] == "s1"
    assert to_notice.fields["location_type_value"] == 3
    assert to_notice.fields["location_type_name"] == "GENERIC_NODE"


def test_boarding_area_stop_emits_notice() -> None:
    """BOARDING_AREA (4) on to_stop_id produces one notice."""
    stops = _make_stops([{"stop_id": "s0", "location_type": 0}, {"stop_id": "s1", "location_type": 4}])
    transfers = _make_transfers([{"from_stop_id": "s0", "to_stop_id": "s1"}])
    feed = {"stops": stops, "transfers": transfers}
    notices = validate_transfers_stop_type(feed, _ctx())
    assert len(notices) == 1
    assert notices[0].fields["stop_id_field_name"] == "to_stop_id"
    assert notices[0].fields["location_type_value"] == 4
    assert notices[0].fields["location_type_name"] == "BOARDING_AREA"


def test_both_valid_stop_types_no_notices() -> None:
    """All valid pairs of {0, 1} produce no notices."""
    # STOP -> STOP
    stops = _make_stops([{"stop_id": "s0", "location_type": 0}, {"stop_id": "s1", "location_type": 0}])
    transfers = _make_transfers([{"from_stop_id": "s0", "to_stop_id": "s1"}])
    assert validate_transfers_stop_type({"stops": stops, "transfers": transfers}, _ctx()) == []

    # STATION -> STATION
    stops = _make_stops([{"stop_id": "s0", "location_type": 1}, {"stop_id": "s1", "location_type": 1}])
    transfers = _make_transfers([{"from_stop_id": "s0", "to_stop_id": "s1"}])
    assert validate_transfers_stop_type({"stops": stops, "transfers": transfers}, _ctx()) == []

    # STOP -> STATION
    stops = _make_stops([{"stop_id": "s0", "location_type": 0}, {"stop_id": "s1", "location_type": 1}])
    transfers = _make_transfers([{"from_stop_id": "s0", "to_stop_id": "s1"}])
    assert validate_transfers_stop_type({"stops": stops, "transfers": transfers}, _ctx()) == []


def test_null_from_stop_id_skips_that_direction() -> None:
    """Null from_stop_id is silently skipped; only to_stop_id produces a notice."""
    stops = _make_stops([{"stop_id": "s1", "location_type": 3}])
    transfers = _make_transfers([{"from_stop_id": None, "to_stop_id": "s1"}])
    feed = {"stops": stops, "transfers": transfers}
    notices = validate_transfers_stop_type(feed, _ctx())
    assert len(notices) == 1
    assert notices[0].fields["stop_id_field_name"] == "to_stop_id"


def test_null_to_stop_id_skips_that_direction() -> None:
    """Null to_stop_id is silently skipped; only from_stop_id produces a notice."""
    stops = _make_stops([{"stop_id": "s0", "location_type": 2}])
    transfers = _make_transfers([{"from_stop_id": "s0", "to_stop_id": None}])
    feed = {"stops": stops, "transfers": transfers}
    notices = validate_transfers_stop_type(feed, _ctx())
    assert len(notices) == 1
    assert notices[0].fields["stop_id_field_name"] == "from_stop_id"


def test_stop_id_not_in_stops_table_skips_silently() -> None:
    """Stop IDs not found in stops.txt are silently skipped."""
    stops = _make_stops([{"stop_id": "other", "location_type": 0}])
    transfers = _make_transfers([{"from_stop_id": "unknown_id", "to_stop_id": "also_unknown"}])
    feed = {"stops": stops, "transfers": transfers}
    notices = validate_transfers_stop_type(feed, _ctx())
    assert notices == []


def test_transfers_absent_returns_empty() -> None:
    """Missing transfers table returns empty list."""
    notices = validate_transfers_stop_type({}, _ctx())
    assert notices == []


def test_transfers_empty_returns_empty() -> None:
    """Zero-row transfers DataFrame returns empty list."""
    transfers = pl.DataFrame(
        {"from_stop_id": [], "to_stop_id": [], "csv_row_number": []},
        schema={"from_stop_id": pl.Utf8, "to_stop_id": pl.Utf8, "csv_row_number": pl.Int64},
    )
    notices = validate_transfers_stop_type({"transfers": transfers}, _ctx())
    assert notices == []


def test_stops_absent_returns_empty() -> None:
    """Missing stops table means all stop lookups miss; no notices produced."""
    transfers = _make_transfers([{"from_stop_id": "s0", "to_stop_id": "s1"}])
    feed = {"transfers": transfers}
    notices = validate_transfers_stop_type(feed, _ctx())
    assert notices == []


def test_required_column_absent_returns_empty() -> None:
    """Guard clause fires when required columns are missing."""
    # Missing csv_row_number
    transfers_no_csv_row = pl.DataFrame({"from_stop_id": ["s0"], "to_stop_id": ["s1"]})
    notices = validate_transfers_stop_type({"transfers": transfers_no_csv_row}, _ctx())
    assert notices == []

    # Missing from_stop_id
    transfers_no_from = pl.DataFrame({"to_stop_id": ["s1"], "csv_row_number": [2]})
    notices = validate_transfers_stop_type({"transfers": transfers_no_from}, _ctx())
    assert notices == []

    # Missing to_stop_id
    transfers_no_to = pl.DataFrame({"from_stop_id": ["s0"], "csv_row_number": [2]})
    notices = validate_transfers_stop_type({"transfers": transfers_no_to}, _ctx())
    assert notices == []


def test_multiple_rows_some_valid_some_invalid() -> None:
    """Mixed valid/invalid rows; only invalid directions produce notices."""
    stops = _make_stops([
        {"stop_id": "s0", "location_type": 0},
        {"stop_id": "s1", "location_type": 1},
        {"stop_id": "s2", "location_type": 2},
    ])
    transfers = _make_transfers([
        {"from_stop_id": "s0", "to_stop_id": "s1", "csv_row_number": 2},  # Row A: both valid
        {"from_stop_id": "s2", "to_stop_id": "s0", "csv_row_number": 3},  # Row B: FROM invalid
        {"from_stop_id": "s0", "to_stop_id": "s2", "csv_row_number": 4},  # Row C: TO invalid
    ])
    feed = {"stops": stops, "transfers": transfers}
    notices = validate_transfers_stop_type(feed, _ctx())
    assert len(notices) == 2
    assert notices[0].fields["stop_id_field_name"] == "from_stop_id"
    assert notices[0].fields["csv_row_number"] == 3
    assert notices[1].fields["stop_id_field_name"] == "to_stop_id"
    assert notices[1].fields["csv_row_number"] == 4


def test_same_invalid_stop_in_multiple_transfers() -> None:
    """Same bad stop in multiple transfers produces one notice per transfer."""
    stops = _make_stops([
        {"stop_id": "s0", "location_type": 0},
        {"stop_id": "bad", "location_type": 3},
    ])
    transfers = _make_transfers([
        {"from_stop_id": "bad", "to_stop_id": "s0", "csv_row_number": 2},
        {"from_stop_id": "bad", "to_stop_id": "s0", "csv_row_number": 3},
    ])
    feed = {"stops": stops, "transfers": transfers}
    notices = validate_transfers_stop_type(feed, _ctx())
    assert len(notices) == 2
    for notice in notices:
        assert notice.fields["stop_id_field_name"] == "from_stop_id"
        assert notice.fields["stop_id"] == "bad"


def test_in_seat_transfer_type_does_not_suppress_check() -> None:
    """Validator applies to all transfer_type values, including in-seat (4)."""
    stops = _make_stops([
        {"stop_id": "s0", "location_type": 2},
        {"stop_id": "s1", "location_type": 3},
    ])
    transfers = _make_transfers(
        [{"from_stop_id": "s0", "to_stop_id": "s1", "transfer_type": 4, "csv_row_number": 2}],
        include_transfer_type=True,
    )
    feed = {"stops": stops, "transfers": transfers}
    notices = validate_transfers_stop_type(feed, _ctx())
    assert len(notices) == 2
    assert notices[0].fields["stop_id_field_name"] == "from_stop_id"
    assert notices[1].fields["stop_id_field_name"] == "to_stop_id"


def test_csv_row_number_carried_into_notice() -> None:
    """csv_row_number from the transfer row is reflected in the notice."""
    stops = _make_stops([{"stop_id": "s0", "location_type": 4}])
    transfers = _make_transfers([{"from_stop_id": "s0", "to_stop_id": None, "csv_row_number": 42}])
    feed = {"stops": stops, "transfers": transfers}
    notices = validate_transfers_stop_type(feed, _ctx())
    assert len(notices) == 1
    assert notices[0].fields["csv_row_number"] == 42


def test_notice_ordering_from_before_to() -> None:
    """FROM direction notice appears before TO direction notice within a row."""
    stops = _make_stops([
        {"stop_id": "s0", "location_type": 2},
        {"stop_id": "s1", "location_type": 3},
    ])
    transfers = _make_transfers([{"from_stop_id": "s0", "to_stop_id": "s1", "csv_row_number": 2}])
    feed = {"stops": stops, "transfers": transfers}
    notices = validate_transfers_stop_type(feed, _ctx())
    assert len(notices) == 2
    assert notices[0].fields["stop_id_field_name"] == "from_stop_id"
    assert notices[1].fields["stop_id_field_name"] == "to_stop_id"
