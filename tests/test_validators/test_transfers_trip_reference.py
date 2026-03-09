"""Tests for TransfersTripReferenceValidator."""

from __future__ import annotations

import polars as pl
import pytest

from gtfs_validator.context import ValidationContext
from gtfs_validator.notices import Severity
from gtfs_validator.validators.transfers_trip_reference import (
    validate_transfers_trip_reference,
)
from datetime import date

CTX = ValidationContext(country_code="US", date_for_validation=date(2024, 1, 1))


def make_trips(rows: list[tuple[str, str]]) -> pl.DataFrame:
    """Build a minimal trips DataFrame from (trip_id, route_id) tuples."""
    trip_ids = [r[0] for r in rows]
    route_ids = [r[1] for r in rows]
    return pl.DataFrame({"trip_id": trip_ids, "route_id": route_ids})


def make_stops(rows: list[tuple[str, int, str | None]]) -> pl.DataFrame:
    """Build a minimal stops DataFrame from (stop_id, location_type, parent_station) tuples."""
    stop_ids = [r[0] for r in rows]
    location_types = [r[1] for r in rows]
    parent_stations = [r[2] for r in rows]
    return pl.DataFrame(
        {
            "stop_id": stop_ids,
            "location_type": location_types,
            "parent_station": parent_stations,
        }
    )


def make_stop_times(rows: list[tuple[str, str]]) -> pl.DataFrame:
    """Build a minimal stop_times DataFrame from (trip_id, stop_id) tuples."""
    trip_ids = [r[0] for r in rows]
    stop_ids = [r[1] for r in rows]
    return pl.DataFrame({"trip_id": trip_ids, "stop_id": stop_ids})


def make_transfers(
    from_trip_id: str | None = None,
    to_trip_id: str | None = None,
    from_route_id: str | None = None,
    to_route_id: str | None = None,
    from_stop_id: str | None = None,
    to_stop_id: str | None = None,
    csv_row_number: int = 2,
) -> pl.DataFrame:
    """Build a single-row transfers DataFrame."""
    return pl.DataFrame(
        {
            "from_trip_id": [from_trip_id],
            "to_trip_id": [to_trip_id],
            "from_route_id": [from_route_id],
            "to_route_id": [to_route_id],
            "from_stop_id": [from_stop_id],
            "to_stop_id": [to_stop_id],
            "csv_row_number": [csv_row_number],
        }
    )


# ---------------------------------------------------------------------------
# Happy-path tests
# ---------------------------------------------------------------------------


def test_valid_direct_stop_reference() -> None:
    """Valid transfer with direct stop reference — no notices."""
    feed = {
        "trips": make_trips([("t0", "r0")]),
        "stops": make_stops([("s0", 0, None)]),
        "stop_times": make_stop_times([("t0", "s0")]),
        "transfers": make_transfers(
            from_trip_id="t0",
            from_route_id="r0",
            from_stop_id="s0",
            csv_row_number=2,
        ),
    }
    assert validate_transfers_trip_reference(feed, CTX) == []


def test_valid_station_reference() -> None:
    """Referencing a STATION is valid when a child stop appears in the trip's stop-times."""
    feed = {
        "trips": make_trips([("t1", "r1")]),
        "stops": make_stops([("s1_station", 1, None), ("s1_stop", 0, "s1_station")]),
        "stop_times": make_stop_times([("t1", "s1_stop")]),
        "transfers": make_transfers(
            to_trip_id="t1",
            to_route_id="r1",
            to_stop_id="s1_station",
            csv_row_number=2,
        ),
    }
    assert validate_transfers_trip_reference(feed, CTX) == []


# ---------------------------------------------------------------------------
# Error cases mirroring Java contracts
# ---------------------------------------------------------------------------


def test_invalid_route_from_direction() -> None:
    """FROM-side route mismatch emits transfer_with_invalid_trip_and_route."""
    feed = {
        "trips": make_trips([("t0", "r0")]),
        "stops": make_stops([("s0", 0, None)]),
        "stop_times": make_stop_times([("t0", "s0")]),
        "transfers": make_transfers(
            from_trip_id="t0",
            from_route_id="DNE",
            from_stop_id="s0",
            csv_row_number=2,
        ),
    }
    notices = validate_transfers_trip_reference(feed, CTX)
    assert len(notices) == 1
    n = notices[0]
    assert n.code == "transfer_with_invalid_trip_and_route"
    assert n.severity == Severity.ERROR
    assert n.fields["trip_field_name"] == "from_trip_id"
    assert n.fields["trip_id"] == "t0"
    assert n.fields["route_field_name"] == "from_route_id"
    assert n.fields["route_id"] == "DNE"
    assert n.fields["expected_route_id"] == "r0"
    assert n.fields["csv_row_number"] == 2


def test_invalid_stop_to_direction() -> None:
    """TO-side stop not in trip's stop-times emits transfer_with_invalid_trip_and_stop."""
    feed = {
        "trips": make_trips([("t1", "r1")]),
        "stops": make_stops([("s1", 0, None), ("s2", 0, None)]),
        "stop_times": make_stop_times([("t1", "s1")]),
        "transfers": make_transfers(
            to_trip_id="t1",
            to_route_id="r1",
            to_stop_id="s2",
            csv_row_number=2,
        ),
    }
    notices = validate_transfers_trip_reference(feed, CTX)
    assert len(notices) == 1
    n = notices[0]
    assert n.code == "transfer_with_invalid_trip_and_stop"
    assert n.severity == Severity.ERROR
    assert n.fields["trip_field_name"] == "to_trip_id"
    assert n.fields["trip_id"] == "t1"
    assert n.fields["stop_field_name"] == "to_stop_id"
    assert n.fields["stop_id"] == "s2"


def test_full_invalid_transfer_both_directions() -> None:
    """Both directions invalid: route error for FROM, stop error for TO, in that order."""
    feed = {
        "trips": make_trips([("t0", "r0"), ("t1", "r1")]),
        "stops": make_stops([("s0", 0, None), ("s1", 0, None), ("s2", 0, None)]),
        "stop_times": make_stop_times([("t0", "s0"), ("t1", "s1")]),
        "transfers": make_transfers(
            from_trip_id="t0",
            from_route_id="DNE",
            from_stop_id="s0",
            to_trip_id="t1",
            to_route_id="r1",
            to_stop_id="s2",
            csv_row_number=2,
        ),
    }
    notices = validate_transfers_trip_reference(feed, CTX)
    assert len(notices) == 2

    route_notice = notices[0]
    assert route_notice.code == "transfer_with_invalid_trip_and_route"
    assert route_notice.fields["trip_id"] == "t0"
    assert route_notice.fields["route_id"] == "DNE"
    assert route_notice.fields["expected_route_id"] == "r0"

    stop_notice = notices[1]
    assert stop_notice.code == "transfer_with_invalid_trip_and_stop"
    assert stop_notice.fields["trip_id"] == "t1"
    assert stop_notice.fields["stop_id"] == "s2"


# ---------------------------------------------------------------------------
# Guard / edge cases
# ---------------------------------------------------------------------------


def test_no_transfers_table() -> None:
    """Missing transfers key returns empty list."""
    feed: dict[str, pl.DataFrame] = {}
    assert validate_transfers_trip_reference(feed, CTX) == []


def test_empty_transfers_table() -> None:
    """Empty transfers DataFrame returns empty list."""
    feed = {
        "transfers": pl.DataFrame(
            {
                "from_trip_id": [],
                "to_trip_id": [],
                "from_route_id": [],
                "to_route_id": [],
                "from_stop_id": [],
                "to_stop_id": [],
                "csv_row_number": [],
            }
        )
    }
    assert validate_transfers_trip_reference(feed, CTX) == []


def test_trips_table_absent() -> None:
    """When trips is absent all directions are silently skipped."""
    feed = {
        "transfers": make_transfers(from_trip_id="t0", csv_row_number=2),
    }
    assert validate_transfers_trip_reference(feed, CTX) == []


def test_stop_times_table_absent() -> None:
    """When stop_times is absent the trip's stop set is empty → stop notice."""
    feed = {
        "trips": make_trips([("t0", "r0")]),
        "stops": make_stops([("s0", 0, None)]),
        "transfers": make_transfers(
            from_trip_id="t0",
            from_route_id="r0",
            from_stop_id="s0",
            csv_row_number=2,
        ),
    }
    notices = validate_transfers_trip_reference(feed, CTX)
    assert len(notices) == 1
    assert notices[0].code == "transfer_with_invalid_trip_and_stop"


def test_null_trip_id_skips_direction() -> None:
    """Direction with null trip_id is silently skipped; valid TO direction is fine."""
    feed = {
        "trips": make_trips([("t0", "r0")]),
        "stops": make_stops([("s0", 0, None)]),
        "stop_times": make_stop_times([("t0", "s0")]),
        "transfers": make_transfers(
            from_trip_id=None,
            to_trip_id="t0",
            to_route_id="r0",
            to_stop_id="s0",
            csv_row_number=2,
        ),
    }
    assert validate_transfers_trip_reference(feed, CTX) == []


def test_trip_id_not_in_trips_table() -> None:
    """Unknown trip_id is silently skipped (FK validator owns it)."""
    feed = {
        "trips": make_trips([("t0", "r0")]),
        "stops": make_stops([("s0", 0, None)]),
        "stop_times": make_stop_times([("t0", "s0")]),
        "transfers": make_transfers(from_trip_id="ghost_trip", csv_row_number=2),
    }
    assert validate_transfers_trip_reference(feed, CTX) == []


def test_stop_id_not_in_stops_table() -> None:
    """Unknown stop_id is silently skipped (FK validator owns it)."""
    feed = {
        "trips": make_trips([("t0", "r0")]),
        "stops": make_stops([("s0", 0, None)]),
        "stop_times": make_stop_times([("t0", "s0")]),
        "transfers": make_transfers(
            from_trip_id="t0",
            from_route_id="r0",
            from_stop_id="ghost_stop",
            csv_row_number=2,
        ),
    }
    assert validate_transfers_trip_reference(feed, CTX) == []


def test_station_with_no_children() -> None:
    """A STATION stop with no children produces an empty candidate set → stop notice."""
    feed = {
        "trips": make_trips([("t0", "r0")]),
        "stops": make_stops([("s0", 0, None), ("empty_station", 1, None)]),
        "stop_times": make_stop_times([("t0", "s0")]),
        "transfers": make_transfers(
            from_trip_id="t0",
            from_route_id="r0",
            from_stop_id="empty_station",
            csv_row_number=2,
        ),
    }
    notices = validate_transfers_trip_reference(feed, CTX)
    assert len(notices) == 1
    assert notices[0].code == "transfer_with_invalid_trip_and_stop"
    assert notices[0].fields["stop_id"] == "empty_station"


def test_stop_with_invalid_location_type() -> None:
    """An ENTRANCE (location_type=2) produces empty candidates → stop notice."""
    feed = {
        "trips": make_trips([("t0", "r0")]),
        "stops": make_stops([("s0", 0, None), ("entrance", 2, None)]),
        "stop_times": make_stop_times([("t0", "s0")]),
        "transfers": make_transfers(
            from_trip_id="t0",
            from_route_id="r0",
            from_stop_id="entrance",
            csv_row_number=2,
        ),
    }
    notices = validate_transfers_trip_reference(feed, CTX)
    assert len(notices) == 1
    assert notices[0].code == "transfer_with_invalid_trip_and_stop"
    assert notices[0].fields["stop_id"] == "entrance"


def test_both_route_and_stop_invalid_same_direction() -> None:
    """One direction with both route mismatch and stop mismatch → two notices, route first."""
    feed = {
        "trips": make_trips([("t0", "r0")]),
        "stops": make_stops([("s0", 0, None), ("s_wrong", 0, None)]),
        "stop_times": make_stop_times([("t0", "s0")]),
        "transfers": make_transfers(
            from_trip_id="t0",
            from_route_id="wrong_route",
            from_stop_id="s_wrong",
            csv_row_number=2,
        ),
    }
    notices = validate_transfers_trip_reference(feed, CTX)
    assert len(notices) == 2
    assert notices[0].code == "transfer_with_invalid_trip_and_route"
    assert notices[1].code == "transfer_with_invalid_trip_and_stop"


def test_multiple_transfer_rows() -> None:
    """First row valid, second row invalid → only second row produces notices."""
    valid_row = {
        "from_trip_id": ["t0"],
        "to_trip_id": [None],
        "from_route_id": ["r0"],
        "to_route_id": [None],
        "from_stop_id": ["s0"],
        "to_stop_id": [None],
        "csv_row_number": [2],
    }
    invalid_row = {
        "from_trip_id": ["t0"],
        "to_trip_id": [None],
        "from_route_id": ["DNE"],
        "to_route_id": [None],
        "from_stop_id": ["s0"],
        "to_stop_id": [None],
        "csv_row_number": [3],
    }
    combined = {
        col: valid_row[col] + invalid_row[col]  # type: ignore[operator]
        for col in valid_row
    }
    feed = {
        "trips": make_trips([("t0", "r0")]),
        "stops": make_stops([("s0", 0, None)]),
        "stop_times": make_stop_times([("t0", "s0")]),
        "transfers": pl.DataFrame(combined),
    }
    notices = validate_transfers_trip_reference(feed, CTX)
    assert len(notices) == 1
    assert notices[0].code == "transfer_with_invalid_trip_and_route"
    assert notices[0].fields["csv_row_number"] == 3


def test_route_check_absent_route_field() -> None:
    """Null route field skips route check; stop check still runs."""
    feed = {
        "trips": make_trips([("t0", "r0")]),
        "stops": make_stops([("s0", 0, None)]),
        "stop_times": make_stop_times([("t0", "s0")]),
        "transfers": make_transfers(
            from_trip_id="t0",
            from_route_id=None,
            from_stop_id="s0",
            csv_row_number=2,
        ),
    }
    # Route is null → skip. Stop is valid → no stop notice.
    assert validate_transfers_trip_reference(feed, CTX) == []


def test_stop_check_absent_stop_field() -> None:
    """Null stop field skips stop check; route check still runs."""
    feed = {
        "trips": make_trips([("t0", "r0")]),
        "stops": make_stops([("s0", 0, None)]),
        "stop_times": make_stop_times([("t0", "s0")]),
        "transfers": make_transfers(
            from_trip_id="t0",
            from_route_id="r0",
            from_stop_id=None,
            csv_row_number=2,
        ),
    }
    # Route matches → no route notice. Stop is null → skip.
    assert validate_transfers_trip_reference(feed, CTX) == []
