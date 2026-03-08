"""Tests for validate_parent_station."""

from datetime import date

import polars as pl
import pytest

from gtfs_validator.context import ValidationContext
from gtfs_validator.notices import Severity
from gtfs_validator.validators.parent_station import validate_parent_station

CTX = ValidationContext(country_code="US", date_for_validation=date(2026, 3, 8))


def make_stops(rows: list[dict]) -> pl.DataFrame:
    """Build a stops DataFrame from row dicts."""
    schema = {
        "csv_row_number": pl.Int64,
        "stop_id": pl.Utf8,
        "stop_name": pl.Utf8,
        "location_type": pl.Int64,
        "parent_station": pl.Utf8,
    }
    data: dict[str, list] = {col: [] for col in schema}
    for r in rows:
        for col in schema:
            data[col].append(r.get(col))
    return pl.DataFrame(data, schema=schema)


def test_stop_with_station_parent_no_notice() -> None:
    """STOP child with STATION parent — no notices."""
    stops = make_stops([
        {"csv_row_number": 1, "stop_id": "child", "location_type": 0, "parent_station": "parent"},
        {"csv_row_number": 2, "stop_id": "parent", "stop_name": "Parent location", "location_type": 1},
    ])
    notices = validate_parent_station({"stops": stops}, CTX)
    assert notices == []


def test_stop_with_entrance_parent_yields_wrong_parent_location_type() -> None:
    """STOP child with ENTRANCE parent — wrong_parent_location_type notice."""
    stops = make_stops([
        {"csv_row_number": 1, "stop_id": "child", "stop_name": "Child location", "location_type": 0, "parent_station": "parent"},
        {"csv_row_number": 2, "stop_id": "parent", "stop_name": "Parent location", "location_type": 2},
    ])
    notices = validate_parent_station({"stops": stops}, CTX)
    assert len(notices) == 1
    n = notices[0]
    assert n.code == "wrong_parent_location_type"
    assert n.severity == Severity.ERROR
    assert n.fields["csvRowNumber"] == 1
    assert n.fields["stopId"] == "child"
    assert n.fields["stopName"] == "Child location"
    assert n.fields["locationType"] == 0
    assert n.fields["parentCsvRowNumber"] == 2
    assert n.fields["parentStation"] == "parent"
    assert n.fields["parentStopName"] == "Parent location"
    assert n.fields["parentLocationType"] == 2
    assert n.fields["expectedLocationType"] == 1


def test_entrance_with_station_parent_yields_unused_station() -> None:
    """ENTRANCE child with STATION parent — valid type but station is unused by STOP children."""
    stops = make_stops([
        {"csv_row_number": 1, "stop_id": "child", "location_type": 2, "parent_station": "parent"},
        {"csv_row_number": 2, "stop_id": "parent", "stop_name": "Parent location", "location_type": 1},
    ])
    notices = validate_parent_station({"stops": stops}, CTX)
    assert len(notices) == 1
    n = notices[0]
    assert n.code == "unused_station"
    assert n.severity == Severity.INFO
    assert n.fields["csvRowNumber"] == 2
    assert n.fields["stopId"] == "parent"
    assert n.fields["stopName"] == "Parent location"


def test_entrance_with_stop_parent_yields_wrong_parent_location_type() -> None:
    """ENTRANCE child with STOP parent — wrong parent type."""
    stops = make_stops([
        {"csv_row_number": 1, "stop_id": "child", "location_type": 2, "parent_station": "parent"},
        {"csv_row_number": 2, "stop_id": "parent", "location_type": 0},
    ])
    notices = validate_parent_station({"stops": stops}, CTX)
    wrong = [n for n in notices if n.code == "wrong_parent_location_type"]
    assert len(wrong) == 1
    assert wrong[0].fields["parentLocationType"] == 0
    assert wrong[0].fields["expectedLocationType"] == 1


def test_generic_node_with_station_parent_yields_unused_station() -> None:
    """GENERIC_NODE child with STATION parent — valid type, station unused."""
    stops = make_stops([
        {"csv_row_number": 1, "stop_id": "child", "location_type": 3, "parent_station": "parent"},
        {"csv_row_number": 2, "stop_id": "parent", "stop_name": "Parent location", "location_type": 1},
    ])
    notices = validate_parent_station({"stops": stops}, CTX)
    assert len(notices) == 1
    assert notices[0].code == "unused_station"
    wrong = [n for n in notices if n.code == "wrong_parent_location_type"]
    assert len(wrong) == 0


def test_generic_node_with_stop_parent_yields_wrong_parent_location_type() -> None:
    """GENERIC_NODE child with STOP parent — wrong parent type."""
    stops = make_stops([
        {"csv_row_number": 1, "stop_id": "child", "location_type": 3, "parent_station": "parent"},
        {"csv_row_number": 2, "stop_id": "parent", "location_type": 0},
    ])
    notices = validate_parent_station({"stops": stops}, CTX)
    wrong = [n for n in notices if n.code == "wrong_parent_location_type"]
    assert len(wrong) == 1
    assert wrong[0].fields["locationType"] == 3
    assert wrong[0].fields["parentLocationType"] == 0
    assert wrong[0].fields["expectedLocationType"] == 1


def test_boarding_area_with_stop_parent_no_notice() -> None:
    """BOARDING_AREA child with STOP parent — valid, no notices."""
    stops = make_stops([
        {"csv_row_number": 1, "stop_id": "child", "location_type": 4, "parent_station": "parent"},
        {"csv_row_number": 2, "stop_id": "parent", "location_type": 0},
    ])
    notices = validate_parent_station({"stops": stops}, CTX)
    assert notices == []


def test_boarding_area_with_station_parent_yields_both_notices() -> None:
    """BOARDING_AREA child with STATION parent — both wrong_parent_location_type and unused_station."""
    stops = make_stops([
        {"csv_row_number": 1, "stop_id": "child", "location_type": 4, "parent_station": "parent"},
        {"csv_row_number": 2, "stop_id": "parent", "stop_name": "Parent location", "location_type": 1},
    ])
    notices = validate_parent_station({"stops": stops}, CTX)
    assert len(notices) == 2
    wrong = [n for n in notices if n.code == "wrong_parent_location_type"]
    unused = [n for n in notices if n.code == "unused_station"]
    assert len(wrong) == 1
    assert len(unused) == 1
    assert wrong[0].fields["locationType"] == 4
    assert wrong[0].fields["parentLocationType"] == 1
    assert wrong[0].fields["expectedLocationType"] == 0
    assert unused[0].fields["stopId"] == "parent"
    assert unused[0].fields["csvRowNumber"] == 2


def test_no_parent_station_no_notice() -> None:
    """Rows with location_type in {0, 2, 3, 4} and no parent_station — no notices."""
    stops = make_stops([
        {"csv_row_number": 1, "stop_id": "s0", "location_type": 0},
        {"csv_row_number": 2, "stop_id": "s2", "location_type": 2},
        {"csv_row_number": 3, "stop_id": "s3", "location_type": 3},
        {"csv_row_number": 4, "stop_id": "s4", "location_type": 4},
    ])
    notices = validate_parent_station({"stops": stops}, CTX)
    assert notices == []


def test_unused_station_one_used_one_unused() -> None:
    """One used station, one unused — only the unused one gets a notice."""
    stops = make_stops([
        {"csv_row_number": 1, "stop_id": "child", "location_type": 0, "parent_station": "used_station"},
        {"csv_row_number": 2, "stop_id": "unused_station", "stop_name": "Unused", "location_type": 1},
        {"csv_row_number": 3, "stop_id": "used_station", "stop_name": "Used", "location_type": 1},
    ])
    notices = validate_parent_station({"stops": stops}, CTX)
    unused = [n for n in notices if n.code == "unused_station"]
    assert len(unused) == 1
    assert unused[0].fields["stopId"] == "unused_station"
    assert unused[0].fields["csvRowNumber"] == 2


def test_foreign_key_violation_handled_gracefully() -> None:
    """STOP with parent_station that doesn't exist — inner join drops it, no notices."""
    stops = make_stops([
        {"csv_row_number": 1, "stop_id": "child", "location_type": 0, "parent_station": "parent"},
    ])
    notices = validate_parent_station({"stops": stops}, CTX)
    assert notices == []


def test_stops_absent_no_notices() -> None:
    """No stops key in feed — returns empty list."""
    notices = validate_parent_station({}, CTX)
    assert notices == []


def test_stops_empty_no_notices() -> None:
    """Empty stops DataFrame — returns empty list."""
    notices = validate_parent_station({"stops": make_stops([])}, CTX)
    assert notices == []


def test_parent_station_column_absent() -> None:
    """Stops DataFrame without parent_station column — unused_station notice for a station."""
    schema = {
        "csv_row_number": pl.Int64,
        "stop_id": pl.Utf8,
        "stop_name": pl.Utf8,
        "location_type": pl.Int64,
    }
    stops = pl.DataFrame(
        {"csv_row_number": [1], "stop_id": ["sta"], "stop_name": ["Station"], "location_type": [1]},
        schema=schema,
    )
    notices = validate_parent_station({"stops": stops}, CTX)
    assert len(notices) == 1
    assert notices[0].code == "unused_station"
    assert notices[0].fields["stopId"] == "sta"


def test_stop_name_null_passed_through() -> None:
    """Null stop_name is passed through as None, not converted to empty string."""
    stops = make_stops([
        {"csv_row_number": 1, "stop_id": "child", "stop_name": None, "location_type": 0, "parent_station": "parent"},
        {"csv_row_number": 2, "stop_id": "parent", "stop_name": None, "location_type": 2},
    ])
    notices = validate_parent_station({"stops": stops}, CTX)
    assert len(notices) == 1
    n = notices[0]
    assert n.code == "wrong_parent_location_type"
    assert n.fields["stopName"] is None
    assert n.fields["parentStopName"] is None


def test_multiple_stops_same_parent_station_only_one_used_station_notice_suppressed() -> None:
    """Two STOP rows with same parent_station — station is used, no unused_station notice."""
    stops = make_stops([
        {"csv_row_number": 1, "stop_id": "child1", "location_type": 0, "parent_station": "s"},
        {"csv_row_number": 2, "stop_id": "child2", "location_type": 0, "parent_station": "s"},
        {"csv_row_number": 3, "stop_id": "s", "stop_name": "Station", "location_type": 1},
    ])
    notices = validate_parent_station({"stops": stops}, CTX)
    assert notices == []


def test_station_with_only_entrance_child_still_unused() -> None:
    """Station with only an ENTRANCE child — station is still unused (ENTRANCE != STOP)."""
    stops = make_stops([
        {"csv_row_number": 1, "stop_id": "ent", "location_type": 2, "parent_station": "sta"},
        {"csv_row_number": 2, "stop_id": "sta", "stop_name": "Station", "location_type": 1},
    ])
    notices = validate_parent_station({"stops": stops}, CTX)
    unused = [n for n in notices if n.code == "unused_station"]
    assert len(unused) == 1
    assert unused[0].fields["stopId"] == "sta"
    wrong = [n for n in notices if n.code == "wrong_parent_location_type"]
    assert len(wrong) == 0


def test_notices_ordered_by_csv_row_number() -> None:
    """Notices are emitted in ascending csvRowNumber order regardless of feed order."""
    # Provide rows in descending csvRowNumber order; all have wrong parent type
    stops = make_stops([
        {"csv_row_number": 3, "stop_id": "child3", "location_type": 0, "parent_station": "sta"},
        {"csv_row_number": 2, "stop_id": "child2", "location_type": 0, "parent_station": "sta"},
        {"csv_row_number": 1, "stop_id": "child1", "location_type": 0, "parent_station": "sta"},
        {"csv_row_number": 4, "stop_id": "sta", "stop_name": "Station", "location_type": 2},
    ])
    notices = validate_parent_station({"stops": stops}, CTX)
    wrong = [n for n in notices if n.code == "wrong_parent_location_type"]
    assert len(wrong) == 3
    row_numbers = [n.fields["csvRowNumber"] for n in wrong]
    assert row_numbers == [1, 2, 3]
