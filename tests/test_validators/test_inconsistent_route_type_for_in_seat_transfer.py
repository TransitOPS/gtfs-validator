"""Tests for inconsistent_route_type_for_in_seat_transfer validator."""

from datetime import date

import polars as pl
import pytest

from gtfs_validator.context import ValidationContext
from gtfs_validator.notices import Severity
from gtfs_validator.validators.inconsistent_route_type_for_in_seat_transfer import (
    validate_inconsistent_route_type_for_in_seat_transfer,
)

CTX = ValidationContext(country_code="US", date_for_validation=date(2026, 3, 8))


def make_routes(rows: list[tuple[str, str]]) -> pl.DataFrame:
    """Each row is (route_id, route_type)."""
    return pl.DataFrame(
        {"route_id": [r[0] for r in rows], "route_type": [r[1] for r in rows]}
    )


def make_transfers(rows: list[tuple[str, str, str]], row_start: int = 2) -> pl.DataFrame:
    """Each row is (from_route_id, to_route_id, transfer_type).

    csvRowNumber starts at row_start (default 2, since row 1 is the header).
    """
    return pl.DataFrame(
        {
            "csvRowNumber": list(range(row_start, row_start + len(rows))),
            "from_route_id": [r[0] for r in rows],
            "to_route_id": [r[1] for r in rows],
            "transfer_type": [r[2] for r in rows],
        }
    )


def test_same_route_type_no_notice():
    feed = {
        "routes": make_routes([("r0", "3"), ("r1", "3")]),
        "transfers": make_transfers([("r0", "r1", "4")]),
    }
    assert validate_inconsistent_route_type_for_in_seat_transfer(feed, CTX) == []


def test_other_transfer_type_no_notice():
    feed = {
        "routes": make_routes([("r0", "3"), ("r1", "2")]),
        "transfers": make_transfers([("r0", "r1", "0")]),
    }
    assert validate_inconsistent_route_type_for_in_seat_transfer(feed, CTX) == []


def test_different_route_type_generates_notice():
    feed = {
        "routes": make_routes([("r0", "3"), ("r1", "2")]),
        "transfers": make_transfers([("r0", "r1", "4")]),
    }
    notices = validate_inconsistent_route_type_for_in_seat_transfer(feed, CTX)
    assert len(notices) == 1
    n = notices[0]
    assert n.code == "inconsistent_route_type_for_in_seat_transfer"
    assert n.severity == Severity.WARNING
    assert n.fields["from_route_id"] == "r0"
    assert n.fields["to_route_id"] == "r1"
    assert n.fields["from_route_type"] == 3
    assert n.fields["to_route_type"] == 2


def test_transfers_missing_no_notices():
    feed = {"routes": make_routes([("r0", "3")])}
    assert validate_inconsistent_route_type_for_in_seat_transfer(feed, CTX) == []


def test_routes_missing_no_notices():
    feed = {"transfers": make_transfers([("r0", "r1", "4")])}
    assert validate_inconsistent_route_type_for_in_seat_transfer(feed, CTX) == []


def test_from_route_id_not_in_routes_no_notice():
    feed = {
        "routes": make_routes([("r1", "2")]),
        "transfers": make_transfers([("r_missing", "r1", "4")]),
    }
    assert validate_inconsistent_route_type_for_in_seat_transfer(feed, CTX) == []


def test_to_route_id_not_in_routes_no_notice():
    feed = {
        "routes": make_routes([("r0", "3")]),
        "transfers": make_transfers([("r0", "r_missing", "4")]),
    }
    assert validate_inconsistent_route_type_for_in_seat_transfer(feed, CTX) == []


def test_empty_transfers_no_notices():
    feed = {
        "routes": make_routes([("r0", "3"), ("r1", "2")]),
        "transfers": pl.DataFrame(
            schema={
                "csvRowNumber": pl.Int64,
                "from_route_id": pl.Utf8,
                "to_route_id": pl.Utf8,
                "transfer_type": pl.Utf8,
            }
        ),
    }
    assert validate_inconsistent_route_type_for_in_seat_transfer(feed, CTX) == []


def test_empty_routes_no_notices():
    feed = {
        "routes": pl.DataFrame(schema={"route_id": pl.Utf8, "route_type": pl.Utf8}),
        "transfers": make_transfers([("r0", "r1", "4")]),
    }
    assert validate_inconsistent_route_type_for_in_seat_transfer(feed, CTX) == []


def test_multiple_transfers_mixed():
    feed = {
        "routes": make_routes([("r0", "3"), ("r1", "2"), ("r2", "3")]),
        "transfers": make_transfers([("r0", "r2", "4"), ("r0", "r1", "4")]),
    }
    notices = validate_inconsistent_route_type_for_in_seat_transfer(feed, CTX)
    assert len(notices) == 1
    assert notices[0].fields["from_route_id"] == "r0"
    assert notices[0].fields["to_route_id"] == "r1"


def test_transfer_type_5_no_notice():
    feed = {
        "routes": make_routes([("r0", "3"), ("r1", "2")]),
        "transfers": make_transfers([("r0", "r1", "5")]),
    }
    assert validate_inconsistent_route_type_for_in_seat_transfer(feed, CTX) == []


def test_from_route_id_column_absent_no_notices():
    feed = {
        "routes": make_routes([("r0", "3"), ("r1", "2")]),
        "transfers": pl.DataFrame({
            "csvRowNumber": [2],
            "to_route_id": ["r1"],
            "transfer_type": ["4"],
        }),
    }
    assert validate_inconsistent_route_type_for_in_seat_transfer(feed, CTX) == []


def test_to_route_id_column_absent_no_notices():
    feed = {
        "routes": make_routes([("r0", "3"), ("r1", "2")]),
        "transfers": pl.DataFrame({
            "csvRowNumber": [2],
            "from_route_id": ["r0"],
            "transfer_type": ["4"],
        }),
    }
    assert validate_inconsistent_route_type_for_in_seat_transfer(feed, CTX) == []


def test_transfer_type_column_absent_no_notices():
    feed = {
        "routes": make_routes([("r0", "3"), ("r1", "2")]),
        "transfers": pl.DataFrame({
            "csvRowNumber": [2],
            "from_route_id": ["r0"],
            "to_route_id": ["r1"],
        }),
    }
    assert validate_inconsistent_route_type_for_in_seat_transfer(feed, CTX) == []


def test_notice_fields_are_complete():
    feed = {
        "routes": make_routes([("r0", "3"), ("r1", "2")]),
        "transfers": make_transfers([("r0", "r1", "4")]),
    }
    notices = validate_inconsistent_route_type_for_in_seat_transfer(feed, CTX)
    assert len(notices) == 1
    fields = notices[0].fields
    assert set(fields.keys()) == {
        "csv_row_number", "from_route_id", "to_route_id",
        "from_route_type", "to_route_type",
    }
    assert isinstance(fields["csv_row_number"], int)
    assert isinstance(fields["from_route_type"], int)
    assert isinstance(fields["to_route_type"], int)


def test_multiple_mismatched_transfers_produce_multiple_notices():
    feed = {
        "routes": make_routes([("r0", "3"), ("r1", "2"), ("r2", "0")]),
        "transfers": make_transfers([("r0", "r1", "4"), ("r0", "r2", "4")]),
    }
    notices = validate_inconsistent_route_type_for_in_seat_transfer(feed, CTX)
    assert len(notices) == 2
