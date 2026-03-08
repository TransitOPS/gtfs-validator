"""Tests for InconsistentRouteTypeForBlockIdValidator."""

from datetime import date

import polars as pl
import pytest

from gtfs_validator.context import ValidationContext
from gtfs_validator.notices import Severity
from gtfs_validator.validators.inconsistent_route_type_for_block_id import (
    validate_inconsistent_route_type_for_block_id,
)

CTX = ValidationContext(country_code="US", date_for_validation=date(2026, 3, 8))


def make_routes(rows: list[tuple[str, str]]) -> pl.DataFrame:
    """Each row is (route_id, route_type)."""
    return pl.DataFrame(
        {"route_id": [r[0] for r in rows], "route_type": [r[1] for r in rows]}
    )


def make_trips(rows: list[tuple[str, str, str | None]]) -> pl.DataFrame:
    """Each row is (trip_id, route_id, block_id)."""
    return pl.DataFrame(
        {
            "trip_id": [r[0] for r in rows],
            "route_id": [r[1] for r in rows],
            "block_id": [r[2] for r in rows],
        }
    )


def test_same_route_type_for_block_id_no_notice():
    feed = {
        "routes": make_routes([("r0", "3"), ("r1", "3")]),
        "trips": make_trips([("t0", "r0", "b0"), ("t1", "r1", "b0")]),
    }
    notices = validate_inconsistent_route_type_for_block_id(feed, CTX)
    assert len(notices) == 0


def test_different_route_type_for_block_id_generates_notice():
    feed = {
        "routes": make_routes([("r0", "3"), ("r1", "2")]),
        "trips": make_trips([("t0", "r0", "b0"), ("t1", "r1", "b0")]),
    }
    notices = validate_inconsistent_route_type_for_block_id(feed, CTX)
    assert len(notices) == 1
    n = notices[0]
    assert n.code == "inconsistent_route_type_for_block_id"
    assert n.severity == Severity.WARNING
    assert n.fields["block_id"] == "b0"
    assert "r0" in n.fields["route_ids"]
    assert "r1" in n.fields["route_ids"]
    assert "2" in n.fields["route_types"]
    assert "3" in n.fields["route_types"]


def test_null_block_id_skipped():
    feed = {
        "routes": make_routes([("r0", "3"), ("r1", "2")]),
        "trips": make_trips([("t0", "r0", None), ("t1", "r1", None)]),
    }
    notices = validate_inconsistent_route_type_for_block_id(feed, CTX)
    assert len(notices) == 0


def test_empty_block_id_skipped():
    feed = {
        "routes": make_routes([("r0", "3"), ("r1", "2")]),
        "trips": make_trips([("t0", "r0", ""), ("t1", "r1", "")]),
    }
    notices = validate_inconsistent_route_type_for_block_id(feed, CTX)
    assert len(notices) == 0


def test_multiple_blocks_only_inconsistent_produce_notices():
    feed = {
        "routes": make_routes([("r0", "3"), ("r1", "2"), ("r2", "3")]),
        "trips": make_trips([
            ("t0", "r0", "b0"),
            ("t1", "r1", "b0"),
            ("t2", "r0", "b1"),
            ("t3", "r2", "b1"),
        ]),
    }
    notices = validate_inconsistent_route_type_for_block_id(feed, CTX)
    assert len(notices) == 1
    assert notices[0].fields["block_id"] == "b0"


def test_trip_with_unmatched_route_id_excluded():
    feed = {
        "routes": make_routes([("r0", "3")]),
        "trips": make_trips([("t0", "r0", "b0"), ("t1", "r_missing", "b0")]),
    }
    notices = validate_inconsistent_route_type_for_block_id(feed, CTX)
    assert len(notices) == 0


def test_empty_trips_no_notices():
    feed = {
        "routes": make_routes([("r0", "3")]),
        "trips": pl.DataFrame(
            {"trip_id": [], "route_id": [], "block_id": []},
            schema={"trip_id": pl.Utf8, "route_id": pl.Utf8, "block_id": pl.Utf8},
        ),
    }
    notices = validate_inconsistent_route_type_for_block_id(feed, CTX)
    assert len(notices) == 0


def test_empty_routes_no_notices():
    feed = {
        "routes": pl.DataFrame(
            {"route_id": [], "route_type": []},
            schema={"route_id": pl.Utf8, "route_type": pl.Utf8},
        ),
        "trips": make_trips([("t0", "r0", "b0")]),
    }
    notices = validate_inconsistent_route_type_for_block_id(feed, CTX)
    assert len(notices) == 0


def test_single_trip_in_block_no_notice():
    feed = {
        "routes": make_routes([("r0", "3")]),
        "trips": make_trips([("t0", "r0", "b0")]),
    }
    notices = validate_inconsistent_route_type_for_block_id(feed, CTX)
    assert len(notices) == 0


def test_block_id_column_absent_no_notices():
    feed = {
        "routes": make_routes([("r0", "3"), ("r1", "2")]),
        "trips": pl.DataFrame({"trip_id": ["t0", "t1"], "route_id": ["r0", "r1"]}),
    }
    notices = validate_inconsistent_route_type_for_block_id(feed, CTX)
    assert len(notices) == 0


def test_multiple_trips_same_route_id_in_block_deduplicated():
    feed = {
        "routes": make_routes([("r0", "3"), ("r1", "2")]),
        "trips": make_trips([
            ("t0", "r0", "b0"),
            ("t1", "r0", "b0"),
            ("t2", "r1", "b0"),
        ]),
    }
    notices = validate_inconsistent_route_type_for_block_id(feed, CTX)
    assert len(notices) == 1
    assert "r0" in notices[0].fields["route_ids"]
    assert "r1" in notices[0].fields["route_ids"]
    assert "2" in notices[0].fields["route_types"]
    assert "3" in notices[0].fields["route_types"]


def test_notice_fields_are_complete():
    feed = {
        "routes": make_routes([("r0", "3"), ("r1", "2")]),
        "trips": make_trips([("t0", "r0", "b0"), ("t1", "r1", "b0")]),
    }
    notices = validate_inconsistent_route_type_for_block_id(feed, CTX)
    assert len(notices) == 1
    fields = notices[0].fields
    assert set(fields.keys()) == {"block_id", "route_ids", "route_types"}
    assert isinstance(fields["block_id"], str)
    # Sorted comma-separated values
    assert fields["route_ids"] == "r0, r1"
    assert fields["route_types"] == "2, 3"


def test_three_distinct_route_types_in_block():
    feed = {
        "routes": make_routes([("r0", "3"), ("r1", "2"), ("r2", "0")]),
        "trips": make_trips([
            ("t0", "r0", "b0"),
            ("t1", "r1", "b0"),
            ("t2", "r2", "b0"),
        ]),
    }
    notices = validate_inconsistent_route_type_for_block_id(feed, CTX)
    assert len(notices) == 1
    assert "0" in notices[0].fields["route_types"]
    assert "2" in notices[0].fields["route_types"]
    assert "3" in notices[0].fields["route_types"]
