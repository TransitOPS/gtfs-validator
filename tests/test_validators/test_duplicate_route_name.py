"""Tests for validate_duplicate_route_name."""

from datetime import date

import polars as pl

from gtfs_validator.context import ValidationContext
from gtfs_validator.notices import Severity
from gtfs_validator.validators.duplicate_route_name import validate_duplicate_route_name

CTX = ValidationContext(country_code="US", date_for_validation=date(2024, 1, 1))

LIGHT_RAIL = 0
BUS = 3


def make_routes(rows: list[dict]) -> pl.DataFrame:
    """Create a routes DataFrame with the columns used by this validator."""
    if not rows:
        return pl.DataFrame(
            schema={
                "route_id": pl.Utf8,
                "agency_id": pl.Utf8,
                "route_short_name": pl.Utf8,
                "route_long_name": pl.Utf8,
                "route_type": pl.Int64,
            }
        )
    return pl.DataFrame(rows).cast(
        {
            "route_id": pl.Utf8,
            "agency_id": pl.Utf8,
            "route_short_name": pl.Utf8,
            "route_long_name": pl.Utf8,
            "route_type": pl.Int64,
        }
    )


def test_same_names_type_agency_duplicate():
    feed = {
        "routes": make_routes(
            [
                {"route_id": "route1", "agency_id": "agency1", "route_short_name": "L1", "route_long_name": "Dulwich Hill", "route_type": LIGHT_RAIL},
                {"route_id": "route2", "agency_id": "agency1", "route_short_name": "L1", "route_long_name": "Dulwich Hill", "route_type": LIGHT_RAIL},
            ]
        )
    }
    notices = validate_duplicate_route_name(feed, CTX)
    assert len(notices) == 1
    n = notices[0]
    assert n.fields["csv_row_number_1"] == 2
    assert n.fields["route_id_1"] == "route1"
    assert n.fields["csv_row_number_2"] == 3
    assert n.fields["route_id_2"] == "route2"
    assert n.fields["route_short_name"] == "L1"
    assert n.fields["route_long_name"] == "Dulwich Hill"
    assert n.fields["route_type_value"] == LIGHT_RAIL
    assert n.fields["agency_id"] == "agency1"


def test_three_way_duplicate():
    feed = {
        "routes": make_routes(
            [
                {"route_id": "r1", "agency_id": "agency1", "route_short_name": "L1", "route_long_name": "Dulwich Hill", "route_type": LIGHT_RAIL},
                {"route_id": "r2", "agency_id": "agency1", "route_short_name": "L1", "route_long_name": "Dulwich Hill", "route_type": LIGHT_RAIL},
                {"route_id": "r3", "agency_id": "agency1", "route_short_name": "L1", "route_long_name": "Dulwich Hill", "route_type": LIGHT_RAIL},
            ]
        )
    }
    notices = validate_duplicate_route_name(feed, CTX)
    assert len(notices) == 2
    assert notices[0].fields["csv_row_number_1"] == 2
    assert notices[0].fields["route_id_1"] == "r1"
    assert notices[0].fields["csv_row_number_2"] == 3
    assert notices[0].fields["route_id_2"] == "r2"
    assert notices[1].fields["csv_row_number_1"] == 2
    assert notices[1].fields["route_id_1"] == "r1"
    assert notices[1].fields["csv_row_number_2"] == 4
    assert notices[1].fields["route_id_2"] == "r3"


def test_no_short_names_duplicate():
    feed = {
        "routes": make_routes(
            [
                {"route_id": "r1", "agency_id": "a1", "route_short_name": None, "route_long_name": "Dulwich Hill", "route_type": LIGHT_RAIL},
                {"route_id": "r2", "agency_id": "a1", "route_short_name": None, "route_long_name": "Dulwich Hill", "route_type": LIGHT_RAIL},
            ]
        )
    }
    notices = validate_duplicate_route_name(feed, CTX)
    assert len(notices) == 1
    assert notices[0].code == "duplicate_route_name"


def test_no_long_names_duplicate():
    feed = {
        "routes": make_routes(
            [
                {"route_id": "r1", "agency_id": "a1", "route_short_name": "L1", "route_long_name": None, "route_type": LIGHT_RAIL},
                {"route_id": "r2", "agency_id": "a1", "route_short_name": "L1", "route_long_name": None, "route_type": LIGHT_RAIL},
            ]
        )
    }
    notices = validate_duplicate_route_name(feed, CTX)
    assert len(notices) == 1
    assert notices[0].code == "duplicate_route_name"


def test_different_short_names_no_notice():
    feed = {
        "routes": make_routes(
            [
                {"route_id": "r1", "agency_id": "a1", "route_short_name": "L1", "route_long_name": "North Line", "route_type": LIGHT_RAIL},
                {"route_id": "r2", "agency_id": "a1", "route_short_name": "L2", "route_long_name": "North Line", "route_type": LIGHT_RAIL},
            ]
        )
    }
    notices = validate_duplicate_route_name(feed, CTX)
    assert len(notices) == 0


def test_different_long_names_no_notice():
    feed = {
        "routes": make_routes(
            [
                {"route_id": "r1", "agency_id": "a1", "route_short_name": "L1", "route_long_name": "North Line", "route_type": LIGHT_RAIL},
                {"route_id": "r2", "agency_id": "a1", "route_short_name": "L1", "route_long_name": "South Line", "route_type": LIGHT_RAIL},
            ]
        )
    }
    notices = validate_duplicate_route_name(feed, CTX)
    assert len(notices) == 0


def test_different_agencies_no_notice():
    feed = {
        "routes": make_routes(
            [
                {"route_id": "r1", "agency_id": "agency1", "route_short_name": "L1", "route_long_name": "North Line", "route_type": LIGHT_RAIL},
                {"route_id": "r2", "agency_id": "agency2", "route_short_name": "L1", "route_long_name": "North Line", "route_type": LIGHT_RAIL},
            ]
        )
    }
    notices = validate_duplicate_route_name(feed, CTX)
    assert len(notices) == 0


def test_different_types_no_notice():
    feed = {
        "routes": make_routes(
            [
                {"route_id": "r1", "agency_id": "a1", "route_short_name": "L1", "route_long_name": "North Line", "route_type": LIGHT_RAIL},
                {"route_id": "r2", "agency_id": "a1", "route_short_name": "L1", "route_long_name": "North Line", "route_type": BUS},
            ]
        )
    }
    notices = validate_duplicate_route_name(feed, CTX)
    assert len(notices) == 0


def test_empty_table_no_notice():
    feed = {"routes": make_routes([])}
    notices = validate_duplicate_route_name(feed, CTX)
    assert len(notices) == 0


def test_single_route_no_notice():
    feed = {
        "routes": make_routes(
            [{"route_id": "r1", "agency_id": "a1", "route_short_name": "L1", "route_long_name": "North Line", "route_type": LIGHT_RAIL}]
        )
    }
    notices = validate_duplicate_route_name(feed, CTX)
    assert len(notices) == 0


def test_both_names_null_duplicate():
    feed = {
        "routes": make_routes(
            [
                {"route_id": "r1", "agency_id": "a1", "route_short_name": None, "route_long_name": None, "route_type": BUS},
                {"route_id": "r2", "agency_id": "a1", "route_short_name": None, "route_long_name": None, "route_type": BUS},
            ]
        )
    }
    notices = validate_duplicate_route_name(feed, CTX)
    assert len(notices) == 1
    assert notices[0].code == "duplicate_route_name"


def test_same_names_different_agency_and_type_no_notice():
    feed = {
        "routes": make_routes(
            [
                {"route_id": "r1", "agency_id": "a1", "route_short_name": "X", "route_long_name": "Y", "route_type": BUS},
                {"route_id": "r2", "agency_id": "a2", "route_short_name": "X", "route_long_name": "Y", "route_type": LIGHT_RAIL},
            ]
        )
    }
    notices = validate_duplicate_route_name(feed, CTX)
    assert len(notices) == 0


def test_missing_table_no_notice():
    feed: dict[str, pl.DataFrame] = {}
    notices = validate_duplicate_route_name(feed, CTX)
    assert len(notices) == 0


def test_null_agency_id_match():
    feed = {
        "routes": make_routes(
            [
                {"route_id": "r1", "agency_id": None, "route_short_name": "L1", "route_long_name": "North Line", "route_type": BUS},
                {"route_id": "r2", "agency_id": None, "route_short_name": "L1", "route_long_name": "North Line", "route_type": BUS},
            ]
        )
    }
    notices = validate_duplicate_route_name(feed, CTX)
    assert len(notices) == 1
    assert notices[0].fields["agency_id"] == ""


def test_notice_severity_is_warning():
    feed = {
        "routes": make_routes(
            [
                {"route_id": "r1", "agency_id": "a1", "route_short_name": "L1", "route_long_name": "X", "route_type": BUS},
                {"route_id": "r2", "agency_id": "a1", "route_short_name": "L1", "route_long_name": "X", "route_type": BUS},
            ]
        )
    }
    notices = validate_duplicate_route_name(feed, CTX)
    assert all(n.severity == Severity.WARNING for n in notices)


def test_notice_code():
    feed = {
        "routes": make_routes(
            [
                {"route_id": "r1", "agency_id": "a1", "route_short_name": "L1", "route_long_name": "X", "route_type": BUS},
                {"route_id": "r2", "agency_id": "a1", "route_short_name": "L1", "route_long_name": "X", "route_type": BUS},
            ]
        )
    }
    notices = validate_duplicate_route_name(feed, CTX)
    assert all(n.code == "duplicate_route_name" for n in notices)


def test_multiple_duplicate_groups():
    feed = {
        "routes": make_routes(
            [
                {"route_id": "r1", "agency_id": "a1", "route_short_name": "L1", "route_long_name": "A", "route_type": LIGHT_RAIL},
                {"route_id": "r2", "agency_id": "a1", "route_short_name": "L1", "route_long_name": "A", "route_type": LIGHT_RAIL},
                {"route_id": "r3", "agency_id": "a1", "route_short_name": "B2", "route_long_name": "C", "route_type": BUS},
                {"route_id": "r4", "agency_id": "a1", "route_short_name": "B2", "route_long_name": "C", "route_type": BUS},
            ]
        )
    }
    notices = validate_duplicate_route_name(feed, CTX)
    assert len(notices) == 2
    codes = {n.fields["route_short_name"] for n in notices}
    assert codes == {"L1", "B2"}
