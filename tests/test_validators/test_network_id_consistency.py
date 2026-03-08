"""Tests for NetworkIdConsistencyValidator."""

from datetime import date

import polars as pl
import pytest

from gtfs_validator.context import ValidationContext
from gtfs_validator.notices import Severity
from gtfs_validator.validators.network_id_consistency import validate_network_id_consistency

CTX = ValidationContext(country_code="US", date_for_validation=date(2026, 3, 8))


def make_routes(with_network_id: bool) -> pl.DataFrame:
    """Build a minimal routes DataFrame, optionally including a network_id column."""
    schema: dict[str, pl.PolarsDataType] = {
        "route_id": pl.Utf8,
        "agency_id": pl.Utf8,
        "route_type": pl.Int64,
    }
    data: dict[str, list] = {
        "route_id": ["123"],
        "agency_id": ["agency1"],
        "route_type": [3],
    }
    if with_network_id:
        schema["network_id"] = pl.Utf8
        data["network_id"] = ["network1"]
    return pl.DataFrame(data, schema=schema)


def make_route_networks() -> pl.DataFrame:
    """Build a minimal route_networks DataFrame."""
    return pl.DataFrame(
        {"route_id": ["123"], "network_id": ["network1"]},
        schema={"route_id": pl.Utf8, "network_id": pl.Utf8},
    )


def make_networks() -> pl.DataFrame:
    """Build a minimal networks DataFrame."""
    return pl.DataFrame(
        {"network_id": ["network1"]},
        schema={"network_id": pl.Utf8},
    )


def test_no_conflict_when_no_competing_files() -> None:
    feed = {"routes": make_routes(with_network_id=True)}
    notices = validate_network_id_consistency(feed, CTX)
    assert notices == []


def test_conflict_with_route_networks() -> None:
    feed = {
        "routes": make_routes(with_network_id=True),
        "route_networks": make_route_networks(),
    }
    notices = validate_network_id_consistency(feed, CTX)
    assert len(notices) == 1
    n = notices[0]
    assert n.code == "route_networks_specified_in_more_than_one_file"
    assert n.severity == Severity.ERROR
    assert n.fields["file_name_a"] == "routes.txt"
    assert n.fields["file_name_b"] == "route_networks.txt"
    assert n.fields["field_name"] == "network_id"


def test_conflict_with_networks() -> None:
    feed = {
        "routes": make_routes(with_network_id=True),
        "networks": make_networks(),
    }
    notices = validate_network_id_consistency(feed, CTX)
    assert len(notices) == 1
    n = notices[0]
    assert n.code == "route_networks_specified_in_more_than_one_file"
    assert n.severity == Severity.ERROR
    assert n.fields["file_name_a"] == "routes.txt"
    assert n.fields["file_name_b"] == "networks.txt"
    assert n.fields["field_name"] == "network_id"


def test_conflict_with_both_files_yields_two_notices() -> None:
    feed = {
        "routes": make_routes(with_network_id=True),
        "route_networks": make_route_networks(),
        "networks": make_networks(),
    }
    notices = validate_network_id_consistency(feed, CTX)
    assert len(notices) == 2
    file_name_bs = {n.fields["file_name_b"] for n in notices}
    assert file_name_bs == {"route_networks.txt", "networks.txt"}
    assert all(n.code == "route_networks_specified_in_more_than_one_file" for n in notices)
    assert all(n.severity == Severity.ERROR for n in notices)
    assert all(n.fields["file_name_a"] == "routes.txt" for n in notices)
    assert all(n.fields["field_name"] == "network_id" for n in notices)


def test_no_notice_when_network_id_column_absent() -> None:
    feed = {
        "routes": make_routes(with_network_id=False),
        "route_networks": make_route_networks(),
        "networks": make_networks(),
    }
    notices = validate_network_id_consistency(feed, CTX)
    assert notices == []


def test_no_notice_when_routes_absent() -> None:
    feed: dict = {
        "route_networks": make_route_networks(),
        "networks": make_networks(),
    }
    notices = validate_network_id_consistency(feed, CTX)
    assert notices == []


def test_no_notice_when_routes_empty_but_network_id_column_present() -> None:
    empty_routes = pl.DataFrame(
        schema={"route_id": pl.Utf8, "network_id": pl.Utf8}
    )
    feed = {"routes": empty_routes}
    notices = validate_network_id_consistency(feed, CTX)
    assert notices == []


def test_network_id_all_null_still_triggers_check() -> None:
    routes_with_null_network_id = pl.DataFrame(
        {"route_id": ["123"], "network_id": [None]},
        schema={"route_id": pl.Utf8, "network_id": pl.Utf8},
    )
    feed = {
        "routes": routes_with_null_network_id,
        "route_networks": make_route_networks(),
    }
    notices = validate_network_id_consistency(feed, CTX)
    assert len(notices) == 1
    assert notices[0].fields["file_name_b"] == "route_networks.txt"


def test_empty_feed_no_notices() -> None:
    notices = validate_network_id_consistency({}, CTX)
    assert notices == []
