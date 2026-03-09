"""Tests for fare_leg_rule_network_id_foreign_key validator."""

from datetime import date

import polars as pl
import pytest

from gtfs_validator.context import ValidationContext
from gtfs_validator.notices import Severity
from gtfs_validator.validators.fare_leg_rule_network_id_foreign_key import (
    validate_fare_leg_rule_network_id_foreign_key,
)

CTX = ValidationContext(country_code="US", date_for_validation=date(2026, 3, 8))


def make_fare_leg_rules(network_ids: list[str | None]) -> pl.DataFrame:
    return pl.DataFrame({
        "network_id": network_ids,
        "csvRowNumber": list(range(2, 2 + len(network_ids))),
    })


def make_routes(network_ids: list[str]) -> pl.DataFrame:
    return pl.DataFrame({"network_id": network_ids})


def make_networks(network_ids: list[str]) -> pl.DataFrame:
    return pl.DataFrame({"network_id": network_ids})


def test_network_id_not_in_route_or_network_generates_notice():
    feed = {
        "fare_leg_rules": make_fare_leg_rules(["testNetworkId"]),
        "routes": make_routes(["otherNetworkId"]),
        "networks": make_networks(["otherNetworkId"]),
    }
    notices = validate_fare_leg_rule_network_id_foreign_key(feed, CTX)
    assert len(notices) == 1
    assert notices[0].code == "foreign_key_violation"
    assert notices[0].severity == Severity.ERROR
    assert notices[0].fields["fieldValue"] == "testNetworkId"
    assert notices[0].fields["csvRowNumber"] == 2


def test_network_id_in_route_but_not_in_network_no_notice():
    feed = {
        "fare_leg_rules": make_fare_leg_rules(["testNetworkId"]),
        "routes": make_routes(["testNetworkId"]),
        "networks": make_networks(["otherNetworkId"]),
    }
    notices = validate_fare_leg_rule_network_id_foreign_key(feed, CTX)
    assert len(notices) == 0


def test_network_id_not_in_route_but_in_network_no_notice():
    feed = {
        "fare_leg_rules": make_fare_leg_rules(["testNetworkId"]),
        "routes": make_routes(["otherNetworkId"]),
        "networks": make_networks(["testNetworkId"]),
    }
    notices = validate_fare_leg_rule_network_id_foreign_key(feed, CTX)
    assert len(notices) == 0


def test_network_id_in_both_route_and_network_no_notice():
    feed = {
        "fare_leg_rules": make_fare_leg_rules(["testNetworkId"]),
        "routes": make_routes(["testNetworkId"]),
        "networks": make_networks(["testNetworkId"]),
    }
    notices = validate_fare_leg_rule_network_id_foreign_key(feed, CTX)
    assert len(notices) == 0


def test_empty_network_id_skipped():
    feed = {
        "fare_leg_rules": make_fare_leg_rules([""]),
        "routes": pl.DataFrame({"network_id": pl.Series([], dtype=pl.Utf8)}),
        "networks": pl.DataFrame({"network_id": pl.Series([], dtype=pl.Utf8)}),
    }
    notices = validate_fare_leg_rule_network_id_foreign_key(feed, CTX)
    assert len(notices) == 0


def test_null_network_id_skipped():
    feed = {
        "fare_leg_rules": make_fare_leg_rules([None]),
        "routes": pl.DataFrame({"network_id": pl.Series([], dtype=pl.Utf8)}),
    }
    notices = validate_fare_leg_rule_network_id_foreign_key(feed, CTX)
    assert len(notices) == 0


def test_mixed_valid_and_invalid_rows():
    feed = {
        "fare_leg_rules": make_fare_leg_rules(["validId", "invalidId"]),
        "routes": make_routes(["validId"]),
    }
    notices = validate_fare_leg_rule_network_id_foreign_key(feed, CTX)
    assert len(notices) == 1
    assert notices[0].fields["csvRowNumber"] == 3
    assert notices[0].fields["fieldValue"] == "invalidId"


def test_routes_absent_networks_has_match():
    feed = {
        "fare_leg_rules": make_fare_leg_rules(["netA"]),
        "networks": make_networks(["netA"]),
    }
    notices = validate_fare_leg_rule_network_id_foreign_key(feed, CTX)
    assert len(notices) == 0


def test_both_parent_tables_absent():
    feed = {
        "fare_leg_rules": make_fare_leg_rules(["anyId"]),
    }
    notices = validate_fare_leg_rule_network_id_foreign_key(feed, CTX)
    assert len(notices) == 1
    assert notices[0].fields["fieldValue"] == "anyId"


def test_multiple_invalid_network_ids():
    feed = {
        "fare_leg_rules": make_fare_leg_rules(["bad1", "bad2", "bad3"]),
        "routes": make_routes(["other"]),
    }
    notices = validate_fare_leg_rule_network_id_foreign_key(feed, CTX)
    assert len(notices) == 3


def test_fare_leg_rules_empty_no_notices():
    feed = {
        "fare_leg_rules": pl.DataFrame({
            "network_id": pl.Series([], dtype=pl.Utf8),
            "csvRowNumber": pl.Series([], dtype=pl.Int64),
        }),
    }
    notices = validate_fare_leg_rule_network_id_foreign_key(feed, CTX)
    assert len(notices) == 0


def test_notice_fields_are_complete():
    feed = {
        "fare_leg_rules": make_fare_leg_rules(["unresolved"]),
    }
    notices = validate_fare_leg_rule_network_id_foreign_key(feed, CTX)
    assert len(notices) == 1
    fields = notices[0].fields
    assert set(fields.keys()) == {
        "childFilename",
        "childFieldName",
        "parentFilename",
        "parentFieldName",
        "fieldValue",
        "csvRowNumber",
    }
    assert fields["childFilename"] == "fare_leg_rules.txt"
    assert fields["parentFilename"] == "routes.txt or networks.txt"
    assert fields["parentFieldName"] == "network_id"


def test_duplicate_network_ids_in_parents_still_valid():
    feed = {
        "fare_leg_rules": make_fare_leg_rules(["dupId"]),
        "routes": make_routes(["dupId", "dupId"]),
    }
    notices = validate_fare_leg_rule_network_id_foreign_key(feed, CTX)
    assert len(notices) == 0
