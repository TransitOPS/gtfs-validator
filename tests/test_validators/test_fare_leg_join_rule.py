"""Tests for validate_fare_leg_join_rule."""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from gtfs_validator.context import ValidationContext
from gtfs_validator.notices import Severity
from gtfs_validator.validators.fare_leg_join_rule import validate_fare_leg_join_rule

CTX = ValidationContext(country_code="US", date_for_validation=date(2024, 1, 1))


def make_join_rules(**kwargs) -> pl.DataFrame:
    """Build a fare_leg_join_rules DataFrame with sensible defaults."""
    defaults: dict[str, list] = {
        "from_network_id": ["network1"],
        "to_network_id": ["network2"],
        "from_stop_id": [None],
        "to_stop_id": [None],
    }
    defaults.update(kwargs)
    return pl.DataFrame(defaults)


def make_networks(network_ids: list[str | None]) -> pl.DataFrame:
    return pl.DataFrame({"network_id": network_ids})


def make_routes(network_ids: list[str | None]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "route_id": [f"r{i}" for i in range(len(network_ids))],
            "network_id": network_ids,
        }
    )


# ---------------------------------------------------------------------------
# Test 1: missing FK for to_network_id (Java contract)
# ---------------------------------------------------------------------------

def test_missing_foreign_key_to_network_id_yields_notice():
    feed = {
        "fare_leg_join_rules": make_join_rules(
            from_network_id=["network1"],
            to_network_id=["network2"],
        ),
        "routes": make_routes(["network1"]),
        "networks": make_networks(["network1"]),
    }
    notices = validate_fare_leg_join_rule(feed, CTX)
    assert len(notices) == 1
    n = notices[0]
    assert n.code == "foreign_key_violation"
    assert n.severity == Severity.ERROR
    assert n.fields["child_field_name"] == "to_network_id"
    assert n.fields["field_value"] == "network2"
    assert n.fields["parent_filename"] == "routes.txt or networks.txt"


# ---------------------------------------------------------------------------
# Test 2: missing_required_field for to_stop_id (Java contract)
# ---------------------------------------------------------------------------

def test_missing_required_field_to_stop_id_yields_notice():
    feed = {
        "fare_leg_join_rules": make_join_rules(
            from_network_id=["network1"],
            to_network_id=["network2"],
            from_stop_id=["stop1"],
            to_stop_id=[None],
        ),
        "routes": make_routes(["network1"]),
        "networks": make_networks(["network2"]),
    }
    notices = validate_fare_leg_join_rule(feed, CTX)
    assert len(notices) == 1
    n = notices[0]
    assert n.code == "missing_required_field"
    assert n.fields["field_name"] == "to_stop_id"


# ---------------------------------------------------------------------------
# Test 3: both network IDs valid in networks.txt, no routes table
# ---------------------------------------------------------------------------

def test_both_network_ids_valid_in_networks_no_notices():
    feed = {
        "fare_leg_join_rules": make_join_rules(
            from_network_id=["n1"],
            to_network_id=["n2"],
        ),
        "networks": make_networks(["n1", "n2"]),
    }
    notices = validate_fare_leg_join_rule(feed, CTX)
    assert notices == []


# ---------------------------------------------------------------------------
# Test 4: both network IDs valid via routes.txt (no networks table)
# ---------------------------------------------------------------------------

def test_both_network_ids_valid_via_routes_no_notices():
    feed = {
        "fare_leg_join_rules": make_join_rules(
            from_network_id=["n1"],
            to_network_id=["n2"],
        ),
        "routes": make_routes(["n1", "n2"]),
    }
    notices = validate_fare_leg_join_rule(feed, CTX)
    assert notices == []


# ---------------------------------------------------------------------------
# Test 5: both network IDs invalid — two FK notices
# ---------------------------------------------------------------------------

def test_both_network_ids_invalid_yields_two_notices():
    feed = {
        "fare_leg_join_rules": make_join_rules(
            from_network_id=["bad1"],
            to_network_id=["bad2"],
        ),
        "networks": make_networks(["other"]),
        "routes": make_routes(["other"]),
    }
    notices = validate_fare_leg_join_rule(feed, CTX)
    assert len(notices) == 2
    codes = [n.code for n in notices]
    assert all(c == "foreign_key_violation" for c in codes)
    fields_names = [n.fields["child_field_name"] for n in notices]
    assert "from_network_id" in fields_names
    assert "to_network_id" in fields_names


# ---------------------------------------------------------------------------
# Test 6: both stop IDs absent — no missing_required_field notices
# ---------------------------------------------------------------------------

def test_both_stop_ids_absent_no_notices():
    feed = {
        "fare_leg_join_rules": make_join_rules(
            from_network_id=["n1"],
            to_network_id=["n2"],
            from_stop_id=[None],
            to_stop_id=[None],
        ),
        "networks": make_networks(["n1", "n2"]),
    }
    notices = validate_fare_leg_join_rule(feed, CTX)
    assert notices == []


# ---------------------------------------------------------------------------
# Test 7: both stop IDs present — no missing_required_field notices
# ---------------------------------------------------------------------------

def test_both_stop_ids_present_no_notices():
    feed = {
        "fare_leg_join_rules": make_join_rules(
            from_network_id=["n1"],
            to_network_id=["n2"],
            from_stop_id=["stop1"],
            to_stop_id=["stop2"],
        ),
        "networks": make_networks(["n1", "n2"]),
    }
    notices = validate_fare_leg_join_rule(feed, CTX)
    assert notices == []


# ---------------------------------------------------------------------------
# Test 8: to_stop_id present, from_stop_id absent
# ---------------------------------------------------------------------------

def test_to_stop_id_present_from_stop_id_absent_yields_notice():
    feed = {
        "fare_leg_join_rules": make_join_rules(
            from_network_id=["n1"],
            to_network_id=["n2"],
            from_stop_id=[None],
            to_stop_id=["stop2"],
        ),
        "networks": make_networks(["n1", "n2"]),
    }
    notices = validate_fare_leg_join_rule(feed, CTX)
    assert len(notices) == 1
    assert notices[0].code == "missing_required_field"
    assert notices[0].fields["field_name"] == "from_stop_id"


# ---------------------------------------------------------------------------
# Test 9: from_stop_id present, to_stop_id absent
# ---------------------------------------------------------------------------

def test_from_stop_id_present_to_stop_id_absent_yields_notice():
    feed = {
        "fare_leg_join_rules": make_join_rules(
            from_network_id=["n1"],
            to_network_id=["n2"],
            from_stop_id=["stop1"],
            to_stop_id=[None],
        ),
        "networks": make_networks(["n1", "n2"]),
    }
    notices = validate_fare_leg_join_rule(feed, CTX)
    assert len(notices) == 1
    assert notices[0].code == "missing_required_field"
    assert notices[0].fields["field_name"] == "to_stop_id"


# ---------------------------------------------------------------------------
# Test 10: fare_leg_join_rules absent — skip guard fires
# ---------------------------------------------------------------------------

def test_fare_leg_join_rules_absent_skips():
    feed: dict = {}
    notices = validate_fare_leg_join_rule(feed, CTX)
    assert notices == []


# ---------------------------------------------------------------------------
# Test 11: fare_leg_join_rules empty — skip guard fires
# ---------------------------------------------------------------------------

def test_fare_leg_join_rules_empty_skips():
    feed = {
        "fare_leg_join_rules": pl.DataFrame(
            {"from_network_id": [], "to_network_id": []}
        ),
    }
    notices = validate_fare_leg_join_rule(feed, CTX)
    assert notices == []


# ---------------------------------------------------------------------------
# Test 12: both parent tables absent — all FK violations
# ---------------------------------------------------------------------------

def test_both_parent_tables_absent_all_fk_violations():
    feed = {
        "fare_leg_join_rules": make_join_rules(
            from_network_id=["n1"],
            to_network_id=["n2"],
        ),
    }
    notices = validate_fare_leg_join_rule(feed, CTX)
    assert len(notices) == 2
    assert all(n.code == "foreign_key_violation" for n in notices)


# ---------------------------------------------------------------------------
# Test 13: network_id found only in routes, no networks table
# ---------------------------------------------------------------------------

def test_network_id_found_only_in_routes_no_notice():
    feed = {
        "fare_leg_join_rules": make_join_rules(
            from_network_id=["n1"],
            to_network_id=["n2"],
        ),
        "routes": make_routes(["n1", "n2"]),
    }
    notices = validate_fare_leg_join_rule(feed, CTX)
    assert notices == []


# ---------------------------------------------------------------------------
# Test 14: empty string from_network_id treated as absent
# ---------------------------------------------------------------------------

def test_empty_string_from_network_id_treated_as_absent():
    feed = {
        "fare_leg_join_rules": make_join_rules(
            from_network_id=[""],
            to_network_id=["n1"],
        ),
        # No networks or routes — to_network_id="n1" would fail
    }
    notices = validate_fare_leg_join_rule(feed, CTX)
    assert len(notices) == 1
    assert notices[0].fields["child_field_name"] == "to_network_id"


# ---------------------------------------------------------------------------
# Test 15: row with FK violation and missing stop ID emits both
# ---------------------------------------------------------------------------

def test_row_with_fk_violation_and_missing_stop_id_emits_both():
    feed = {
        "fare_leg_join_rules": make_join_rules(
            from_network_id=["bad"],
            to_network_id=["good"],
            from_stop_id=["stop1"],
            to_stop_id=[None],
        ),
        "networks": make_networks(["good"]),
    }
    notices = validate_fare_leg_join_rule(feed, CTX)
    assert len(notices) == 2
    codes = {n.code for n in notices}
    assert "foreign_key_violation" in codes
    assert "missing_required_field" in codes
    fk = next(n for n in notices if n.code == "foreign_key_violation")
    assert fk.fields["child_field_name"] == "from_network_id"
    mrf = next(n for n in notices if n.code == "missing_required_field")
    assert mrf.fields["field_name"] == "to_stop_id"


# ---------------------------------------------------------------------------
# Test 16: multiple rows each produce independent notices
# ---------------------------------------------------------------------------

def test_multiple_rows_each_produce_independent_notices():
    # row 0: to_network_id invalid; row 1: from_network_id invalid
    feed = {
        "fare_leg_join_rules": pl.DataFrame(
            {
                "from_network_id": ["good", "bad"],
                "to_network_id": ["bad", "good"],
                "from_stop_id": [None, None],
                "to_stop_id": [None, None],
            }
        ),
        "networks": make_networks(["good"]),
    }
    notices = validate_fare_leg_join_rule(feed, CTX)
    assert len(notices) == 2
    assert all(n.code == "foreign_key_violation" for n in notices)
    # from_network_id loop runs first: row 1 (idx=1) has bad from_network_id
    from_notice = next(n for n in notices if n.fields["child_field_name"] == "from_network_id")
    to_notice = next(n for n in notices if n.fields["child_field_name"] == "to_network_id")
    assert from_notice.fields["csv_row_number"] == 1
    assert to_notice.fields["csv_row_number"] == 0


# ---------------------------------------------------------------------------
# Test 17: parent_filename wire format
# ---------------------------------------------------------------------------

def test_parent_filename_wire_format():
    feed = {
        "fare_leg_join_rules": make_join_rules(
            from_network_id=["bad"],
            to_network_id=["n2"],
        ),
    }
    notices = validate_fare_leg_join_rule(feed, CTX)
    fk_notices = [n for n in notices if n.code == "foreign_key_violation"]
    assert len(fk_notices) >= 1
    for n in fk_notices:
        assert n.fields["parent_filename"] == "routes.txt or networks.txt"


# ---------------------------------------------------------------------------
# Test 18: networks absent, routes present with valid ID
# ---------------------------------------------------------------------------

def test_networks_absent_routes_present_valid_id():
    feed = {
        "fare_leg_join_rules": make_join_rules(
            from_network_id=["n1"],
            to_network_id=["n1"],
        ),
        "routes": make_routes(["n1"]),
    }
    notices = validate_fare_leg_join_rule(feed, CTX)
    assert notices == []


# ---------------------------------------------------------------------------
# Test 19: routes absent, networks present with valid ID
# ---------------------------------------------------------------------------

def test_routes_absent_networks_present_valid_id():
    feed = {
        "fare_leg_join_rules": make_join_rules(
            from_network_id=["n1"],
            to_network_id=["n1"],
        ),
        "networks": make_networks(["n1"]),
    }
    notices = validate_fare_leg_join_rule(feed, CTX)
    assert notices == []


# ---------------------------------------------------------------------------
# Test 20: null network_id in routes not included in valid set
# ---------------------------------------------------------------------------

def test_null_network_id_in_routes_not_included_in_valid_set():
    feed = {
        "fare_leg_join_rules": make_join_rules(
            from_network_id=["n1"],
            to_network_id=["n1"],
        ),
        "routes": make_routes([None]),
    }
    notices = validate_fare_leg_join_rule(feed, CTX)
    fk_notices = [n for n in notices if n.code == "foreign_key_violation"]
    assert len(fk_notices) == 2  # both from and to are invalid
