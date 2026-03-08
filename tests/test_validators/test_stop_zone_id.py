"""Tests for validate_stop_zone_id."""

from datetime import date

import polars as pl
import pytest

from gtfs_validator.context import ValidationContext
from gtfs_validator.notices import Severity
from gtfs_validator.validators.stop_zone_id import validate_stop_zone_id

CTX = ValidationContext(country_code="US", date_for_validation=date(2024, 1, 1))


def make_stops(rows: list[dict]) -> pl.DataFrame:
    schema = {
        "stop_id": pl.Utf8,
        "stop_name": pl.Utf8,
        "location_type": pl.Int64,
        "zone_id": pl.Utf8,
        "csv_row_number": pl.Int64,
    }
    if not rows:
        return pl.DataFrame(schema=schema)
    return pl.DataFrame(rows).cast(schema)


def make_fare_rules(rows: list[dict]) -> pl.DataFrame:
    schema = {
        "route_id": pl.Utf8,
        "origin_id": pl.Utf8,
        "destination_id": pl.Utf8,
        "contains_id": pl.Utf8,
    }
    if not rows:
        return pl.DataFrame(schema=schema)
    return pl.DataFrame(rows).cast(schema)


def make_stop_times(rows: list[dict]) -> pl.DataFrame:
    schema = {"stop_id": pl.Utf8, "trip_id": pl.Utf8}
    if not rows:
        return pl.DataFrame(schema=schema)
    return pl.DataFrame(rows).cast(schema)


def make_trips(rows: list[dict]) -> pl.DataFrame:
    schema = {"trip_id": pl.Utf8, "route_id": pl.Utf8}
    if not rows:
        return pl.DataFrame(schema=schema)
    return pl.DataFrame(rows).cast(schema)


def base_feed(stops=None, fare_rules=None, stop_times=None, trips=None) -> dict:
    """Build a minimal feed with sensible defaults for each table."""
    return {
        "stops": stops if stops is not None else make_stops([]),
        "fare_rules": fare_rules if fare_rules is not None else make_fare_rules([]),
        "stop_times": stop_times if stop_times is not None else make_stop_times([]),
        "trips": trips if trips is not None else make_trips([]),
    }


def test_fare_rules_absent_no_notice():
    """Guard 1: fare_rules key not in feed."""
    feed = {
        "stops": make_stops([{"stop_id": "S1", "stop_name": None, "location_type": 0, "zone_id": None, "csv_row_number": 2}]),
        "stop_times": make_stop_times([{"stop_id": "S1", "trip_id": "T1"}]),
        "trips": make_trips([{"trip_id": "T1", "route_id": "R1"}]),
    }
    assert validate_stop_zone_id(feed, CTX) == []


def test_fare_rules_empty_no_notice():
    """Guard 1: fare_rules present but zero rows."""
    feed = base_feed(
        fare_rules=make_fare_rules([]),
        stops=make_stops([{"stop_id": "S1", "stop_name": None, "location_type": 0, "zone_id": None, "csv_row_number": 2}]),
    )
    assert validate_stop_zone_id(feed, CTX) == []


def test_no_zone_structure_in_fare_rules_no_notice():
    """Guard 2: fare_rules has rows, but all zone columns are null."""
    feed = base_feed(
        fare_rules=make_fare_rules([{"route_id": "R1", "origin_id": None, "destination_id": None, "contains_id": None}]),
        stops=make_stops([{"stop_id": "S1", "stop_name": None, "location_type": 0, "zone_id": None, "csv_row_number": 2}]),
    )
    assert validate_stop_zone_id(feed, CTX) == []


def test_stop_zone_id_not_provided_route_not_in_fare_rules_no_notice():
    """Stop's route is not referenced in any fare rule with zone columns."""
    feed = base_feed(
        stops=make_stops([{"stop_id": "S1", "stop_name": None, "location_type": 0, "zone_id": None, "csv_row_number": 2}]),
        fare_rules=make_fare_rules([
            {"route_id": "R2", "origin_id": "Z1", "destination_id": None, "contains_id": None},
            {"route_id": "R3", "origin_id": None, "destination_id": "Z2", "contains_id": None},
            {"route_id": "R4", "origin_id": None, "destination_id": None, "contains_id": "Z3"},
            {"route_id": None, "origin_id": "Z1", "destination_id": None, "contains_id": None},
        ]),
        stop_times=make_stop_times([{"stop_id": "S1", "trip_id": "T1"}]),
        trips=make_trips([{"trip_id": "T1", "route_id": "R1"}]),
    )
    assert validate_stop_zone_id(feed, CTX) == []


def test_stop_zone_id_not_provided_route_in_fare_rules_no_zone_fields_no_notice():
    """Stop's route appears in fare rules, but none of those rows have zone columns set."""
    feed = base_feed(
        fare_rules=make_fare_rules([{"route_id": "R1", "origin_id": None, "destination_id": None, "contains_id": None}]),
        stops=make_stops([{"stop_id": "S1", "stop_name": None, "location_type": 0, "zone_id": None, "csv_row_number": 2}]),
        stop_times=make_stop_times([{"stop_id": "S1", "trip_id": "T1"}]),
        trips=make_trips([{"trip_id": "T1", "route_id": "R1"}]),
    )
    assert validate_stop_zone_id(feed, CTX) == []


def test_stop_zone_id_not_provided_route_in_fare_rules_with_origin_id_yields_notice():
    """Trigger: origin_id set."""
    feed = base_feed(
        fare_rules=make_fare_rules([{"route_id": "R1", "origin_id": "Z1", "destination_id": None, "contains_id": None}]),
        stops=make_stops([{"stop_id": "S1", "stop_name": "Main St", "location_type": 0, "zone_id": None, "csv_row_number": 2}]),
        stop_times=make_stop_times([{"stop_id": "S1", "trip_id": "T1"}]),
        trips=make_trips([{"trip_id": "T1", "route_id": "R1"}]),
    )
    notices = validate_stop_zone_id(feed, CTX)
    assert len(notices) == 1
    assert notices[0].code == "stop_without_zone_id"
    assert notices[0].severity == Severity.INFO
    assert notices[0].fields["stop_id"] == "S1"
    assert notices[0].fields["stop_name"] == "Main St"
    assert notices[0].fields["csv_row_number"] == 2


def test_stop_zone_id_not_provided_route_in_fare_rules_with_destination_id_yields_notice():
    """Trigger: destination_id set."""
    feed = base_feed(
        fare_rules=make_fare_rules([{"route_id": "R1", "origin_id": None, "destination_id": "Z2", "contains_id": None}]),
        stops=make_stops([{"stop_id": "S1", "stop_name": "Main St", "location_type": 0, "zone_id": None, "csv_row_number": 2}]),
        stop_times=make_stop_times([{"stop_id": "S1", "trip_id": "T1"}]),
        trips=make_trips([{"trip_id": "T1", "route_id": "R1"}]),
    )
    notices = validate_stop_zone_id(feed, CTX)
    assert len(notices) == 1
    assert notices[0].fields["stop_id"] == "S1"


def test_stop_zone_id_not_provided_route_in_fare_rules_with_contains_id_yields_notice():
    """Trigger: contains_id set."""
    feed = base_feed(
        fare_rules=make_fare_rules([{"route_id": "R1", "origin_id": None, "destination_id": None, "contains_id": "Z3"}]),
        stops=make_stops([{"stop_id": "S1", "stop_name": "Main St", "location_type": 0, "zone_id": None, "csv_row_number": 2}]),
        stop_times=make_stop_times([{"stop_id": "S1", "trip_id": "T1"}]),
        trips=make_trips([{"trip_id": "T1", "route_id": "R1"}]),
    )
    notices = validate_stop_zone_id(feed, CTX)
    assert len(notices) == 1
    assert notices[0].fields["stop_id"] == "S1"


def test_stop_zone_id_not_provided_mixed_fare_rules_yields_notice():
    """Stop's route in two fare rules: one without zone fields, one with origin_id."""
    feed = base_feed(
        fare_rules=make_fare_rules([
            {"route_id": "R1", "origin_id": None, "destination_id": None, "contains_id": None},
            {"route_id": "R1", "origin_id": "Z1", "destination_id": None, "contains_id": None},
        ]),
        stops=make_stops([{"stop_id": "S1", "stop_name": "Main St", "location_type": 0, "zone_id": None, "csv_row_number": 2}]),
        stop_times=make_stop_times([{"stop_id": "S1", "trip_id": "T1"}]),
        trips=make_trips([{"trip_id": "T1", "route_id": "R1"}]),
    )
    notices = validate_stop_zone_id(feed, CTX)
    assert len(notices) == 1
    assert notices[0].fields["stop_id"] == "S1"


def test_stop_zone_id_provided_no_notice():
    """Stop has zone_id set; it is not a candidate."""
    feed = base_feed(
        stops=make_stops([{"stop_id": "S1", "stop_name": None, "location_type": 0, "zone_id": "Z1", "csv_row_number": 2}]),
        fare_rules=make_fare_rules([
            {"route_id": "R1", "origin_id": "Z1", "destination_id": None, "contains_id": None},
            {"route_id": "R1", "origin_id": None, "destination_id": "Z2", "contains_id": None},
            {"route_id": "R1", "origin_id": None, "destination_id": None, "contains_id": "Z3"},
            {"route_id": "R1", "origin_id": "Z1", "destination_id": "Z2", "contains_id": None},
            {"route_id": "R1", "origin_id": "Z1", "destination_id": None, "contains_id": "Z3"},
            {"route_id": "R1", "origin_id": None, "destination_id": "Z2", "contains_id": "Z3"},
        ]),
        stop_times=make_stop_times([{"stop_id": "S1", "trip_id": "T1"}]),
        trips=make_trips([{"trip_id": "T1", "route_id": "R1"}]),
    )
    assert validate_stop_zone_id(feed, CTX) == []


def test_non_stop_location_types_no_notice():
    """Stops with location_type != 0 are unconditionally skipped."""
    feed = base_feed(
        stops=make_stops([
            {"stop_id": "S1", "stop_name": None, "location_type": 1, "zone_id": None, "csv_row_number": 2},
            {"stop_id": "S2", "stop_name": None, "location_type": 2, "zone_id": None, "csv_row_number": 3},
            {"stop_id": "S3", "stop_name": None, "location_type": 3, "zone_id": None, "csv_row_number": 4},
            {"stop_id": "S4", "stop_name": None, "location_type": 4, "zone_id": None, "csv_row_number": 5},
        ]),
        fare_rules=make_fare_rules([{"route_id": "R1", "origin_id": "Z1", "destination_id": None, "contains_id": None}]),
        stop_times=make_stop_times([
            {"stop_id": "S1", "trip_id": "T1"},
            {"stop_id": "S2", "trip_id": "T1"},
            {"stop_id": "S3", "trip_id": "T1"},
            {"stop_id": "S4", "trip_id": "T1"},
        ]),
        trips=make_trips([{"trip_id": "T1", "route_id": "R1"}]),
    )
    assert validate_stop_zone_id(feed, CTX) == []


def test_stop_not_referenced_in_stop_times_no_notice():
    """Stop is a valid STOP (type 0, no zone_id) but not referenced in stop_times."""
    feed = base_feed(
        stops=make_stops([{"stop_id": "S1", "stop_name": None, "location_type": 0, "zone_id": None, "csv_row_number": 2}]),
        stop_times=make_stop_times([]),
        fare_rules=make_fare_rules([{"route_id": "R1", "origin_id": "Z1", "destination_id": None, "contains_id": None}]),
        trips=make_trips([{"trip_id": "T1", "route_id": "R1"}]),
    )
    assert validate_stop_zone_id(feed, CTX) == []


def test_stop_trip_id_not_in_trips_no_notice():
    """Stop is referenced in stop_times, but the trip_id is not in trips.txt."""
    feed = base_feed(
        stop_times=make_stop_times([{"stop_id": "S1", "trip_id": "T_MISSING"}]),
        trips=make_trips([]),
        fare_rules=make_fare_rules([{"route_id": "R1", "origin_id": "Z1", "destination_id": None, "contains_id": None}]),
        stops=make_stops([{"stop_id": "S1", "stop_name": None, "location_type": 0, "zone_id": None, "csv_row_number": 2}]),
    )
    assert validate_stop_zone_id(feed, CTX) == []


def test_fare_rule_with_null_route_id_no_notice():
    """Fare rule has zone columns set but route_id=None (global zone rule)."""
    feed = base_feed(
        fare_rules=make_fare_rules([{"route_id": None, "origin_id": "Z1", "destination_id": None, "contains_id": None}]),
        stops=make_stops([{"stop_id": "S1", "stop_name": None, "location_type": 0, "zone_id": None, "csv_row_number": 2}]),
        stop_times=make_stop_times([{"stop_id": "S1", "trip_id": "T1"}]),
        trips=make_trips([{"trip_id": "T1", "route_id": "R1"}]),
    )
    assert validate_stop_zone_id(feed, CTX) == []


def test_multiple_violating_stops_one_notice_each():
    """Multiple stops each missing zone_id, each reachable from zone-fare routes."""
    feed = base_feed(
        stops=make_stops([
            {"stop_id": "S1", "stop_name": "Stop 1", "location_type": 0, "zone_id": None, "csv_row_number": 2},
            {"stop_id": "S2", "stop_name": "Stop 2", "location_type": 0, "zone_id": None, "csv_row_number": 3},
            {"stop_id": "S3", "stop_name": "Stop 3", "location_type": 0, "zone_id": None, "csv_row_number": 4},
        ]),
        fare_rules=make_fare_rules([{"route_id": "R1", "origin_id": "Z1", "destination_id": None, "contains_id": None}]),
        stop_times=make_stop_times([
            {"stop_id": "S1", "trip_id": "T1"},
            {"stop_id": "S2", "trip_id": "T1"},
            {"stop_id": "S3", "trip_id": "T1"},
        ]),
        trips=make_trips([{"trip_id": "T1", "route_id": "R1"}]),
    )
    notices = validate_stop_zone_id(feed, CTX)
    assert len(notices) == 3
    stop_ids = {n.fields["stop_id"] for n in notices}
    assert stop_ids == {"S1", "S2", "S3"}


def test_multiple_routes_one_in_zone_fare_triggers_notice():
    """A stop is served by two routes; only one route appears in a zone-dependent fare rule."""
    feed = base_feed(
        stop_times=make_stop_times([
            {"stop_id": "S1", "trip_id": "T1"},
            {"stop_id": "S1", "trip_id": "T2"},
        ]),
        trips=make_trips([
            {"trip_id": "T1", "route_id": "R1"},
            {"trip_id": "T2", "route_id": "R2"},
        ]),
        fare_rules=make_fare_rules([{"route_id": "R1", "origin_id": "Z1", "destination_id": None, "contains_id": None}]),
        stops=make_stops([{"stop_id": "S1", "stop_name": None, "location_type": 0, "zone_id": None, "csv_row_number": 2}]),
    )
    notices = validate_stop_zone_id(feed, CTX)
    assert len(notices) == 1
    assert notices[0].fields["stop_id"] == "S1"


def test_stops_absent_no_notice():
    """Feed has no stops key."""
    feed = {
        "fare_rules": make_fare_rules([{"route_id": "R1", "origin_id": "Z1", "destination_id": None, "contains_id": None}]),
        "stop_times": make_stop_times([{"stop_id": "S1", "trip_id": "T1"}]),
        "trips": make_trips([{"trip_id": "T1", "route_id": "R1"}]),
    }
    assert validate_stop_zone_id(feed, CTX) == []


def test_stop_times_absent_no_notice():
    """Feed has no stop_times key."""
    feed = {
        "stops": make_stops([{"stop_id": "S1", "stop_name": None, "location_type": 0, "zone_id": None, "csv_row_number": 2}]),
        "fare_rules": make_fare_rules([{"route_id": "R1", "origin_id": "Z1", "destination_id": None, "contains_id": None}]),
        "trips": make_trips([{"trip_id": "T1", "route_id": "R1"}]),
    }
    assert validate_stop_zone_id(feed, CTX) == []


def test_trips_absent_no_notice():
    """Feed has no trips key."""
    feed = {
        "stops": make_stops([{"stop_id": "S1", "stop_name": None, "location_type": 0, "zone_id": None, "csv_row_number": 2}]),
        "fare_rules": make_fare_rules([{"route_id": "R1", "origin_id": "Z1", "destination_id": None, "contains_id": None}]),
        "stop_times": make_stop_times([{"stop_id": "S1", "trip_id": "T1"}]),
    }
    assert validate_stop_zone_id(feed, CTX) == []


def test_notice_code_and_severity():
    """Assert exact notice metadata."""
    feed = base_feed(
        fare_rules=make_fare_rules([{"route_id": "R1", "origin_id": "Z1", "destination_id": None, "contains_id": None}]),
        stops=make_stops([{"stop_id": "S1", "stop_name": "Main St", "location_type": 0, "zone_id": None, "csv_row_number": 2}]),
        stop_times=make_stop_times([{"stop_id": "S1", "trip_id": "T1"}]),
        trips=make_trips([{"trip_id": "T1", "route_id": "R1"}]),
    )
    notices = validate_stop_zone_id(feed, CTX)
    assert len(notices) == 1
    assert notices[0].code == "stop_without_zone_id"
    assert notices[0].severity == Severity.INFO


def test_all_stops_have_zone_id_no_notice():
    """All STOP-type stops have zone_id set."""
    feed = base_feed(
        stops=make_stops([
            {"stop_id": "S1", "stop_name": None, "location_type": 0, "zone_id": "Z1", "csv_row_number": 2},
            {"stop_id": "S2", "stop_name": None, "location_type": 0, "zone_id": "Z2", "csv_row_number": 3},
        ]),
        fare_rules=make_fare_rules([{"route_id": "R1", "origin_id": "Z1", "destination_id": None, "contains_id": None}]),
        stop_times=make_stop_times([
            {"stop_id": "S1", "trip_id": "T1"},
            {"stop_id": "S2", "trip_id": "T1"},
        ]),
        trips=make_trips([{"trip_id": "T1", "route_id": "R1"}]),
    )
    assert validate_stop_zone_id(feed, CTX) == []
