"""Tests for validate_unique_geography_id."""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from gtfs_validator.context import ValidationContext
from gtfs_validator.notices import Severity
from gtfs_validator.validators.unique_geography_id import validate_unique_geography_id


@pytest.fixture
def ctx() -> ValidationContext:
    return ValidationContext(country_code="US", date_for_validation=date(2024, 1, 1))


def make_stops(rows: list[dict]) -> pl.DataFrame:
    """Build a minimal stops DataFrame with stop_id and csv_row_number."""
    return pl.DataFrame(
        {
            "stop_id": [r["stop_id"] for r in rows],
            "csv_row_number": [r["csv_row_number"] for r in rows],
        },
        schema={"stop_id": pl.Utf8, "csv_row_number": pl.Int64},
    )


def make_location_groups(rows: list[dict]) -> pl.DataFrame:
    """Build a minimal location_groups DataFrame."""
    return pl.DataFrame(
        {
            "location_group_id": [r["location_group_id"] for r in rows],
            "csv_row_number": [r["csv_row_number"] for r in rows],
        },
        schema={"location_group_id": pl.Utf8, "csv_row_number": pl.Int64},
    )


# --- Tests derived from Java contracts ---


def test_three_way_collision_yields_notice(ctx: ValidationContext) -> None:
    feed = {
        "stops": make_stops([{"stop_id": "locationId", "csv_row_number": 1}]),
        "location_groups": make_location_groups([{"location_group_id": "locationId", "csv_row_number": 1}]),
        "locations_geojson": [{"id": "locationId", "index": 0}],
    }
    notices = validate_unique_geography_id(feed, ctx)
    assert len(notices) == 1
    n = notices[0]
    assert n.code == "duplicate_geography_id"
    assert n.severity == Severity.ERROR
    assert n.fields["geography_id"] == "locationId"
    assert n.fields["csv_row_number_stops"] == 1
    assert n.fields["csv_row_number_location_groups"] == 1
    assert n.fields["feature_index"] == 0


def test_unique_ids_yields_no_notice(ctx: ValidationContext) -> None:
    feed = {
        "stops": make_stops([{"stop_id": "locationId2", "csv_row_number": 1}]),
        "location_groups": make_location_groups([{"location_group_id": "locationId3", "csv_row_number": 1}]),
        "locations_geojson": [{"id": "locationId1", "index": 0}],
    }
    notices = validate_unique_geography_id(feed, ctx)
    assert notices == []


# --- Additional Python-specific edge cases ---


def test_all_sources_absent_yields_no_notice(ctx: ValidationContext) -> None:
    notices = validate_unique_geography_id({}, ctx)
    assert notices == []


def test_only_stops_present_yields_no_notice(ctx: ValidationContext) -> None:
    feed = {
        "stops": make_stops([{"stop_id": "x", "csv_row_number": 1}]),
    }
    notices = validate_unique_geography_id(feed, ctx)
    assert notices == []


def test_only_geojson_present_yields_no_notice(ctx: ValidationContext) -> None:
    feed: dict = {
        "locations_geojson": [{"id": "x", "index": 0}],
    }
    notices = validate_unique_geography_id(feed, ctx)
    assert notices == []


def test_two_way_collision_stops_and_geojson(ctx: ValidationContext) -> None:
    feed = {
        "stops": make_stops([{"stop_id": "x", "csv_row_number": 2}]),
        "locations_geojson": [{"id": "x", "index": 0}],
    }
    notices = validate_unique_geography_id(feed, ctx)
    assert len(notices) == 1
    n = notices[0]
    assert n.code == "duplicate_geography_id"
    assert n.severity == Severity.ERROR
    assert n.fields["geography_id"] == "x"
    assert n.fields["csv_row_number_stops"] == 2
    assert n.fields["csv_row_number_location_groups"] is None
    assert n.fields["feature_index"] == 0


def test_two_way_collision_stops_and_location_groups(ctx: ValidationContext) -> None:
    feed = {
        "stops": make_stops([{"stop_id": "y", "csv_row_number": 3}]),
        "location_groups": make_location_groups([{"location_group_id": "y", "csv_row_number": 5}]),
    }
    notices = validate_unique_geography_id(feed, ctx)
    assert len(notices) == 1
    n = notices[0]
    assert n.code == "duplicate_geography_id"
    assert n.severity == Severity.ERROR
    assert n.fields["geography_id"] == "y"
    assert n.fields["csv_row_number_stops"] == 3
    assert n.fields["csv_row_number_location_groups"] == 5
    assert n.fields["feature_index"] is None


def test_same_source_duplicate_only_yields_no_notice(ctx: ValidationContext) -> None:
    feed = {
        "stops": make_stops([
            {"stop_id": "x", "csv_row_number": 1},
            {"stop_id": "x", "csv_row_number": 2},
        ]),
    }
    notices = validate_unique_geography_id(feed, ctx)
    assert notices == []


def test_multiple_colliding_ids_yield_multiple_notices(ctx: ValidationContext) -> None:
    feed = {
        "stops": make_stops([
            {"stop_id": "a", "csv_row_number": 1},
            {"stop_id": "b", "csv_row_number": 2},
        ]),
        "locations_geojson": [
            {"id": "a", "index": 0},
            {"id": "b", "index": 1},
        ],
    }
    notices = validate_unique_geography_id(feed, ctx)
    assert len(notices) == 2
    codes = {n.code for n in notices}
    assert codes == {"duplicate_geography_id"}
    geo_ids = {n.fields["geography_id"] for n in notices}
    assert geo_ids == {"a", "b"}


def test_null_ids_are_skipped(ctx: ValidationContext) -> None:
    feed: dict = {
        "stops": pl.DataFrame(
            {"stop_id": [None], "csv_row_number": [1]},
            schema={"stop_id": pl.Utf8, "csv_row_number": pl.Int64},
        ),
        "locations_geojson": [{"id": None, "index": 0}],
    }
    notices = validate_unique_geography_id(feed, ctx)
    assert notices == []


def test_first_occurrence_row_number_reported(ctx: ValidationContext) -> None:
    feed = {
        "stops": make_stops([
            {"stop_id": "x", "csv_row_number": 1},
            {"stop_id": "x", "csv_row_number": 3},
        ]),
        "locations_geojson": [{"id": "x", "index": 0}],
    }
    notices = validate_unique_geography_id(feed, ctx)
    assert len(notices) == 1
    n = notices[0]
    assert n.fields["geography_id"] == "x"
    assert n.fields["csv_row_number_stops"] == 1  # first occurrence only
    assert n.fields["feature_index"] == 0
