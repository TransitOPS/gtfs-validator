"""Tests for validate_unique_geography_id."""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from gtfs_validator.context import ValidationContext
from gtfs_validator.notices import Severity
from gtfs_validator.validators.unique_geography_id import validate_unique_geography_id

CTX = ValidationContext(country_code="US", date_for_validation=date(2024, 1, 1))


def test_unique_geography_ids_no_notice() -> None:
    """All IDs unique across sources → no notices."""
    feed = {
        "stops": pl.DataFrame({
            "stop_id": ["stop1", "stop2"],
            "csv_row_number": [1, 2],
        }),
        "location_groups": pl.DataFrame({
            "location_group_id": ["lg1", "lg2"],
            "csv_row_number": [1, 2],
        }),
        "geojson_features": pl.DataFrame({
            "feature_id": ["feat1", "feat2"],
            "feature_index": [0, 1],
        }),
    }
    notices = validate_unique_geography_id(feed, CTX)
    assert notices == []


def test_duplicate_id_across_sources_yields_notice() -> None:
    """Same ID in stops, location_groups, and geojson → one notice."""
    feed = {
        "stops": pl.DataFrame({
            "stop_id": ["locationId"],
            "csv_row_number": [1],
        }),
        "location_groups": pl.DataFrame({
            "location_group_id": ["locationId"],
            "csv_row_number": [1],
        }),
        "geojson_features": pl.DataFrame({
            "feature_id": ["locationId"],
            "feature_index": [0],
        }),
    }
    notices = validate_unique_geography_id(feed, CTX)
    assert len(notices) == 1
    n = notices[0]
    assert n.code == "duplicate_geography_id"
    assert n.severity == Severity.ERROR
    assert n.fields["geography_id"] == "locationId"
    assert n.fields["csv_row_number_stops"] == 1
    assert n.fields["csv_row_number_location_groups"] == 1
    assert n.fields["feature_index"] == 0


def test_duplicate_id_stops_and_location_groups() -> None:
    """Same ID in stops and location_groups → notice."""
    feed = {
        "stops": pl.DataFrame({
            "stop_id": ["shared_id"],
            "csv_row_number": [5],
        }),
        "location_groups": pl.DataFrame({
            "location_group_id": ["shared_id"],
            "csv_row_number": [10],
        }),
    }
    notices = validate_unique_geography_id(feed, CTX)
    assert len(notices) == 1
    assert notices[0].fields["geography_id"] == "shared_id"
    assert notices[0].fields["csv_row_number_stops"] == 5
    assert notices[0].fields["csv_row_number_location_groups"] == 10
    assert notices[0].fields["feature_index"] is None


def test_duplicate_id_stops_and_geojson() -> None:
    """Same ID in stops and geojson → notice."""
    feed = {
        "stops": pl.DataFrame({
            "stop_id": ["geo_stop"],
            "csv_row_number": [3],
        }),
        "geojson_features": pl.DataFrame({
            "feature_id": ["geo_stop"],
            "feature_index": [2],
        }),
    }
    notices = validate_unique_geography_id(feed, CTX)
    assert len(notices) == 1
    assert notices[0].fields["csv_row_number_stops"] == 3
    assert notices[0].fields["csv_row_number_location_groups"] is None
    assert notices[0].fields["feature_index"] == 2


def test_duplicate_id_location_groups_and_geojson() -> None:
    """Same ID in location_groups and geojson → notice."""
    feed = {
        "location_groups": pl.DataFrame({
            "location_group_id": ["lg_geo"],
            "csv_row_number": [7],
        }),
        "geojson_features": pl.DataFrame({
            "feature_id": ["lg_geo"],
            "feature_index": [4],
        }),
    }
    notices = validate_unique_geography_id(feed, CTX)
    assert len(notices) == 1
    assert notices[0].fields["csv_row_number_stops"] is None
    assert notices[0].fields["csv_row_number_location_groups"] == 7
    assert notices[0].fields["feature_index"] == 4


def test_no_sources_no_notice() -> None:
    """No tables present → no notices."""
    feed: dict[str, pl.DataFrame] = {}
    notices = validate_unique_geography_id(feed, CTX)
    assert notices == []


def test_empty_tables_no_notice() -> None:
    """Empty tables → no notices."""
    feed = {
        "stops": pl.DataFrame({
            "stop_id": pl.Series([], dtype=pl.String),
            "csv_row_number": pl.Series([], dtype=pl.Int64),
        }),
        "location_groups": pl.DataFrame({
            "location_group_id": pl.Series([], dtype=pl.String),
            "csv_row_number": pl.Series([], dtype=pl.Int64),
        }),
    }
    notices = validate_unique_geography_id(feed, CTX)
    assert notices == []


def test_duplicate_within_same_source_no_notice() -> None:
    """Duplicate ID within same source (e.g., two stops with same ID) → no notice from this validator."""
    feed = {
        "stops": pl.DataFrame({
            "stop_id": ["dup", "dup"],
            "csv_row_number": [1, 2],
        }),
    }
    notices = validate_unique_geography_id(feed, CTX)
    # This validator only checks cross-source duplicates
    assert notices == []


def test_multiple_distinct_duplicates() -> None:
    """Multiple distinct duplicate IDs → multiple notices."""
    feed = {
        "stops": pl.DataFrame({
            "stop_id": ["dup1", "dup2"],
            "csv_row_number": [1, 2],
        }),
        "location_groups": pl.DataFrame({
            "location_group_id": ["dup1", "dup2"],
            "csv_row_number": [10, 20],
        }),
    }
    notices = validate_unique_geography_id(feed, CTX)
    assert len(notices) == 2
    geo_ids = {n.fields["geography_id"] for n in notices}
    assert geo_ids == {"dup1", "dup2"}


def test_notice_fields_are_complete() -> None:
    """Notice fields contain all expected keys."""
    feed = {
        "stops": pl.DataFrame({
            "stop_id": ["shared"],
            "csv_row_number": [1],
        }),
        "location_groups": pl.DataFrame({
            "location_group_id": ["shared"],
            "csv_row_number": [1],
        }),
    }
    notices = validate_unique_geography_id(feed, CTX)
    assert len(notices) == 1
    expected_keys = {
        "geography_id",
        "csv_row_number_stops",
        "csv_row_number_location_groups",
        "feature_index",
    }
    assert set(notices[0].fields.keys()) == expected_keys


def test_ctx_unused() -> None:
    """ctx value has no effect on the output."""
    wrong_ctx = ValidationContext(country_code="XX", date_for_validation=date(1900, 1, 1))
    feed = {
        "stops": pl.DataFrame({
            "stop_id": ["shared"],
            "csv_row_number": [1],
        }),
        "location_groups": pl.DataFrame({
            "location_group_id": ["shared"],
            "csv_row_number": [1],
        }),
    }
    notices = validate_unique_geography_id(feed, wrong_ctx)
    assert len(notices) == 1
    assert notices[0].fields["geography_id"] == "shared"