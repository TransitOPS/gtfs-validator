"""Tests for LocationIdForeignKeyValidator."""

from datetime import date

import polars as pl
import pytest

from gtfs_validator.context import ValidationContext
from gtfs_validator.notices import Severity
from gtfs_validator.validators.location_id_foreign_key import (
    validate_location_id_foreign_key,
)

CTX = ValidationContext(country_code="US", date_for_validation=date(2026, 3, 8))


def make_stop_times(
    rows: list[tuple[int, str | None]],
) -> pl.DataFrame:
    """Each row is (csvRowNumber, location_id)."""
    return pl.DataFrame(
        {
            "csvRowNumber": [r[0] for r in rows],
            "location_id": [r[1] for r in rows],
        },
        schema={
            "csvRowNumber": pl.Int64,
            "location_id": pl.Utf8,
        },
    )


def make_geojson_features(ids: list[str]) -> list[dict]:
    """Create minimal GeoJSON feature dicts with the given IDs."""
    return [
        {"id": fid, "index": i, "geometry_type": "Polygon", "coordinates": []}
        for i, fid in enumerate(ids)
    ]


def test_missing_geojson_id_yields_notice():
    feed = {
        "stop_times": make_stop_times([(2, "locationId2")]),
        "locations_geojson": make_geojson_features(["locationId"]),
    }
    notices = validate_location_id_foreign_key(feed, CTX)
    assert len(notices) == 1
    n = notices[0]
    assert n.code == "foreign_key_violation"
    assert n.severity == Severity.ERROR
    assert n.fields["childFilename"] == "stop_times.txt"
    assert n.fields["childFieldName"] == "location_id"
    assert n.fields["parentFilename"] == "locations.geojson"
    assert n.fields["parentFieldName"] == "id"
    assert n.fields["fieldValue"] == "locationId2"
    assert n.fields["csvRowNumber"] == 2


def test_existing_geojson_id_yields_no_notice():
    feed = {
        "stop_times": make_stop_times([(2, "locationId")]),
        "locations_geojson": make_geojson_features(["locationId"]),
    }
    notices = validate_location_id_foreign_key(feed, CTX)
    assert len(notices) == 0


def test_stop_times_missing_no_notices():
    feed = {"locations_geojson": make_geojson_features(["a"])}
    notices = validate_location_id_foreign_key(feed, CTX)
    assert len(notices) == 0


def test_locations_geojson_missing_no_notices():
    feed = {"stop_times": make_stop_times([(2, "a")])}
    notices = validate_location_id_foreign_key(feed, CTX)
    assert len(notices) == 0


def test_locations_geojson_empty_no_notices():
    feed = {
        "stop_times": make_stop_times([(2, "a")]),
        "locations_geojson": [],
    }
    notices = validate_location_id_foreign_key(feed, CTX)
    assert len(notices) == 0


def test_location_id_column_missing_no_notices():
    stop_times = pl.DataFrame(
        {
            "csvRowNumber": [2],
            "stop_id": ["s1"],
        },
        schema={
            "csvRowNumber": pl.Int64,
            "stop_id": pl.Utf8,
        },
    )
    feed = {
        "stop_times": stop_times,
        "locations_geojson": make_geojson_features(["a"]),
    }
    notices = validate_location_id_foreign_key(feed, CTX)
    assert len(notices) == 0


def test_location_id_null_no_notice():
    feed = {
        "stop_times": make_stop_times([(2, None)]),
        "locations_geojson": make_geojson_features(["locationId"]),
    }
    notices = validate_location_id_foreign_key(feed, CTX)
    assert len(notices) == 0


def test_location_id_empty_string_no_notice():
    feed = {
        "stop_times": make_stop_times([(2, "")]),
        "locations_geojson": make_geojson_features(["locationId"]),
    }
    notices = validate_location_id_foreign_key(feed, CTX)
    assert len(notices) == 0


def test_multiple_rows_same_missing_id_yields_multiple_notices():
    feed = {
        "stop_times": make_stop_times([(2, "b"), (3, "b"), (4, "b")]),
        "locations_geojson": make_geojson_features(["a"]),
    }
    notices = validate_location_id_foreign_key(feed, CTX)
    assert len(notices) == 3
    row_numbers = sorted(n.fields["csvRowNumber"] for n in notices)
    assert row_numbers == [2, 3, 4]


def test_mixed_valid_and_invalid_ids():
    feed = {
        "stop_times": make_stop_times([(2, "a"), (3, "c"), (4, "b"), (5, "d")]),
        "locations_geojson": make_geojson_features(["a", "b"]),
    }
    notices = validate_location_id_foreign_key(feed, CTX)
    assert len(notices) == 2
    violations = sorted(notices, key=lambda n: n.fields["csvRowNumber"])
    assert violations[0].fields["fieldValue"] == "c"
    assert violations[0].fields["csvRowNumber"] == 3
    assert violations[1].fields["fieldValue"] == "d"
    assert violations[1].fields["csvRowNumber"] == 5


def test_multiple_geojson_features_all_matched():
    feed = {
        "stop_times": make_stop_times([(2, "loc1"), (3, "loc2"), (4, "loc3")]),
        "locations_geojson": make_geojson_features(["loc1", "loc2", "loc3"]),
    }
    notices = validate_location_id_foreign_key(feed, CTX)
    assert len(notices) == 0
