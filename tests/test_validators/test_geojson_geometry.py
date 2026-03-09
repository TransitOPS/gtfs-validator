"""Tests for GeoJSON geometry validator."""

from datetime import date

import pytest

from gtfs_validator.context import ValidationContext
from gtfs_validator.notices import Severity
from gtfs_validator.validators.geojson_geometry import validate_geojson_geometry

CTX = ValidationContext(country_code="US", date_for_validation=date(2026, 3, 8))


def make_feature(
    feature_id: str,
    index: int,
    geometry_type: str,
    coordinates: list,
) -> dict:
    return {
        "id": feature_id,
        "index": index,
        "geometry_type": geometry_type,
        "coordinates": coordinates,
    }


def make_feed(features: list[dict]) -> dict:
    return {"locations_geojson": features}


# Reusable coordinate sets
VALID_SQUARE = [[(0, 0), (0, 1), (1, 1), (1, 0), (0, 0)]]
VALID_SQUARE_2 = [[(5, 5), (5, 6), (6, 6), (6, 5), (5, 5)]]
DEGENERATE_POLYGON = [[(0, 0), (0, 1), (0, 0)]]  # collinear, only 3 points
TRIANGLE = [[(0, 0), (0, 1), (1, 0), (0, 0)]]  # exactly 4 points (valid)
BOWTIE = [[(0, 0), (2, 2), (2, 0), (0, 2), (0, 0)]]  # self-intersecting


def test_valid_polygon_no_notices():
    feed = make_feed([make_feature("f1", 0, "Polygon", VALID_SQUARE)])
    notices = validate_geojson_geometry(feed, CTX)
    assert notices == []


def test_invalid_polygon_degenerate_emits_notice():
    feed = make_feed([make_feature("f1", 0, "Polygon", DEGENERATE_POLYGON)])
    notices = validate_geojson_geometry(feed, CTX)
    assert len(notices) == 1
    assert notices[0].code == "invalid_geometry"
    assert notices[0].severity == Severity.ERROR


def test_valid_multi_polygon_no_notices():
    feed = make_feed(
        [make_feature("f1", 0, "MultiPolygon", [VALID_SQUARE, VALID_SQUARE_2])]
    )
    notices = validate_geojson_geometry(feed, CTX)
    assert notices == []


def test_invalid_multi_polygon_overlapping_emits_notice():
    feed = make_feed(
        [make_feature("f1", 0, "MultiPolygon", [VALID_SQUARE, VALID_SQUARE])]
    )
    notices = validate_geojson_geometry(feed, CTX)
    assert len(notices) == 1
    assert notices[0].code == "invalid_geometry"
    assert notices[0].severity == Severity.ERROR
    assert notices[0].fields["geometry_type"] == "MultiPolygon"


def test_polygon_with_valid_hole_no_notices():
    coords = [
        [(0, 0), (0, 10), (10, 10), (10, 0), (0, 0)],
        [(2, 2), (2, 4), (4, 4), (4, 2), (2, 2)],
    ]
    feed = make_feed([make_feature("f1", 0, "Polygon", coords)])
    notices = validate_geojson_geometry(feed, CTX)
    assert notices == []


def test_polygon_with_hole_outside_shell_emits_notice():
    coords = [
        [(0, 0), (0, 1), (1, 1), (1, 0), (0, 0)],
        [(5, 5), (5, 6), (6, 6), (6, 5), (5, 5)],
    ]
    feed = make_feed([make_feature("f1", 0, "Polygon", coords)])
    notices = validate_geojson_geometry(feed, CTX)
    assert len(notices) == 1
    assert notices[0].code == "invalid_geometry"


def test_self_intersecting_bowtie_polygon_emits_notice():
    feed = make_feed([make_feature("f1", 0, "Polygon", BOWTIE)])
    notices = validate_geojson_geometry(feed, CTX)
    assert len(notices) == 1
    assert notices[0].code == "invalid_geometry"
    assert "Self-intersection" in notices[0].fields["message"]


def test_multi_polygon_first_invalid_short_circuits():
    feed = make_feed(
        [make_feature("f1", 0, "MultiPolygon", [DEGENERATE_POLYGON, VALID_SQUARE])]
    )
    notices = validate_geojson_geometry(feed, CTX)
    assert len(notices) == 1
    assert notices[0].code == "invalid_geometry"


def test_valid_triangle_polygon_no_notices():
    feed = make_feed([make_feature("f1", 0, "Polygon", TRIANGLE)])
    notices = validate_geojson_geometry(feed, CTX)
    assert notices == []


def test_locations_geojson_absent_no_notices():
    notices = validate_geojson_geometry({}, CTX)
    assert notices == []


def test_locations_geojson_empty_features_no_notices():
    feed = {"locations_geojson": []}
    notices = validate_geojson_geometry(feed, CTX)
    assert notices == []


def test_unsupported_geometry_type_skipped():
    feed = make_feed([make_feature("f1", 0, "Point", [[0, 0]])])
    notices = validate_geojson_geometry(feed, CTX)
    assert notices == []


def test_notice_fields_are_complete():
    feed = make_feed([make_feature("f1", 0, "Polygon", DEGENERATE_POLYGON)])
    notices = validate_geojson_geometry(feed, CTX)
    assert len(notices) == 1
    fields = notices[0].fields
    assert set(fields.keys()) == {"feature_id", "feature_index", "geometry_type", "message"}


def test_multiple_features_mixed_valid_invalid():
    features = [
        make_feature("f1", 0, "Polygon", VALID_SQUARE),
        make_feature("f2", 1, "Polygon", BOWTIE),
    ]
    feed = make_feed(features)
    notices = validate_geojson_geometry(feed, CTX)
    assert len(notices) == 1
    assert notices[0].code == "invalid_geometry"
    assert notices[0].fields["feature_index"] == 1
