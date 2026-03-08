"""Tests for feature detection."""

from __future__ import annotations

import polars as pl

from gtfs_validator.features import detect_features


def test_file_based_detection():
    feed = {
        "shapes.txt": pl.DataFrame({
            "shape_id": ["SH1"],
            "shape_pt_lat": [40.0],
            "shape_pt_lon": [-74.0],
            "shape_pt_sequence": [1],
        }),
        "pathways.txt": pl.DataFrame({"pathway_id": pl.Series([], dtype=pl.Utf8)}),
    }
    features = detect_features(feed)
    assert "Shapes" in features
    assert "Pathway Connections" not in features  # empty table


def test_field_based_detection():
    feed = {
        "routes.txt": pl.DataFrame({
            "route_id": ["R1"],
            "route_color": ["FF0000"],
        }),
        "trips.txt": pl.DataFrame({
            "trip_id": ["T1"],
            "trip_headsign": [None],
            "wheelchair_accessible": ["1"],
        }),
    }
    features = detect_features(feed)
    assert "Route Colors" in features
    assert "Trips Wheelchair Accessibility" in features
    assert "Headsigns" not in features  # all null


def test_no_features_empty_feed():
    feed: dict[str, pl.DataFrame] = {}
    features = detect_features(feed)
    assert features == []
