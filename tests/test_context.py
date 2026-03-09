from __future__ import annotations

import polars as pl

from gtfs_validator.context import build_stop_location_cache


def test_build_stop_location_cache_resolves_parent_within_hops() -> None:
    stops = pl.DataFrame(
        [
            {"stop_id": "P0", "stop_lat": 1.0, "stop_lon": 2.0, "parent_station": None, "stop_name": "P0"},
            {"stop_id": "C1", "stop_lat": None, "stop_lon": None, "parent_station": "P0", "stop_name": "C1"},
        ],
    )
    cache = build_stop_location_cache(stops)
    assert cache.resolved_latlng_by_stop_id["C1"] == (1.0, 2.0)
    assert cache.stop_name_by_stop_id["C1"] == "C1"
    assert "C1" in cache.existing_stop_ids


def test_build_stop_location_cache_honors_parent_hop_limit() -> None:
    stops = pl.DataFrame(
        [
            {"stop_id": "S0", "stop_lat": 1.0, "stop_lon": 2.0, "parent_station": None},
            {"stop_id": "S1", "stop_lat": None, "stop_lon": None, "parent_station": "S0"},
            {"stop_id": "S2", "stop_lat": None, "stop_lon": None, "parent_station": "S1"},
            {"stop_id": "S3", "stop_lat": None, "stop_lon": None, "parent_station": "S2"},
            {"stop_id": "S4", "stop_lat": None, "stop_lon": None, "parent_station": "S3"},
        ],
    )
    cache = build_stop_location_cache(stops)
    assert cache.resolved_latlng_by_stop_id["S2"] == (1.0, 2.0)
    assert cache.resolved_latlng_by_stop_id["S3"] is None


def test_build_stop_location_cache_missing_parent_returns_none() -> None:
    stops = pl.DataFrame(
        [
            {"stop_id": "S1", "stop_lat": None, "stop_lon": None, "parent_station": "MISSING"},
        ],
    )
    cache = build_stop_location_cache(stops)
    assert cache.resolved_latlng_by_stop_id["S1"] is None
