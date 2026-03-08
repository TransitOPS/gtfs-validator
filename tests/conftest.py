"""Shared test fixtures."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import polars as pl
import pytest


@pytest.fixture
def tmp_output(tmp_path: Path) -> Path:
    """Temporary output directory."""
    out = tmp_path / "output"
    out.mkdir()
    return out


@pytest.fixture
def small_feed_dir(tmp_path: Path) -> Path:
    """Create a minimal valid GTFS feed directory."""
    feed = tmp_path / "feed"
    feed.mkdir()

    (feed / "agency.txt").write_text(
        "agency_id,agency_name,agency_url,agency_timezone\n"
        "A1,Test Agency,http://example.com,America/New_York\n"
    )
    (feed / "stops.txt").write_text(
        "stop_id,stop_name,stop_lat,stop_lon\n"
        "S1,Stop One,40.7128,-74.0060\n"
        "S2,Stop Two,40.7580,-73.9855\n"
    )
    (feed / "routes.txt").write_text(
        "route_id,agency_id,route_short_name,route_type\n"
        "R1,A1,Route 1,3\n"
    )
    (feed / "trips.txt").write_text(
        "trip_id,route_id,service_id\n"
        "T1,R1,SVC1\n"
    )
    (feed / "stop_times.txt").write_text(
        "trip_id,arrival_time,departure_time,stop_id,stop_sequence\n"
        "T1,8:00:00,8:00:00,S1,1\n"
        "T1,8:10:00,8:10:00,S2,2\n"
    )
    (feed / "calendar.txt").write_text(
        "service_id,monday,tuesday,wednesday,thursday,friday,saturday,sunday,start_date,end_date\n"
        "SVC1,1,1,1,1,1,0,0,20240101,20241231\n"
    )
    return feed


@pytest.fixture
def small_feed_zip(small_feed_dir: Path, tmp_path: Path) -> Path:
    """Create a minimal GTFS ZIP from the feed directory."""
    zip_path = tmp_path / "feed.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for f in small_feed_dir.iterdir():
            zf.write(f, f.name)
    return zip_path
