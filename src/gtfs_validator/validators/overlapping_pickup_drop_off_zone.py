"""Validator: OverlappingPickupDropOffZoneValidator.

Detects pairs of stop_times.txt entries within the same trip that have
overlapping pickup/drop-off time windows AND overlapping geographic zones
(as defined by their locations.geojson polygons), which would create an
ambiguous service area during that time period.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import polars as pl
from shapely.geometry import shape

from gtfs_validator.notices import Notice, Severity

if TYPE_CHECKING:
    from gtfs_validator.context import ValidationContext

_RECOGNIZED_TYPES: frozenset[int] = frozenset({0, 1, 2, 3})


def _time_to_secs(t: str) -> int:
    """Convert a GTFS HH:MM:SS time string to total seconds since midnight."""
    h, m, s = t.split(":")
    return int(h) * 3600 + int(m) * 60 + int(s)


def _build_geometry_index(geojson_features: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a mapping of feature ID to Shapely geometry."""
    index: dict[str, Any] = {}
    for f in geojson_features:
        fid = f.get("id")
        geom_type = f.get("geometry_type")
        coords = f.get("coordinates")
        if not fid or not geom_type or coords is None:
            continue
        try:
            geom = shape({"type": geom_type, "coordinates": coords})
        except Exception:
            continue
        index[fid] = geom
    return index


def validate_overlapping_pickup_drop_off_zone(
    feed: dict[str, pl.DataFrame],
    ctx: ValidationContext,
) -> list[Notice]:
    """Emit overlapping_zone_and_pickup_drop_off_window (ERROR) for qualifying stop-time pairs."""
    stop_times = feed.get("stop_times")
    geojson_features = feed.get("locations_geojson")

    if stop_times is None or stop_times.is_empty():
        return []
    if not geojson_features:
        return []

    required_cols = {
        "trip_id", "stop_sequence", "location_id",
        "pickup_type", "drop_off_type",
        "start_pickup_drop_off_window", "end_pickup_drop_off_window",
    }
    if not required_cols.issubset(set(stop_times.columns)):
        return []

    candidates = stop_times.filter(
        pl.col("location_id").is_not_null()
        & pl.col("start_pickup_drop_off_window").is_not_null()
        & pl.col("end_pickup_drop_off_window").is_not_null()
    ).select(list(required_cols))

    if candidates.is_empty():
        return []

    geom_index = _build_geometry_index(geojson_features)

    notices: list[Notice] = []

    for (trip_id,), group in candidates.group_by("trip_id"):
        rows = group.to_dicts()
        if len(rows) < 2:
            continue

        for i in range(len(rows)):
            st1 = rows[i]
            for j in range(i + 1, len(rows)):
                st2 = rows[j]

                pt1 = st1["pickup_type"] if st1["pickup_type"] is not None else -1
                dt1 = st1["drop_off_type"] if st1["drop_off_type"] is not None else -1
                pt2 = st2["pickup_type"] if st2["pickup_type"] is not None else -1
                dt2 = st2["drop_off_type"] if st2["drop_off_type"] is not None else -1

                if (pt1 not in _RECOGNIZED_TYPES or dt1 not in _RECOGNIZED_TYPES or
                        pt2 not in _RECOGNIZED_TYPES or dt2 not in _RECOGNIZED_TYPES):
                    continue
                if pt1 != pt2 and dt1 != dt2:
                    continue

                loc1: str = st1["location_id"]
                loc2: str = st2["location_id"]
                if loc1 == loc2:
                    continue

                s1 = _time_to_secs(st1["start_pickup_drop_off_window"])
                e1 = _time_to_secs(st1["end_pickup_drop_off_window"])
                s2 = _time_to_secs(st2["start_pickup_drop_off_window"])
                e2 = _time_to_secs(st2["end_pickup_drop_off_window"])
                if not (s1 < e2 and s2 < e1):
                    continue

                geom1 = geom_index.get(loc1)
                geom2 = geom_index.get(loc2)
                if geom1 is None or geom2 is None:
                    continue

                if not geom1.overlaps(geom2):
                    continue

                notices.append(
                    Notice(
                        code="overlapping_zone_and_pickup_drop_off_window",
                        severity=Severity.ERROR,
                        fields={
                            "trip_id": trip_id,
                            "stop_sequence1": st1["stop_sequence"],
                            "location_id1": loc1,
                            "start_pickup_drop_off_window1": st1["start_pickup_drop_off_window"],
                            "end_pickup_drop_off_window1": st1["end_pickup_drop_off_window"],
                            "stop_sequence2": st2["stop_sequence"],
                            "location_id2": loc2,
                            "start_pickup_drop_off_window2": st2["start_pickup_drop_off_window"],
                            "end_pickup_drop_off_window2": st2["end_pickup_drop_off_window"],
                        },
                    )
                )

    return notices
