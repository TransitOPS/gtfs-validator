"""Validator: TripAndShapeDistanceValidator.

Checks that the last stop time's shape_dist_traveled does not exceed the
maximum shape point's shape_dist_traveled for the associated shape.
"""

from __future__ import annotations

import math

import polars as pl

from gtfs_validator.context import ValidationContext
from gtfs_validator.notices import Notice, Severity


DISTANCE_THRESHOLD_METERS: float = 11.1
EARTH_RADIUS_METERS: float = 6_371_010.0


def _haversine_meters(
    lat1: float, lon1: float, lat2: float, lon2: float
) -> float:
    """Great-circle distance in meters using Earth radius 6,371,010 m.

    Matches S2Earth.getDistanceMeters used by the Java implementation.
    """
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2.0 * math.asin(math.sqrt(a)) * EARTH_RADIUS_METERS


def validate_trip_and_shape_distance(
    feed: dict[str, pl.DataFrame],
    ctx: ValidationContext,
) -> list[Notice]:
    """Validate that trip distances do not exceed shape distances."""
    # Guard clauses: return [] if any required table is absent or empty
    for tbl in ("trips", "stop_times", "stops", "shapes"):
        if tbl not in feed or feed[tbl].is_empty():
            return []

    # shape_dist_traveled is optional in stop_times; skip if not present
    if "shape_dist_traveled" not in feed["stop_times"].columns:
        return []

    # Step 1 — Last stop time per trip
    last_st = (
        feed["stop_times"]
        .select(["trip_id", "stop_id", "stop_sequence", "shape_dist_traveled"])
        .sort("stop_sequence")
        .group_by("trip_id")
        .last()
        # Replace null shape_dist_traveled with 0.0 (matches Java default getter)
        .with_columns(
            pl.col("shape_dist_traveled").fill_null(0.0)
        )
    )

    # Step 2 — Last shape point per shape_id (by max shape_dist_traveled)
    max_shape = (
        feed["shapes"]
        .select(["shape_id", "shape_pt_lat", "shape_pt_lon", "shape_dist_traveled"])
        .with_columns(
            pl.col("shape_dist_traveled").fill_null(0.0)
        )
        .sort("shape_dist_traveled", descending=True)
        .group_by("shape_id")
        .first()
        .rename({
            "shape_dist_traveled": "max_shape_dist",
            "shape_pt_lat": "end_shape_lat",
            "shape_pt_lon": "end_shape_lon",
        })
    )

    # Step 3 — Joins
    # trips → last stop time (inner join; drops trips without stop times)
    # Cast shape_id to String to handle feeds where the column is all-null (Null dtype)
    trips_slim = (
        feed["trips"]
        .select(["trip_id", "shape_id"])
        .with_columns(pl.col("shape_id").cast(pl.String))
    )
    joined = trips_slim.join(last_st, on="trip_id", how="inner")

    # → stops (inner join on stop_id; drops rows where stop not found)
    stops_slim = feed["stops"].select(["stop_id", "stop_lat", "stop_lon"])
    joined = joined.join(stops_slim, on="stop_id", how="inner")

    # → max shape point (inner join on shape_id; drops trips with no shape or
    #   shape_id absent from shapes.txt, including null shape_id)
    joined = joined.join(max_shape, on="shape_id", how="inner")

    # Step 4 — Numeric filter
    violations = joined.filter(
        (pl.col("max_shape_dist") > 0.0)
        & (pl.col("shape_dist_traveled") > pl.col("max_shape_dist"))
    )

    # Step 5 — Geographic distance and notice emission
    notices: list[Notice] = []
    for row in violations.iter_rows(named=True):
        geo_dist = _haversine_meters(
            row["end_shape_lat"], row["end_shape_lon"],
            row["stop_lat"], row["stop_lon"],
        )
        fields = {
            "trip_id": row["trip_id"],
            "shape_id": row["shape_id"],
            "max_trip_distance_traveled": row["shape_dist_traveled"],
            "max_shape_distance_traveled": row["max_shape_dist"],
            "geo_distance_to_shape": geo_dist,
        }
        if geo_dist > DISTANCE_THRESHOLD_METERS:
            notices.append(Notice(
                code="trip_distance_exceeds_shape_distance",
                severity=Severity.ERROR,
                fields=fields,
            ))
        else:
            notices.append(Notice(
                code="trip_distance_exceeds_shape_distance_below_threshold",
                severity=Severity.WARNING,
                fields=fields,
            ))
    return notices
