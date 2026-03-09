"""Validator: block trips with overlapping stop times."""

from __future__ import annotations

import polars as pl

from gtfs_validator.calendar_utils import (
    ServiceIdIntersectionCache,
    build_service_date_map,
)
from gtfs_validator.context import ValidationContext
from gtfs_validator.notices import Notice, Severity


def validate_block_trips_overlapping(
    feed: dict[str, pl.DataFrame],
    ctx: ValidationContext,
) -> list[Notice]:
    """Detect trips sharing a block_id whose stop-time intervals overlap.

    Emits ``block_trips_with_overlapping_stop_times`` (ERROR) when two
    trips in the same block have overlapping time intervals and share at
    least one active service date.
    """
    # 1. Guard: required tables
    if "trips" not in feed or "stop_times" not in feed:
        return []
    trips = feed["trips"]
    stop_times = feed["stop_times"]
    if trips.is_empty() or stop_times.is_empty():
        return []

    # 2. Build service date intersection cache
    service_dates = build_service_date_map(feed)
    cache = ServiceIdIntersectionCache(service_dates)

    # 3. Filter trips with non-null, non-empty block_id
    blocked_trips = trips.filter(
        pl.col("block_id").is_not_null() & (pl.col("block_id") != "")
    )
    if blocked_trips.is_empty():
        return []

    # 4. Build trip time intervals using Polars
    intervals = (
        stop_times
        .sort("trip_id", "stop_sequence")
        .group_by("trip_id")
        .agg([
            pl.col("arrival_time").first().alias("first_arrival"),
            pl.col("departure_time").first().alias("first_departure"),
            pl.col("arrival_time").last().alias("last_arrival"),
            pl.col("departure_time").last().alias("last_departure"),
        ])
    )

    # 5. Join intervals onto blocked trips
    trip_data = blocked_trips.select(
        "trip_id", "block_id", "service_id",
    ).join(intervals, on="trip_id", how="inner")

    # 6. Filter out trips where any time bound is null
    trip_data = trip_data.filter(
        pl.col("first_arrival").is_not_null()
        & pl.col("first_departure").is_not_null()
        & pl.col("last_arrival").is_not_null()
        & pl.col("last_departure").is_not_null()
    )

    if trip_data.is_empty():
        return []

    # 7. Group by block_id and run pairwise overlap detection
    notices: list[Notice] = []

    for block_id, group in trip_data.group_by("block_id"):
        block_id_str = block_id[0]  # group_by returns tuple keys

        # Sort by (first_arrival, last_departure) ascending
        sorted_group = group.sort("first_arrival", "last_departure")
        rows = sorted_group.to_dicts()

        # Pairwise comparison with early termination
        n = len(rows)
        for i in range(n):
            for j in range(i + 1, n):
                ri = rows[i]
                rj = rows[j]

                # Break: no further j can overlap with i
                if ri["last_departure"] <= rj["first_arrival"]:
                    break

                # Block transfer exception
                if (
                    ri["last_arrival"] == rj["first_arrival"]
                    and ri["last_departure"] == rj["first_departure"]
                ):
                    continue

                # Service date intersection check
                intersect_date = cache.first_intersecting_date(
                    ri["service_id"], rj["service_id"],
                )
                if intersect_date is None:
                    continue

                notices.append(Notice(
                    code="block_trips_with_overlapping_stop_times",
                    severity=Severity.ERROR,
                    fields={
                        "trip_id_a": ri["trip_id"],
                        "service_id_a": ri["service_id"],
                        "trip_id_b": rj["trip_id"],
                        "service_id_b": rj["service_id"],
                        "block_id": block_id_str,
                        "intersection": intersect_date.isoformat(),
                    },
                ))

    return notices
