"""Validator: warn when a trip has fewer than two stop time entries."""

from __future__ import annotations

import polars as pl

from gtfs_validator.context import ValidationContext
from gtfs_validator.notices import Notice, Severity


def validate_trip_usability(
    feed: dict[str, pl.DataFrame],
    ctx: ValidationContext,
) -> list[Notice]:
    """Emit unusable_trip WARNING for every trip with fewer than 2 stop times."""
    if "trips" not in feed or "stop_times" not in feed:
        return []
    trips = feed["trips"]
    stop_times = feed["stop_times"]
    if trips.is_empty():
        return []

    counts = (
        stop_times
        .group_by("trip_id")
        .agg(pl.len().alias("stop_time_count"))
    )
    joined = trips.join(counts, on="trip_id", how="left")
    joined = joined.with_columns(pl.col("stop_time_count").fill_null(0))
    violating = joined.filter(pl.col("stop_time_count") <= 1)

    notices: list[Notice] = []
    for row in violating.iter_rows(named=True):
        notices.append(
            Notice(
                code="unusable_trip",
                severity=Severity.WARNING,
                fields={
                    "csv_row_number": row.get("csv_row_number"),
                    "trip_id": row["trip_id"],
                },
            )
        )
    return notices
