"""Bike allowance validator for ferry routes."""

from __future__ import annotations

import polars as pl

from gtfs_validator.context import ValidationContext
from gtfs_validator.notices import Notice, Severity


def validate_bikes_allowance(
    feed: dict[str, pl.DataFrame],
    ctx: ValidationContext,
) -> list[Notice]:
    """Warn when a ferry trip lacks explicit bikes_allowed information."""
    # 1. Guard: both tables must exist
    if "routes" not in feed or "trips" not in feed:
        return []

    routes = feed["routes"]
    trips = feed["trips"]

    # 2. Guard: empty tables
    if routes.is_empty() or trips.is_empty():
        return []

    # 3. Find ferry route IDs (route_type == 4)
    ferry_route_ids = routes.filter(
        pl.col("route_type") == 4
    ).select("route_id")

    # 4. Guard: no ferry routes
    if ferry_route_ids.is_empty():
        return []

    # 5. Semi-join trips against ferry route IDs
    ferry_trips = trips.join(ferry_route_ids, on="route_id", how="semi")

    # 6. Filter for violations: bikes_allowed NOT IN {1, 2}
    #    Explicit is_null() OR is needed because Polars is_in returns null
    #    for null inputs, and ~null is still null which filter drops.
    violations = ferry_trips.filter(
        ~pl.col("bikes_allowed").is_in([1, 2]) | pl.col("bikes_allowed").is_null()
    )

    # 7. Emit notices
    notices: list[Notice] = []
    for row in violations.iter_rows(named=True):
        notices.append(
            Notice(
                code="missing_bike_allowance",
                severity=Severity.WARNING,
                fields={
                    "csv_row_number": row["csv_row_number"],
                    "route_id": row["route_id"],
                    "trip_id": row["trip_id"],
                },
            )
        )

    return notices
