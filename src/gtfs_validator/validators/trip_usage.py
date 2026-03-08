"""Validator: warn when trips.txt contains trips not referenced in stop_times.txt."""

from __future__ import annotations

import polars as pl

from gtfs_validator.context import ValidationContext
from gtfs_validator.notices import Notice, Severity


def validate_trip_usage(
    feed: dict[str, pl.DataFrame],
    ctx: ValidationContext,
) -> list[Notice]:
    """Warn when a trip in trips.txt has no corresponding records in stop_times.txt."""
    if "trips" not in feed or feed["trips"].is_empty():
        return []
    if "stop_times" not in feed:
        return []

    trips_df = feed["trips"]

    # Get set of used trip_ids from stop_times
    if not feed["stop_times"].is_empty():
        used_trip_ids = feed["stop_times"]["trip_id"].unique()
    else:
        # Empty stop_times means all trips are unused.
        used_trip_ids = pl.Series("trip_id", [], dtype=pl.String)

    # Find unused trips (trip_id not in stop_times)
    unused_trips = trips_df.filter(
        ~pl.col("trip_id").is_in(used_trip_ids.implode())
    ).unique(subset=["trip_id"], maintain_order=True)

    notices: list[Notice] = []
    for row in unused_trips.iter_rows(named=True):
        notices.append(
            Notice(
                code="unused_trip",
                severity=Severity.WARNING,
                fields={
                    "trip_id": row["trip_id"],
                    "csv_row_number": row["csv_row_number"],
                },
            )
        )

    return notices
