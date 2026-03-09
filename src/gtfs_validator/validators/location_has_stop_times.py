"""Validator: stops must have stop_times; non-stops must not."""

from __future__ import annotations

import polars as pl

from gtfs_validator.context import ValidationContext
from gtfs_validator.notices import Notice, Severity


def validate_location_has_stop_times(
    feed: dict[str, pl.DataFrame],
    ctx: ValidationContext,
) -> list[Notice]:
    stops = feed.get("stops")
    stop_times = feed.get("stop_times")
    if stops is None or stop_times is None:
        return []
    if stops.is_empty():
        return []

    location_group_stops = feed.get("location_group_stops")

    # Step 1: Build the set of stop_ids referenced in stop_times
    referenced_stop_ids: set[str] = set()
    if "stop_id" in stop_times.columns:
        referenced_stop_ids = set(
            stop_times.select("stop_id").drop_nulls().to_series().to_list()
        )

    if (
        "location_group_id" in stop_times.columns
        and location_group_stops is not None
        and not location_group_stops.is_empty()
    ):
        lg_ids = set(
            stop_times.select("location_group_id").drop_nulls().to_series().to_list()
        )
        if lg_ids:
            lg_stop_ids = (
                location_group_stops
                .filter(pl.col("location_group_id").is_in(list(lg_ids)))
                .select("stop_id")
                .drop_nulls()
                .to_series()
                .to_list()
            )
            referenced_stop_ids.update(lg_stop_ids)

    notices: list[Notice] = []

    # Step 2a: Stops (location_type=0 or null) without any stop time reference
    actual_stops = stops.filter(
        (pl.col("location_type") == 0) | pl.col("location_type").is_null()
    )
    for row in actual_stops.iter_rows(named=True):
        if row["stop_id"] not in referenced_stop_ids:
            notices.append(Notice(
                code="stop_without_stop_time",
                severity=Severity.WARNING,
                fields={
                    "csv_row_number": row["csv_row_number"],
                    "stop_id": row["stop_id"],
                    "stop_name": row.get("stop_name", "") or "",
                },
            ))

    # Step 2b: Non-stop locations with direct stop time references
    non_stops = stops.filter(
        pl.col("location_type").is_not_null() & (pl.col("location_type") != 0)
    )
    if not non_stops.is_empty() and not stop_times.is_empty() and "stop_id" in stop_times.columns:
        bad = non_stops.join(
            stop_times.select("stop_id", "csv_row_number").rename(
                {"csv_row_number": "stop_time_csv_row_number"}
            ),
            on="stop_id",
            how="inner",
        )
        if not bad.is_empty():
            first_bad = bad.group_by("stop_id").agg([
                pl.col("csv_row_number").first(),
                pl.col("stop_name").first(),
                pl.col("stop_time_csv_row_number").min(),
            ])
            for row in first_bad.iter_rows(named=True):
                notices.append(Notice(
                    code="location_with_unexpected_stop_time",
                    severity=Severity.ERROR,
                    fields={
                        "csv_row_number": row["csv_row_number"],
                        "stop_id": row["stop_id"],
                        "stop_name": row["stop_name"],
                        "stop_time_csv_row_number": row["stop_time_csv_row_number"],
                    },
                ))

    return notices
