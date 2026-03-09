"""Validator: detect duplicate route names within the same agency."""

from __future__ import annotations

import polars as pl

from gtfs_validator.context import ValidationContext
from gtfs_validator.notices import Notice, Severity


def validate_duplicate_route_name(
    feed: dict[str, pl.DataFrame],
    ctx: ValidationContext,
) -> list[Notice]:
    """Flag routes sharing (short_name, long_name, route_type, agency_id)."""
    if "routes" not in feed or feed["routes"].is_empty():
        return []

    df = feed["routes"]

    # Add 1-based csv_row_number (header is row 1, data starts at row 2).
    df = df.with_row_index("_idx").with_columns(
        (pl.col("_idx") + 2).alias("csv_row_number")
    )

    # Fill null string fields with empty string to match Java default-getter
    # semantics (null -> "").
    df = df.with_columns(
        pl.col("route_short_name").fill_null("").alias("route_short_name"),
        pl.col("route_long_name").fill_null("").alias("route_long_name"),
        pl.col("agency_id").fill_null("").alias("agency_id"),
    )

    # Group by composite key and collect row numbers and IDs as lists.
    # Filter to groups with more than one row (duplicates).
    groups = (
        df.sort("csv_row_number")
        .group_by("route_short_name", "route_long_name", "route_type", "agency_id")
        .agg(
            pl.col("csv_row_number"),
            pl.col("route_id"),
        )
        .filter(pl.col("csv_row_number").list.len() > 1)
    )

    # Emit notices: pair first row with each subsequent row.
    notices: list[Notice] = []
    for row in groups.iter_rows(named=True):
        row_numbers = row["csv_row_number"]
        route_ids = row["route_id"]
        short_name = row["route_short_name"]
        long_name = row["route_long_name"]
        route_type = row["route_type"]
        agency_id = row["agency_id"]

        first_row_number = row_numbers[0]
        first_route_id = route_ids[0]

        for i in range(1, len(row_numbers)):
            notices.append(
                Notice(
                    code="duplicate_route_name",
                    severity=Severity.WARNING,
                    fields={
                        "csv_row_number_1": first_row_number,
                        "route_id_1": first_route_id,
                        "csv_row_number_2": row_numbers[i],
                        "route_id_2": route_ids[i],
                        "route_short_name": short_name,
                        "route_long_name": long_name,
                        "route_type_value": route_type,
                        "agency_id": agency_id,
                    },
                )
            )

    return notices
