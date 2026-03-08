"""Validator: detect duplicate fare media entries by (name, type) pair."""

from __future__ import annotations

import polars as pl

from gtfs_validator.context import ValidationContext
from gtfs_validator.notices import Notice, Severity


def validate_duplicate_fare_media(
    feed: dict[str, pl.DataFrame],
    ctx: ValidationContext,
) -> list[Notice]:
    """Emit a WARNING for each duplicate (fare_media_name, fare_media_type) pair."""
    if "fare_media" not in feed or feed["fare_media"].is_empty():
        return []

    df = feed["fare_media"]

    # Add 1-based csv_row_number
    df = df.with_row_index("_idx").with_columns(
        (pl.col("_idx") + 1).alias("csv_row_number")
    )

    # Fill null fare_media_name with empty string so null-null matches
    df = df.with_columns(
        pl.col("fare_media_name").fill_null("").alias("fare_media_name")
    )

    # Sort to ensure deterministic intra-group order before group_by
    df = df.sort("csv_row_number")

    # Group by composite key and filter to groups with duplicates
    groups = (
        df.group_by("fare_media_name", "fare_media_type")
        .agg(
            pl.col("csv_row_number"),
            pl.col("fare_media_id"),
        )
        .filter(pl.col("csv_row_number").list.len() > 1)
    )

    # Emit notices: pair first row with each subsequent row
    notices: list[Notice] = []
    for row in groups.iter_rows(named=True):
        row_numbers = row["csv_row_number"]
        media_ids = row["fare_media_id"]

        first_row_number = row_numbers[0]
        first_media_id = media_ids[0]

        for i in range(1, len(row_numbers)):
            notices.append(
                Notice(
                    code="duplicate_fare_media",
                    severity=Severity.WARNING,
                    fields={
                        "csv_row_number_1": first_row_number,
                        "fare_media_id_1": first_media_id,
                        "csv_row_number_2": row_numbers[i],
                        "fare_media_id_2": media_ids[i],
                    },
                )
            )

    return notices
