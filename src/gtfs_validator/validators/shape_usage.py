"""Validator: warn when shapes.txt contains shapes not referenced in trips.txt."""

from __future__ import annotations

import polars as pl

from gtfs_validator.context import ValidationContext
from gtfs_validator.notices import Notice, Severity


def validate_shape_usage(
    feed: dict[str, pl.DataFrame],
    ctx: ValidationContext,
) -> list[Notice]:
    """Warn when a shape in shapes.txt has no corresponding trip in trips.txt."""
    if "shapes" not in feed or feed["shapes"].is_empty():
        return []

    # Build deduplicated set of shape_ids referenced by trips
    if "trips" in feed and not feed["trips"].is_empty():
        trips_shape_ids_df = (
            feed["trips"]
            .select("shape_id")
            .filter(
                pl.col("shape_id").is_not_null() & (pl.col("shape_id") != "")
            )
            .unique()
        )
    else:
        trips_shape_ids_df = pl.DataFrame({"shape_id": pl.Series([], dtype=pl.String)})

    # Compute first occurrence of each shape_id in file order
    first_occurrences = (
        feed["shapes"]
        .with_row_index("_row_idx")
        .group_by("shape_id")
        .agg(
            pl.col("csv_row_number").first(),
            pl.col("_row_idx").min(),
        )
        .sort("_row_idx")
    )

    # Anti-join to find shapes not referenced by any trip
    unused = first_occurrences.join(trips_shape_ids_df, on="shape_id", how="anti")

    notices: list[Notice] = []
    for row in unused.iter_rows(named=True):
        notices.append(
            Notice(
                code="unused_shape",
                severity=Severity.WARNING,
                fields={
                    "shape_id": row["shape_id"],
                    "csv_row_number": row["csv_row_number"],
                },
            )
        )

    return notices
