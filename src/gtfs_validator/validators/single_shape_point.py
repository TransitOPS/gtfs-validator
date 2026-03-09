"""Validator: SingleShapePointValidator — warns when a shape has only one point."""

from __future__ import annotations

import polars as pl

from gtfs_validator.context import ValidationContext
from gtfs_validator.notices import Notice, Severity


def validate_single_shape_point(
    feed: dict[str, pl.DataFrame],
    ctx: ValidationContext,
) -> list[Notice]:
    """Emit a warning for each shape_id that appears in exactly one row of shapes.txt."""
    if "shapes" not in feed:
        return []

    shapes = feed["shapes"]
    if shapes.is_empty():
        return []

    counts = (
        shapes
        .group_by("shape_id")
        .agg(
            pl.len().alias("point_count"),
            pl.col("csv_row_number").last().alias("csv_row_number"),
        )
        .filter(pl.col("point_count") == 1)
    )

    notices: list[Notice] = []
    for row in counts.iter_rows(named=True):
        notices.append(
            Notice(
                code="single_shape_point",
                severity=Severity.WARNING,
                fields={
                    "shape_id": row["shape_id"],
                    "csv_row_number": row["csv_row_number"],
                },
            )
        )
    return notices
