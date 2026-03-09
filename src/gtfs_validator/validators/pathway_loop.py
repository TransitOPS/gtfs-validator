"""Validator: pathway_loop."""

from __future__ import annotations

import polars as pl

from gtfs_validator.context import ValidationContext
from gtfs_validator.notices import Notice, Severity


def validate_pathway_loop(
    feed: dict[str, pl.DataFrame],
    ctx: ValidationContext,
) -> list[Notice]:
    """Warning when a pathway's from_stop_id equals its to_stop_id (loop)."""
    if "pathways" not in feed:
        return []

    pathways = feed["pathways"]

    if pathways.is_empty():
        return []

    violations = pathways.filter(
        pl.col("from_stop_id").is_not_null()
        & pl.col("to_stop_id").is_not_null()
        & (pl.col("from_stop_id") == pl.col("to_stop_id"))
    )

    notices: list[Notice] = []
    for row in violations.select(["csv_row_number", "pathway_id", "from_stop_id"]).iter_rows(named=True):
        notices.append(
            Notice(
                code="pathway_loop",
                severity=Severity.WARNING,
                fields={
                    "csvRowNumber": row["csv_row_number"],
                    "pathwayId": row["pathway_id"],
                    "stopId": row["from_stop_id"],
                },
            )
        )
    return notices
