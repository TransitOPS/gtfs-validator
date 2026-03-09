"""Pathway validators."""

from __future__ import annotations

import polars as pl

from gtfs_validator.context import ValidationContext
from gtfs_validator.notices import Notice, Severity


def validate_bidirectional_exit_gate(
    feed: dict[str, pl.DataFrame],
    ctx: ValidationContext,
) -> list[Notice]:
    """Error when a pathway with mode EXIT_GATE (7) is marked bidirectional."""
    if "pathways" not in feed:
        return []

    df = feed["pathways"]

    if df.is_empty():
        return []

    violations = df.filter(
        (pl.col("pathway_mode") == 7) & (pl.col("is_bidirectional") == 1)
    )

    notices: list[Notice] = []
    for row in violations.iter_rows(named=True):
        notices.append(
            Notice(
                code="bidirectional_exit_gate",
                severity=Severity.ERROR,
                fields={
                    "csv_row_number": row["csv_row_number"],
                    "pathway_mode": row["pathway_mode"],
                    "is_bidirectional": row["is_bidirectional"],
                },
            )
        )

    return notices
