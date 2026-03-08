"""Attribution-level validators."""

from __future__ import annotations

import polars as pl

from gtfs_validator.context import ValidationContext
from gtfs_validator.notices import Notice, Severity


def validate_attribution_without_role(
    feed: dict[str, pl.DataFrame],
    ctx: ValidationContext,
) -> list[Notice]:
    """Warn when an attribution row has none of is_producer/is_operator/is_authority set to 1."""
    if "attributions" not in feed:
        return []

    df = feed["attributions"]

    ROLE_COLS = ["is_producer", "is_operator", "is_authority"]
    present_cols = [c for c in ROLE_COLS if c in df.columns]

    if not present_cols:
        return []

    if df.is_empty():
        return []

    # A row "has a role" if ANY present role column == 1.
    has_role_expr = pl.lit(False)
    for col in present_cols:
        has_role_expr = has_role_expr | (pl.col(col).fill_null(0) == 1)

    no_role_df = df.filter(~has_role_expr)

    notices: list[Notice] = []
    for row in no_role_df.iter_rows(named=True):
        attribution_id = row.get("attribution_id") or ""
        csv_row_number = row["csv_row_number"]
        notices.append(
            Notice(
                code="attribution_without_role",
                severity=Severity.WARNING,
                fields={
                    "csv_row_number": csv_row_number,
                    "attribution_id": str(attribution_id),
                },
            )
        )

    return notices
