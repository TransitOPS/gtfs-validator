"""RouteColorContrastValidator: warn when route color and text color lack contrast."""

from __future__ import annotations

import polars as pl

from gtfs_validator.context import ValidationContext
from gtfs_validator.notices import Notice, Severity

MAX_ROUTE_COLOR_LUMA_DIFFERENCE = 72


def _hex_luma_expr(col_name: str) -> pl.Expr:
    """Compute Rec 601 luma [0, 255] from a 6-digit hex color column."""
    v = pl.col(col_name).str.to_integer(base=16)
    r = (v // 65536) % 256
    g = (v // 256) % 256
    b = v % 256
    return (
        r.cast(pl.Float64) * 0.30
        + g.cast(pl.Float64) * 0.59
        + b.cast(pl.Float64) * 0.11
    ).cast(pl.Int32)


def validate_route_color_contrast(
    feed: dict[str, pl.DataFrame],
    ctx: ValidationContext,
) -> list[Notice]:
    """Emit a warning for each route where color and text color lack sufficient contrast."""
    if "routes" not in feed or feed["routes"].is_empty():
        return []

    routes = feed["routes"]

    both_present = routes.filter(
        pl.col("route_color").is_not_null()
        & pl.col("route_text_color").is_not_null()
    )
    if both_present.is_empty():
        return []

    flagged = both_present.with_columns(
        [
            _hex_luma_expr("route_color").alias("_luma_color"),
            _hex_luma_expr("route_text_color").alias("_luma_text"),
        ]
    ).filter(
        (pl.col("_luma_color") - pl.col("_luma_text")).abs()
        < MAX_ROUTE_COLOR_LUMA_DIFFERENCE
    )

    notices = []
    for row in flagged.iter_rows(named=True):
        notices.append(
            Notice(
                code="route_color_contrast",
                severity=Severity.WARNING,
                fields={
                    "route_id": row["route_id"],
                    "csv_row_number": row["csv_row_number"],
                    "route_color": row["route_color"],
                    "route_text_color": row["route_text_color"],
                },
            )
        )
    return notices
