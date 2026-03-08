"""Validator: PickupDropOffTypeValidator.

Detects stop_times rows where pickup_type is 0 (REGULAR) or 3 (ON_REQUEST_TO_DRIVER),
or drop_off_type is 0 (REGULAR), when a pickup/drop-off time window is set.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from gtfs_validator.notices import Notice, Severity

if TYPE_CHECKING:
    from gtfs_validator.context import ValidationContext


def validate_pickup_drop_off_type(
    feed: dict[str, pl.DataFrame],
    ctx: ValidationContext,
) -> list[Notice]:
    """Emit forbidden_pickup_type/forbidden_drop_off_type (ERROR) for qualifying stop_times rows."""
    if "stop_times" not in feed:
        return []

    df = feed["stop_times"]
    cols = set(df.columns)

    has_start = "start_pickup_drop_off_window" in cols
    has_end = "end_pickup_drop_off_window" in cols

    # Guard: at least one window column must be present in the header
    if not has_start and not has_end:
        return []

    # Build shared window-present expression (OR of non-null checks for each present column)
    window_expr: pl.Expr = pl.lit(False)
    if has_start:
        window_expr = window_expr | pl.col("start_pickup_drop_off_window").is_not_null()
    if has_end:
        window_expr = window_expr | pl.col("end_pickup_drop_off_window").is_not_null()

    notices: list[Notice] = []

    # Sub-check A: forbidden pickup type (REGULAR=0 or ON_REQUEST_TO_DRIVER=3) with window set
    if "pickup_type" in cols:
        pickup_violations = df.filter(
            window_expr & pl.col("pickup_type").is_in([0, 3])
        )
        for row in pickup_violations.iter_rows(named=True):
            notices.append(
                Notice(
                    code="forbidden_pickup_type",
                    severity=Severity.ERROR,
                    fields={
                        "csv_row_number": row["csv_row_number"],
                        "start_pickup_drop_off_window": row.get("start_pickup_drop_off_window"),
                        "end_pickup_drop_off_window": row.get("end_pickup_drop_off_window"),
                    },
                )
            )

    # Sub-check B: forbidden drop-off type (REGULAR=0) with window set
    if "drop_off_type" in cols:
        dropoff_violations = df.filter(
            window_expr & pl.col("drop_off_type").eq(0)
        )
        for row in dropoff_violations.iter_rows(named=True):
            notices.append(
                Notice(
                    code="forbidden_drop_off_type",
                    severity=Severity.ERROR,
                    fields={
                        "csv_row_number": row["csv_row_number"],
                        "start_pickup_drop_off_window": row.get("start_pickup_drop_off_window"),
                        "end_pickup_drop_off_window": row.get("end_pickup_drop_off_window"),
                    },
                )
            )

    return notices
