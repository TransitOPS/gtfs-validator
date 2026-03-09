"""Validator: PickupDropOffWindowValidator.

Detects stop_times rows where pickup/drop-off time windows are inconsistent:
- Check A: arrival_time or departure_time present alongside window fields (forbidden combination).
- Check B: exactly one window field is present (both must be provided together).
- Check C: end_pickup_drop_off_window is not strictly greater than start_pickup_drop_off_window.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from gtfs_validator.notices import Notice, Severity

if TYPE_CHECKING:
    from gtfs_validator.context import ValidationContext


def validate_pickup_drop_off_window(
    feed: dict[str, pl.DataFrame],
    ctx: ValidationContext,
) -> list[Notice]:
    """Emit window-related notices (ERROR) for qualifying stop_times rows."""
    if "stop_times" not in feed:
        return []

    df = feed["stop_times"]
    cols = set(df.columns)

    has_start = "start_pickup_drop_off_window" in cols
    has_end = "end_pickup_drop_off_window" in cols

    # Guard: at least one window column must be present in the header
    if not has_start and not has_end:
        return []

    # Build window_expr: at least one window field is non-null for this row
    window_expr: pl.Expr = pl.lit(False)
    if has_start:
        window_expr = window_expr | pl.col("start_pickup_drop_off_window").is_not_null()
    if has_end:
        window_expr = window_expr | pl.col("end_pickup_drop_off_window").is_not_null()

    # Subset: rows where at least one window field is non-null
    windowed = df.filter(window_expr)

    notices: list[Notice] = []

    # ---- Check A: arrival_time or departure_time present alongside window fields ----

    has_arrival = "arrival_time" in cols
    has_departure = "departure_time" in cols

    time_expr: pl.Expr = pl.lit(False)
    if has_arrival:
        time_expr = time_expr | pl.col("arrival_time").is_not_null()
    if has_departure:
        time_expr = time_expr | pl.col("departure_time").is_not_null()

    check_a_violations = windowed.filter(time_expr)
    for row in check_a_violations.iter_rows(named=True):
        notices.append(
            Notice(
                code="forbidden_arrival_or_departure_time",
                severity=Severity.ERROR,
                fields={
                    "csv_row_number": row["csv_row_number"],
                    "arrival_time": row.get("arrival_time"),
                    "departure_time": row.get("departure_time"),
                    "start_pickup_drop_off_window": row.get("start_pickup_drop_off_window"),
                    "end_pickup_drop_off_window": row.get("end_pickup_drop_off_window"),
                },
            )
        )

    # ---- Check B: exactly one window present (missing the other) ----

    # both_windows_expr: True only when ALL present window columns are non-null
    both_windows_expr: pl.Expr = pl.lit(True)
    if has_start:
        both_windows_expr = both_windows_expr & pl.col("start_pickup_drop_off_window").is_not_null()
    if has_end:
        both_windows_expr = both_windows_expr & pl.col("end_pickup_drop_off_window").is_not_null()

    check_b_violations = windowed.filter(~both_windows_expr)
    for row in check_b_violations.iter_rows(named=True):
        notices.append(
            Notice(
                code="missing_pickup_or_drop_off_window",
                severity=Severity.ERROR,
                fields={
                    "csv_row_number": row["csv_row_number"],
                    "start_pickup_drop_off_window": row.get("start_pickup_drop_off_window"),
                    "end_pickup_drop_off_window": row.get("end_pickup_drop_off_window"),
                },
            )
        )

    # ---- Check C: end window not strictly after start window ----
    # Only evaluated for rows where both windows are present (complement of check B rows)
    # Guard: both columns must exist in header for check C to be safe
    if has_start and has_end:
        check_c_candidates = windowed.filter(both_windows_expr)
        check_c_violations = check_c_candidates.filter(
            pl.col("end_pickup_drop_off_window") <= pl.col("start_pickup_drop_off_window")
        )
        for row in check_c_violations.iter_rows(named=True):
            notices.append(
                Notice(
                    code="invalid_pickup_drop_off_window",
                    severity=Severity.ERROR,
                    fields={
                        "csv_row_number": row["csv_row_number"],
                        "start_pickup_drop_off_window": row["start_pickup_drop_off_window"],
                        "end_pickup_drop_off_window": row["end_pickup_drop_off_window"],
                    },
                )
            )

    return notices
