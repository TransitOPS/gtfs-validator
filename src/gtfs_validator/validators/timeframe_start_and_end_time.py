"""Validate timeframes start_time and end_time fields."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from gtfs_validator.notices import Notice, Severity

if TYPE_CHECKING:
    from gtfs_validator.context import ValidationContext

TWENTY_FOUR_HOURS_SECONDS = 24 * 3600  # 86400


def _parse_time_seconds(col_name: str) -> pl.Expr:
    """Parse a GTFS time string column (H:MM:SS or HH:MM:SS) to total seconds.

    Returns an Int64 expression: hours*3600 + minutes*60 + seconds.
    Only safe to call on non-null rows (caller must gate with is_not_null()).
    """
    parts = pl.col(col_name).str.split(":")
    hours = parts.list.get(0).cast(pl.Int64)
    minutes = parts.list.get(1).cast(pl.Int64)
    seconds = parts.list.get(2).cast(pl.Int64)
    return hours * 3600 + minutes * 60 + seconds


def validate_timeframe_start_and_end_time(
    feed: dict[str, pl.DataFrame],
    ctx: ValidationContext,
) -> list[Notice]:
    if "timeframes" not in feed or feed["timeframes"].is_empty():
        return []

    tf = feed["timeframes"].select(["start_time", "end_time", "csv_row_number"])

    has_start = pl.col("start_time").is_not_null()
    has_end = pl.col("end_time").is_not_null()

    # Check 1: XOR — exactly one field is present
    xor_mask = has_start != has_end
    xor_violations = tf.filter(xor_mask)

    # Check 2: start_time present and > 24:00:00
    start_violations = tf.filter(
        has_start & (_parse_time_seconds("start_time") > TWENTY_FOUR_HOURS_SECONDS)
    )

    # Check 3: end_time present and > 24:00:00
    end_violations = tf.filter(
        has_end & (_parse_time_seconds("end_time") > TWENTY_FOUR_HOURS_SECONDS)
    )

    notices: list[Notice] = []

    for row in xor_violations.iter_rows(named=True):
        notices.append(
            Notice(
                code="timeframe_only_start_or_end_time_specified",
                severity=Severity.ERROR,
                fields={"csv_row_number": row["csv_row_number"]},
            )
        )

    for row in start_violations.iter_rows(named=True):
        notices.append(
            Notice(
                code="timeframe_start_or_end_time_greater_than_twenty_four_hours",
                severity=Severity.ERROR,
                fields={
                    "csv_row_number": row["csv_row_number"],
                    "field_name": "start_time",
                    "time": row["start_time"],
                },
            )
        )

    for row in end_violations.iter_rows(named=True):
        notices.append(
            Notice(
                code="timeframe_start_or_end_time_greater_than_twenty_four_hours",
                severity=Severity.ERROR,
                fields={
                    "csv_row_number": row["csv_row_number"],
                    "field_name": "end_time",
                    "time": row["end_time"],
                },
            )
        )

    return notices
