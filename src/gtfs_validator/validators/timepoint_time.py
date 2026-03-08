"""Validate timepoint field consistency in stop_times."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from gtfs_validator.notices import Notice, Severity

if TYPE_CHECKING:
    from gtfs_validator.context import ValidationContext

_REQUIRED_COLS = {
    "csv_row_number",
    "trip_id",
    "stop_sequence",
    "arrival_time",
    "departure_time",
    "timepoint",
}

TIMEPOINT_EXACT = 1


def validate_timepoint_time(
    feed: dict[str, pl.DataFrame],
    ctx: ValidationContext,
) -> list[Notice]:
    if "stop_times" not in feed or feed["stop_times"].is_empty():
        return []

    st = feed["stop_times"]

    # Guard: timepoint column must exist in the header
    if "timepoint" not in st.columns:
        return []

    # Guard: all required columns must be present
    if not _REQUIRED_COLS.issubset(set(st.columns)):
        return []

    notices: list[Notice] = []

    # Check 1: row has at least one time but timepoint cell is null (WARNING)
    has_time = pl.col("arrival_time").is_not_null() | pl.col("departure_time").is_not_null()
    missing_tp = pl.col("timepoint").is_null()

    check1_rows = st.filter(has_time & missing_tp)
    for row in check1_rows.iter_rows(named=True):
        notices.append(
            Notice(
                code="missing_timepoint_value",
                severity=Severity.WARNING,
                fields={
                    "csv_row_number": row["csv_row_number"],
                    "trip_id": row["trip_id"],
                    "stop_sequence": row["stop_sequence"],
                },
            )
        )

    # Check 2: exact timepoints (timepoint == 1) missing arrival_time or departure_time
    exact = st.filter(pl.col("timepoint") == TIMEPOINT_EXACT)

    for row in exact.filter(pl.col("arrival_time").is_null()).iter_rows(named=True):
        notices.append(
            Notice(
                code="stop_time_timepoint_without_times",
                severity=Severity.ERROR,
                fields={
                    "csv_row_number": row["csv_row_number"],
                    "trip_id": row["trip_id"],
                    "stop_sequence": row["stop_sequence"],
                    "specified_field": "arrival_time",
                },
            )
        )

    for row in exact.filter(pl.col("departure_time").is_null()).iter_rows(named=True):
        notices.append(
            Notice(
                code="stop_time_timepoint_without_times",
                severity=Severity.ERROR,
                fields={
                    "csv_row_number": row["csv_row_number"],
                    "trip_id": row["trip_id"],
                    "stop_sequence": row["stop_sequence"],
                    "specified_field": "departure_time",
                },
            )
        )

    return notices
