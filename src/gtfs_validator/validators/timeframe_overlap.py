"""Validator: TimeframeOverlapValidator.

Detects pairs of timeframes.txt entries for the same (timeframe_group_id,
service_id) group whose time windows overlap (curr.start_time < prev.end_time
after sorting by start_time ASC, end_time ASC).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from gtfs_validator.notices import Notice, Severity

if TYPE_CHECKING:
    from gtfs_validator.context import ValidationContext


def _time_to_secs(t: str) -> int:
    """Convert a GTFS HH:MM:SS time string to total seconds since midnight."""
    h, m, s = t.split(":")
    return int(h) * 3600 + int(m) * 60 + int(s)


def validate_timeframe_overlap(
    feed: dict[str, pl.DataFrame],
    ctx: ValidationContext,
) -> list[Notice]:
    """Emit timeframe_overlap (ERROR) for timeframes.txt entries whose windows overlap."""
    if "timeframes" not in feed or feed["timeframes"].is_empty():
        return []

    tf = feed["timeframes"]

    # Drop rows where start_time or end_time are null.
    # These should have been caught at load time, but be defensive.
    tf = tf.filter(
        pl.col("start_time").is_not_null() & pl.col("end_time").is_not_null()
    )
    if tf.is_empty():
        return []

    # Select only the columns this validator needs
    tf = tf.select(
        ["timeframe_group_id", "service_id", "start_time", "end_time", "csv_row_number"]
    )

    # Add integer-seconds columns for sorting and comparison.
    # HH:MM:SS strings can represent times >= 24:00:00 (e.g. "25:00:00"),
    # so we cannot use Polars time dtype — convert to int seconds instead.
    tf = tf.with_columns([
        pl.col("start_time")
        .map_elements(_time_to_secs, return_dtype=pl.Int64)
        .alias("start_secs"),
        pl.col("end_time")
        .map_elements(_time_to_secs, return_dtype=pl.Int64)
        .alias("end_secs"),
    ])

    notices: list[Notice] = []

    # Group by (timeframe_group_id, service_id); check each group independently
    for (group_id, svc_id), group in tf.group_by(["timeframe_group_id", "service_id"]):
        if len(group) < 2:
            continue

        # Sort by start_secs ASC, then end_secs ASC (tie-break)
        sorted_group = group.sort(["start_secs", "end_secs"])
        rows = sorted_group.to_dicts()

        # Scan consecutive pairs only (not all-pairs)
        for i in range(len(rows) - 1):
            prev = rows[i]
            curr = rows[i + 1]
            if curr["start_secs"] < prev["end_secs"]:
                notices.append(
                    Notice(
                        code="timeframe_overlap",
                        severity=Severity.ERROR,
                        fields={
                            "prev_csv_row_number": prev["csv_row_number"],
                            "prev_end_time": prev["end_time"],
                            "curr_csv_row_number": curr["csv_row_number"],
                            "curr_start_time": curr["start_time"],
                            "timeframe_group_id": group_id,
                            "service_id": svc_id,
                        },
                    )
                )

    return notices
