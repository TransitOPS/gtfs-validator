"""Validator: OverlappingFrequencyValidator.

Detects pairs of frequencies.txt entries for the same trip whose time
windows overlap (curr.start_time < prev.end_time after sorting).
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


def validate_overlapping_frequency(
    feed: dict[str, pl.DataFrame],
    ctx: ValidationContext,
) -> list[Notice]:
    """Emit overlapping_frequency (ERROR) for frequencies.txt entries whose windows overlap."""
    frequencies = feed.get("frequencies")
    if frequencies is None or frequencies.is_empty():
        return []

    # Guard against null required time columns
    frequencies = frequencies.filter(
        pl.col("start_time").is_not_null() & pl.col("end_time").is_not_null()
    )
    if frequencies.is_empty():
        return []

    # Add integer-seconds columns for sorting and comparison
    frequencies = frequencies.with_columns([
        pl.col("start_time")
        .map_elements(_time_to_secs, return_dtype=pl.Int64)
        .alias("start_secs"),
        pl.col("end_time")
        .map_elements(_time_to_secs, return_dtype=pl.Int64)
        .alias("end_secs"),
    ])

    notices: list[Notice] = []

    for (trip_id,), group in frequencies.group_by("trip_id"):
        if len(group) < 2:
            continue

        sorted_group = group.sort(["start_secs", "end_secs", "headway_secs"])
        rows = sorted_group.to_dicts()

        for i in range(len(rows) - 1):
            prev = rows[i]
            curr = rows[i + 1]
            if curr["start_secs"] < prev["end_secs"]:
                notices.append(
                    Notice(
                        code="overlapping_frequency",
                        severity=Severity.ERROR,
                        fields={
                            "prev_csv_row_number": prev["csv_row_number"],
                            "prev_end_time": prev["end_time"],
                            "curr_csv_row_number": curr["csv_row_number"],
                            "curr_start_time": curr["start_time"],
                            "trip_id": trip_id,
                        },
                    )
                )

    return notices
