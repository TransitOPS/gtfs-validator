"""Validator: StopTimeIncreasingDistanceValidator.

Checks that shape_dist_traveled values are strictly increasing within each
trip (ordered by stop_sequence), ignoring rows where stop_id is null.
"""

from __future__ import annotations

import polars as pl

from gtfs_validator.context import ValidationContext
from gtfs_validator.notices import Notice, Severity


def validate_stop_time_increasing_distance(
    feed: dict[str, pl.DataFrame],
    ctx: ValidationContext,
) -> list[Notice]:
    """Emit a notice for each adjacent pair of eligible stop_times rows where
    shape_dist_traveled is non-increasing within a trip.

    ctx is accepted for interface consistency but is not consulted.
    """
    if "stop_times" not in feed:
        return []

    stop_times = feed["stop_times"]

    if stop_times.is_empty():
        return []

    if "stop_id" not in stop_times.columns:
        return []

    if "shape_dist_traveled" not in stop_times.columns:
        return []

    sorted_st = stop_times.sort(["trip_id", "stop_sequence"])

    notices: list[Notice] = []

    for trip_df in sorted_st.partition_by("trip_id", maintain_order=True):
        prev: dict | None = None
        for row in trip_df.select([
            "csv_row_number",
            "trip_id",
            "stop_id",
            "stop_sequence",
            "shape_dist_traveled",
        ]).iter_rows(named=True):
            if row["stop_id"] is None:
                continue  # skip; do not update prev
            if (
                prev is not None
                and prev["shape_dist_traveled"] is not None
                and row["shape_dist_traveled"] is not None
                and prev["shape_dist_traveled"] >= row["shape_dist_traveled"]
            ):
                notices.append(Notice(
                    code="decreasing_or_equal_stop_time_distance",
                    severity=Severity.ERROR,
                    fields={
                        "trip_id": row["trip_id"],
                        "stop_id": row["stop_id"],
                        "csv_row_number": row["csv_row_number"],
                        "shape_dist_traveled": row["shape_dist_traveled"],
                        "stop_sequence": row["stop_sequence"],
                        "prev_csv_row_number": prev["csv_row_number"],
                        "prev_shape_dist_traveled": prev["shape_dist_traveled"],
                        "prev_stop_sequence": prev["stop_sequence"],
                    },
                ))
            prev = row  # always update prev for eligible rows

    return notices
