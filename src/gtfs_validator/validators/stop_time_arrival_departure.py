"""Validator: StopTimeArrivalAndDepartureTimeValidator.

Two checks on stop_times.txt:
- Check 1 (vectorized): each row must have either both arrival_time and
  departure_time present, or both absent (XOR → error).
- Check 2 (stateful per-trip): no stop's arrival_time may be strictly less than
  the most recent preceding stop's departure_time within the same trip.
"""

from __future__ import annotations

import polars as pl

from gtfs_validator.context import ValidationContext
from gtfs_validator.notices import Notice, Severity

_REQUIRED_COLS = {
    "csv_row_number",
    "trip_id",
    "stop_sequence",
    "arrival_time",
    "departure_time",
}


def validate_stop_time_arrival_departure(
    feed: dict[str, pl.DataFrame],
    ctx: ValidationContext,
) -> list[Notice]:
    """Emit errors for arrival/departure time pairing and ordering violations."""
    if "stop_times" not in feed:
        return []
    stop_times = feed["stop_times"]
    if stop_times.is_empty():
        return []
    if not _REQUIRED_COLS.issubset(set(stop_times.columns)):
        return []

    notices: list[Notice] = []

    # --- Check 1: XOR — exactly one of arrival_time / departure_time is present ---
    has_arrival = pl.col("arrival_time").is_not_null()
    has_departure = pl.col("departure_time").is_not_null()
    offenders = stop_times.filter(has_arrival != has_departure)

    for row in offenders.iter_rows(named=True):
        specified_field = (
            "arrival_time" if row["arrival_time"] is not None else "departure_time"
        )
        notices.append(
            Notice(
                code="stop_time_with_only_arrival_or_departure_time",
                severity=Severity.ERROR,
                fields={
                    "csv_row_number": row["csv_row_number"],
                    "trip_id": row["trip_id"],
                    "stop_sequence": row["stop_sequence"],
                    "specified_field": specified_field,
                },
            )
        )

    # --- Check 2: arrival must not be strictly before most recent prior departure ---
    sorted_st = stop_times.sort(["trip_id", "stop_sequence"])

    for trip_df in sorted_st.partition_by("trip_id", maintain_order=True):
        rows = trip_df.select(
            [
                "csv_row_number",
                "trip_id",
                "stop_sequence",
                "arrival_time",
                "departure_time",
            ]
        ).iter_rows(named=True)

        prev_dep_row: dict | None = None

        for row in rows:
            arrival = row["arrival_time"]
            departure = row["departure_time"]

            if arrival is not None and prev_dep_row is not None:
                if arrival < prev_dep_row["departure_time"]:
                    notices.append(
                        Notice(
                            code="stop_time_with_arrival_before_previous_departure_time",
                            severity=Severity.ERROR,
                            fields={
                                "csv_row_number": row["csv_row_number"],
                                "prev_csv_row_number": prev_dep_row["csv_row_number"],
                                "trip_id": row["trip_id"],
                                "arrival_time": arrival,
                                "departure_time": prev_dep_row["departure_time"],
                            },
                        )
                    )

            if departure is not None:
                prev_dep_row = row

    return notices
