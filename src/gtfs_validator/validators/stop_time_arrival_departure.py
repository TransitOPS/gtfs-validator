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

    # --- Check 2: vectorized with forward-fill + shift ---
    # Sort so rows within each trip are in stop_sequence order.
    st = stop_times.sort(["trip_id", "stop_sequence"])

    # Carry csv_row_number only for rows where departure is present (null otherwise).
    st = st.with_columns(
        pl.when(pl.col("departure_time").is_not_null())
          .then(pl.col("csv_row_number"))
          .alias("_dep_csv_row"),
    )

    # Forward-fill within each trip: propagate last non-null departure to later stops.
    # (Physical order == stop_sequence order since we pre-sorted.)
    st = st.with_columns([
        pl.col("departure_time").forward_fill().over("trip_id").alias("_ff_dep"),
        pl.col("_dep_csv_row").forward_fill().over("trip_id").alias("_ff_dep_csv_row"),
    ])

    # Shift by 1 within trip so each row sees the previous stop's last known departure.
    st = st.with_columns([
        pl.col("_ff_dep").shift(1).over("trip_id").alias("_prev_dep"),
        pl.col("_ff_dep_csv_row").shift(1).over("trip_id").alias("_prev_csv_row_number"),
    ])

    violations = st.filter(
        pl.col("arrival_time").is_not_null()
        & pl.col("_prev_dep").is_not_null()
        & (pl.col("arrival_time") < pl.col("_prev_dep"))
    )

    for row in violations.iter_rows(named=True):
        notices.append(
            Notice(
                code="stop_time_with_arrival_before_previous_departure_time",
                severity=Severity.ERROR,
                fields={
                    "csv_row_number": row["csv_row_number"],
                    "prev_csv_row_number": row["_prev_csv_row_number"],
                    "trip_id": row["trip_id"],
                    "arrival_time": row["arrival_time"],
                    "departure_time": row["_prev_dep"],
                },
            )
        )

    return notices
