"""StopTimesRecordValidator: validates that a GTFS-Flex trip with a single
stop_times row using MUST_PHONE pickup/drop-off has at least one more record."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from gtfs_validator.notices import Notice, Severity

if TYPE_CHECKING:
    from gtfs_validator.context import ValidationContext

_MUST_PHONE = 2


def validate_stop_times_record(
    feed: dict[str, pl.DataFrame],
    ctx: ValidationContext,
) -> list[Notice]:
    """Validate that a MUST_PHONE pickup/drop-off stop_times row is not the
    sole record for its trip.

    Emits ``missing_stop_times_record`` (ERROR) when a row has both window
    times set, both pickup_type and drop_off_type equal to MUST_PHONE (2), and
    is the only stop_times record for its trip_id.
    """
    # Guard 1: stop_times table absent
    if "stop_times" not in feed:
        return []

    st = feed["stop_times"]

    # Guard 2: any of the four required columns missing from the schema
    required_cols = {
        "start_pickup_drop_off_window",
        "end_pickup_drop_off_window",
        "pickup_type",
        "drop_off_type",
    }
    if not required_cols.issubset(set(st.columns)):
        return []

    # Guard 3: empty table
    if st.is_empty():
        return []

    # Record which optional columns are present before any transformations
    has_location_group = "location_group_id" in st.columns
    has_location_id = "location_id" in st.columns

    # Precompute per-trip row counts via group_by then join back
    trip_counts = st.group_by("trip_id").agg(pl.len().alias("trip_stop_count"))
    st_with_count = st.join(trip_counts, on="trip_id", how="left")

    # Filter for rows that satisfy all five conditions simultaneously
    violations = st_with_count.filter(
        pl.col("start_pickup_drop_off_window").is_not_null()
        & pl.col("end_pickup_drop_off_window").is_not_null()
        & (pl.col("pickup_type") == _MUST_PHONE)
        & (pl.col("drop_off_type") == _MUST_PHONE)
        & (pl.col("trip_stop_count") == 1)
    )

    notices: list[Notice] = []
    for row in violations.iter_rows(named=True):
        notices.append(
            Notice(
                code="missing_stop_times_record",
                severity=Severity.ERROR,
                fields={
                    "csv_row_number": row["csv_row_number"],
                    "trip_id": row["trip_id"],
                    "location_group_id": row.get("location_group_id") if has_location_group else None,
                    "location_id": row.get("location_id") if has_location_id else None,
                },
            )
        )
    return notices
