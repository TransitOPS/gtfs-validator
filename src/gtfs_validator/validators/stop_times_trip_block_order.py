"""Validator: StopTimesTripBlockOrderValidator.

Detects trips whose stop_times rows are non-contiguous in the file OR have
stop_sequence values that are not strictly increasing in file order.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from gtfs_validator.notices import Notice, Severity

if TYPE_CHECKING:
    from gtfs_validator.context import ValidationContext


def validate_stop_times_trip_block_order(
    feed: dict[str, pl.DataFrame],
    ctx: ValidationContext,
) -> list[Notice]:
    """Emit one unsorted_stop_times notice per trip with ordering issues."""
    if "stop_times" not in feed:
        return []

    st = feed["stop_times"]

    if st.is_empty():
        return []

    # Drop rows with null trip_id
    st = st.filter(pl.col("trip_id").is_not_null())
    if st.is_empty():
        return []

    # --- Contiguity check (vectorised groupby) ---
    agg = (
        st.group_by("trip_id")
        .agg(
            pl.col("csv_row_number").count().alias("row_count"),
            pl.col("csv_row_number").min().alias("min_row"),
            pl.col("csv_row_number").max().alias("max_row"),
        )
        .with_columns(
            (pl.col("max_row") - pl.col("min_row") + 1).alias("span")
        )
        .with_columns(
            (pl.col("span") > pl.col("row_count")).alias("non_contiguous")
        )
    )

    # --- Unsorted sequence check (file-order lag within trip) ---
    # Add a monotonic file-position index to preserve row order within groups.
    st_indexed = st.with_row_index("_file_row")

    # Within each trip (file order), find rows where stop_sequence <= previous
    st_lagged = st_indexed.with_columns(
        pl.col("stop_sequence")
          .shift(1)
          .over("trip_id", order_by="_file_row")
          .alias("_prev_seq")
    ).with_columns(
        (pl.col("stop_sequence") <= pl.col("_prev_seq")).alias("_is_unsorted")
    )

    unsorted_trips = (
        st_lagged
        .filter(pl.col("_is_unsorted"))
        .select("trip_id")
        .unique()
    )

    # --- Combine: join unsorted flag onto aggregated contiguity data ---
    result = (
        agg
        .join(
            unsorted_trips.with_columns(pl.lit(True).alias("unsorted_sequence")),
            on="trip_id",
            how="left",
        )
        .with_columns(pl.col("unsorted_sequence").fill_null(False))
        .filter(pl.col("non_contiguous") | pl.col("unsorted_sequence"))
    )

    # --- Construct notices (one per violating trip) ---
    notices = []
    for row in result.iter_rows(named=True):
        notices.append(Notice(
            code="unsorted_stop_times",
            severity=Severity.INFO,
            fields={
                "trip_id": row["trip_id"],
                "start_csv_row_number": row["min_row"],
                "end_csv_row_number": row["max_row"],
            },
        ))
    return notices
