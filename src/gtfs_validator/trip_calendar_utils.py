"""Shared helpers for trip counting and majority service window computation.

Used by DateTripsValidator and FeedExpirationDateValidator.
"""

from __future__ import annotations

from datetime import date

import polars as pl

# --- Constants ---------------------------------------------------------------

MAX_SERVICE_DATE_TRIP_COUNT_RATIO = 0.90
MAX_SERVICE_DATE_TRIP_COUNT_LIMIT = 30
MAJORITY_TRIP_COUNT_RATIO = 0.75


def count_trips_per_service_date(
    feed: dict[str, pl.DataFrame],
    service_date_map: dict[str, set[date]],
) -> dict[date, int]:
    """Count effective trips running on each service date.

    For each trip, determines its effective count (1 for non-frequency
    trips, expanded count for frequency-based trips), aggregates by
    service_id, then distributes across each service_id's active dates.

    Returns a mapping from date to total trip count.
    """
    if "trips" not in feed or feed["trips"].is_empty():
        return {}

    trips = feed["trips"].select("trip_id", "service_id")

    # Expand frequency-based trips
    if "frequencies" in feed and not feed["frequencies"].is_empty():
        freqs = feed["frequencies"].select(
            "trip_id", "start_time", "end_time", "headway_secs"
        )

        # For each frequency row, compute expanded trip count:
        #   1 + floor((end_time - start_time - 1) / headway_secs)
        # but only when headway_secs > 0; otherwise count as 1.
        #
        # start_time and end_time are Duration types (GTFS TIME fields).
        # Convert to total seconds for arithmetic.
        freq_counts = freqs.with_columns(
            [
                (
                    pl.col("end_time").dt.total_seconds()
                    - pl.col("start_time").dt.total_seconds()
                ).alias("span_secs"),
            ]
        ).with_columns(
            pl.when(pl.col("headway_secs") > 0)
            .then(1 + ((pl.col("span_secs") - 1) // pl.col("headway_secs")))
            .otherwise(1)
            .cast(pl.Int64)
            .alias("freq_trip_count")
        )

        # Sum freq_trip_count per trip_id
        freq_per_trip = freq_counts.group_by("trip_id").agg(
            pl.col("freq_trip_count").sum().alias("effective_count")
        )

        # Left join trips with freq counts; trips not in frequencies get count=1
        trips = trips.join(freq_per_trip, on="trip_id", how="left").with_columns(
            pl.col("effective_count").fill_null(1)
        )
    else:
        trips = trips.with_columns(pl.lit(1).alias("effective_count"))

    # Aggregate trip counts per service_id
    service_counts = trips.group_by("service_id").agg(
        pl.col("effective_count").sum().alias("trip_count")
    )

    # Distribute counts across service dates
    date_counts: dict[date, int] = {}
    for row in service_counts.iter_rows(named=True):
        sid = row["service_id"]
        count = row["trip_count"]
        for d in service_date_map.get(sid, set()):
            date_counts[d] = date_counts.get(d, 0) + count

    return date_counts


def compute_majority_service_coverage(
    date_trip_counts: dict[date, int],
) -> tuple[date, date] | None:
    """Compute the majority service window from per-date trip counts.

    Returns (majority_start, majority_end) or None if no trips exist.
    """
    if not date_trip_counts:
        return None

    # Sort all trip counts to find the "typical max"
    sorted_counts = sorted(date_trip_counts.values())
    n = len(sorted_counts)

    # Determine "typical max" index, avoiding outliers
    index = max(int(MAX_SERVICE_DATE_TRIP_COUNT_RATIO * n), n - MAX_SERVICE_DATE_TRIP_COUNT_LIMIT)
    # Clamp to valid range
    index = max(0, min(index, n - 1))
    typical_max = sorted_counts[index]

    if typical_max == 0:
        return None

    # Compute majority threshold
    threshold = int(MAJORITY_TRIP_COUNT_RATIO * typical_max)

    # Find first and last dates meeting the threshold
    sorted_dates = sorted(date_trip_counts.keys())
    majority_start = None
    majority_end = None

    for d in sorted_dates:
        if date_trip_counts[d] >= threshold:
            if majority_start is None:
                majority_start = d
            majority_end = d

    if majority_start is None or majority_end is None:
        return None

    return (majority_start, majority_end)
