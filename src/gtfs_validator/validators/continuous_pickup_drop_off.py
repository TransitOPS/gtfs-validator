"""Validator: forbid pickup/drop-off windows on continuous routes."""

from __future__ import annotations

import polars as pl

from gtfs_validator.context import ValidationContext
from gtfs_validator.notices import Notice, Severity

# The only non-continuous value for continuous_pickup / continuous_drop_off
NOT_AVAILABLE = 1


def _should_run(feed: dict[str, pl.DataFrame]) -> bool:
    if "routes" not in feed or "stop_times" not in feed:
        return False
    routes_cols = feed["routes"].columns
    st_cols = feed["stop_times"].columns

    has_continuous_pickup = "continuous_pickup" in routes_cols
    has_continuous_drop_off = "continuous_drop_off" in routes_cols
    has_start_window = "start_pickup_drop_off_window" in st_cols
    has_end_window = "end_pickup_drop_off_window" in st_cols

    # Java operator precedence: hasCol(cp) || (hasCol(cdo) && (hasCol(sw) || hasCol(ew)))
    return has_continuous_pickup or (
        has_continuous_drop_off and (has_start_window or has_end_window)
    )


def validate_continuous_pickup_drop_off(
    feed: dict[str, pl.DataFrame],
    ctx: ValidationContext,
) -> list[Notice]:
    """Check that stop times on continuous routes don't define pickup/drop-off windows."""
    if not _should_run(feed):
        return []

    routes = feed["routes"]
    stop_times = feed["stop_times"]

    # trips table is needed for the join; if absent, no join path exists
    if "trips" not in feed:
        return []
    trips = feed["trips"]

    # Filter routes to those with continuous behavior.
    routes_cols = routes.columns
    conditions: list[pl.Expr] = []
    if "continuous_pickup" in routes_cols:
        conditions.append(
            pl.col("continuous_pickup").is_not_null()
            & (pl.col("continuous_pickup") != NOT_AVAILABLE)
        )
    if "continuous_drop_off" in routes_cols:
        conditions.append(
            pl.col("continuous_drop_off").is_not_null()
            & (pl.col("continuous_drop_off") != NOT_AVAILABLE)
        )

    if not conditions:
        return []

    # Combine with OR
    continuous_filter = conditions[0]
    for c in conditions[1:]:
        continuous_filter = continuous_filter | c

    continuous_routes = routes.filter(continuous_filter).select(
        "route_id", "csv_row_number"
    )
    if continuous_routes.is_empty():
        return []

    continuous_routes = continuous_routes.rename(
        {"csv_row_number": "route_csv_row_number"}
    )

    # Join continuous routes -> trips -> stop_times
    joined = (
        continuous_routes
        .join(trips.select("trip_id", "route_id"), on="route_id", how="inner")
        .join(stop_times, on="trip_id", how="inner")
    )

    if joined.is_empty():
        return []

    # Filter to rows where at least one window field is non-null
    st_cols = stop_times.columns
    window_conditions: list[pl.Expr] = []
    if "start_pickup_drop_off_window" in st_cols:
        window_conditions.append(pl.col("start_pickup_drop_off_window").is_not_null())
    if "end_pickup_drop_off_window" in st_cols:
        window_conditions.append(pl.col("end_pickup_drop_off_window").is_not_null())

    if not window_conditions:
        return []

    window_filter = window_conditions[0]
    for w in window_conditions[1:]:
        window_filter = window_filter | w

    violations = joined.filter(window_filter)

    if violations.is_empty():
        return []

    # Emit one notice per violation row
    notices: list[Notice] = []
    has_start = "start_pickup_drop_off_window" in violations.columns
    has_end = "end_pickup_drop_off_window" in violations.columns

    for row in violations.iter_rows(named=True):
        start_val = row.get("start_pickup_drop_off_window") if has_start else None
        end_val = row.get("end_pickup_drop_off_window") if has_end else None
        notices.append(Notice(
            code="forbidden_continuous_pickup_drop_off",
            severity=Severity.ERROR,
            fields={
                "route_csv_row_number": row["route_csv_row_number"],
                "trip_id": row["trip_id"],
                "stop_time_csv_row_number": row["csv_row_number"],
                "start_pickup_drop_off_window": (
                    str(start_val) if start_val is not None else None
                ),
                "end_pickup_drop_off_window": (
                    str(end_val) if end_val is not None else None
                ),
            },
        ))
    return notices
