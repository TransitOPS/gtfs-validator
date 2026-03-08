"""Validator: StopZoneIdValidator — stops missing zone_id served by zone-fare routes."""

from __future__ import annotations

import polars as pl

from gtfs_validator.context import ValidationContext
from gtfs_validator.notices import Notice, Severity


def validate_stop_zone_id(
    feed: dict[str, pl.DataFrame],
    ctx: ValidationContext,
) -> list[Notice]:
    """Emit INFO for each stop (location_type=0) missing zone_id that is served
    by a route appearing in a fare_rules row with a non-null zone column."""

    # Guard 1: fare_rules.txt absent or empty
    if "fare_rules" not in feed or feed["fare_rules"].is_empty():
        return []

    fare_rules = feed["fare_rules"]

    # Identify which zone columns are actually present
    zone_cols = [c for c in ("origin_id", "destination_id", "contains_id")
                 if c in fare_rules.columns]

    if not zone_cols:
        return []

    # Guard 2: hasFareZoneStructure — at least one zone column is non-null somewhere
    has_zone_structure = (
        fare_rules
        .select(pl.any_horizontal([pl.col(c).is_not_null() for c in zone_cols]))
        .to_series()
        .any()
    )
    if not has_zone_structure:
        return []

    # Guard 3: required tables present
    if "stops" not in feed or "stop_times" not in feed or "trips" not in feed:
        return []

    stops = feed["stops"]
    stop_times = feed["stop_times"]
    trips = feed["trips"]

    # Build set of route_ids that appear in zone-dependent fare rules.
    # drop_nulls() intentionally excludes global zone rules (route_id null),
    # matching the Java behavior where null route_id never matches a real route.
    zone_fare_routes = (
        fare_rules
        .filter(pl.any_horizontal([pl.col(c).is_not_null() for c in zone_cols]))
        .select("route_id")
        .drop_nulls()
        .unique()
    )

    if zone_fare_routes.is_empty():
        return []

    # Candidate stops: location_type == 0 and zone_id null
    # Retain only columns needed for the join and notice construction.
    required_stop_cols = [c for c in ("stop_id", "stop_name", "csv_row_number")
                          if c in stops.columns]
    candidates = (
        stops
        .filter(
            (pl.col("location_type") == 0) & pl.col("zone_id").is_null()
        )
        .select(required_stop_cols)
    )

    if candidates.is_empty():
        return []

    # Retain only the columns we need from stop_times and trips
    st_cols = [c for c in ("stop_id", "trip_id") if c in stop_times.columns]
    tr_cols = [c for c in ("trip_id", "route_id") if c in trips.columns]

    if "stop_id" not in st_cols or "trip_id" not in st_cols:
        return []
    if "trip_id" not in tr_cols or "route_id" not in tr_cols:
        return []

    # Join chain: candidates -> stop_times -> trips -> zone_fare_routes
    violators = (
        candidates
        .join(stop_times.select(st_cols), on="stop_id", how="inner")
        .join(trips.select(tr_cols), on="trip_id", how="inner")
        .join(zone_fare_routes, on="route_id", how="inner")
        .select([c for c in ("stop_id", "stop_name", "csv_row_number")
                 if c in required_stop_cols])
        .unique(subset=["stop_id"])   # one notice per stop (mirrors Java break)
    )

    notices: list[Notice] = []
    for row in violators.iter_rows(named=True):
        notices.append(
            Notice(
                code="stop_without_zone_id",
                severity=Severity.INFO,
                fields={
                    "stop_id": row["stop_id"],
                    "stop_name": row.get("stop_name"),
                    "csv_row_number": row.get("csv_row_number"),
                },
            )
        )
    return notices
