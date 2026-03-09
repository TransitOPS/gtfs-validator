"""Validator: location_type / parent_station consistency for stops."""

from __future__ import annotations

import polars as pl

from gtfs_validator.context import ValidationContext
from gtfs_validator.notices import Notice, Severity


def validate_location_type_single_entity(
    feed: dict[str, pl.DataFrame],
    ctx: ValidationContext,
) -> list[Notice]:
    stops = feed.get("stops")
    if stops is None:
        return []

    # Ensure required columns exist; return early if not
    required = {"stop_id", "location_type"}
    if not required.issubset(stops.columns):
        return []

    notices: list[Notice] = []

    # Expression: parent_station is present (non-null and non-empty)
    has_parent = (
        pl.col("parent_station").is_not_null()
        & (pl.col("parent_station") != "")
    ) if "parent_station" in stops.columns else pl.lit(False)

    # Expression: platform_code is present (non-null and non-empty)
    has_platform_code = (
        pl.col("platform_code").is_not_null()
        & (pl.col("platform_code") != "")
    ) if "platform_code" in stops.columns else pl.lit(False)

    # --- Check 1: station_with_parent_station ---
    # Stations (location_type == 1) that have a parent_station
    if "parent_station" in stops.columns:
        station_violations = stops.filter(
            (pl.col("location_type") == 1) & has_parent
        )
        for row in station_violations.iter_rows(named=True):
            notices.append(
                Notice(
                    code="station_with_parent_station",
                    severity=Severity.ERROR,
                    fields={
                        "csvRowNumber": row["csv_row_number"],
                        "stopId": row["stop_id"],
                        "stopName": row.get("stop_name", ""),
                        "parentStation": row["parent_station"],
                    },
                )
            )

    # --- Check 2: location_without_parent_station ---
    # Entrances (2), generic nodes (3), boarding areas (4) without parent_station
    loc_violations = stops.filter(
        pl.col("location_type").is_in([2, 3, 4]) & ~has_parent
    )
    for row in loc_violations.iter_rows(named=True):
        notices.append(
            Notice(
                code="location_without_parent_station",
                severity=Severity.ERROR,
                fields={
                    "csvRowNumber": row["csv_row_number"],
                    "stopId": row["stop_id"],
                    "stopName": row.get("stop_name", ""),
                    "locationType": row["location_type"],
                },
            )
        )

    # --- Check 3: platform_without_parent_station ---
    # Stops (location_type == 0) with platform_code set but no parent_station
    if "platform_code" in stops.columns:
        platform_violations = stops.filter(
            (pl.col("location_type") == 0) & ~has_parent & has_platform_code
        )
        for row in platform_violations.iter_rows(named=True):
            notices.append(
                Notice(
                    code="platform_without_parent_station",
                    severity=Severity.INFO,
                    fields={
                        "csvRowNumber": row["csv_row_number"],
                        "stopId": row["stop_id"],
                        "stopName": row.get("stop_name", ""),
                    },
                )
            )

    return notices
