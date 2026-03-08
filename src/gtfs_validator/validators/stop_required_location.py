"""Validator: StopRequiredLocationValidator.

Checks that stops, stations, and entrances (location_type 0, 1, 2) have both
``stop_lat`` and ``stop_lon`` present. Partial presence (only one coordinate)
is treated identically to full absence.
"""

from __future__ import annotations

import polars as pl

from gtfs_validator.context import ValidationContext
from gtfs_validator.notices import Notice, Severity

_REQUIRED_LOCATION_TYPES: list[int] = [0, 1, 2]  # STOP, STATION, ENTRANCE


def validate_stop_required_location(
    feed: dict[str, pl.DataFrame],
    ctx: ValidationContext,
) -> list[Notice]:
    """Validate that stops/stations/entrances have stop_lat and stop_lon."""
    if "stops" not in feed:
        return []

    stops = feed["stops"]

    if stops.is_empty():
        return []

    # Determine location_type expression
    if "location_type" in stops.columns:
        loc_type_expr = pl.col("location_type")
    else:
        loc_type_expr = pl.lit(0)  # all rows default to STOP

    # Determine lat/lon missing expressions
    lat_missing = (
        pl.col("stop_lat").is_null() if "stop_lat" in stops.columns else pl.lit(True)
    )
    lon_missing = (
        pl.col("stop_lon").is_null() if "stop_lon" in stops.columns else pl.lit(True)
    )

    offenders = stops.filter(
        loc_type_expr.is_in(_REQUIRED_LOCATION_TYPES) & (lat_missing | lon_missing)
    )

    notices: list[Notice] = []
    location_type_in_columns = "location_type" in stops.columns

    for row in offenders.iter_rows(named=True):
        notices.append(
            Notice(
                code="stop_without_location",
                severity=Severity.ERROR,
                fields={
                    "csv_row_number": row["csv_row_number"],
                    "stop_id": row["stop_id"],
                    "location_type": row["location_type"] if location_type_in_columns else 0,
                },
            )
        )

    return notices
