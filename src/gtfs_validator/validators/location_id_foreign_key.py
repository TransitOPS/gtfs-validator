"""Validate stop_times.location_id references against locations.geojson feature IDs."""

from __future__ import annotations

import polars as pl

from gtfs_validator.context import ValidationContext
from gtfs_validator.notices import Notice, Severity


def validate_location_id_foreign_key(
    feed: dict[str, pl.DataFrame],
    ctx: ValidationContext,
) -> list[Notice]:
    geojson_features = feed.get("locations_geojson")
    stop_times = feed.get("stop_times")

    # Guard: both tables must be present
    if not geojson_features or stop_times is None:
        return []

    # Guard: location_id column must exist in stop_times
    if "location_id" not in stop_times.columns:
        return []

    # Extract valid feature IDs from GeoJSON list
    valid_ids: set[str] = {f["id"] for f in geojson_features if f.get("id")}

    # Filter to rows with non-null, non-empty location_id
    candidates = stop_times.filter(
        pl.col("location_id").is_not_null() & (pl.col("location_id") != "")
    )

    if candidates.is_empty():
        return []

    # Find violations: location_id values not in the valid set
    violations = candidates.filter(
        ~pl.col("location_id").is_in(list(valid_ids))
    )

    notices: list[Notice] = []
    for row in violations.iter_rows(named=True):
        notices.append(
            Notice(
                code="foreign_key_violation",
                severity=Severity.ERROR,
                fields={
                    "childFilename": "stop_times.txt",
                    "childFieldName": "location_id",
                    "parentFilename": "locations.geojson",
                    "parentFieldName": "id",
                    "fieldValue": row["location_id"],
                    "csvRowNumber": row["csvRowNumber"],
                },
            )
        )
    return notices
