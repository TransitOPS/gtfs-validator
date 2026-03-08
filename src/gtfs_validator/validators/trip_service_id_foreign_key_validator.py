"""Validate trips.service_id references calendar or calendar_dates."""

from __future__ import annotations

import polars as pl

from gtfs_validator.context import ValidationContext
from gtfs_validator.notices import Notice, Severity


def validate_trip_service_id_foreign_key(
    feed: dict[str, pl.DataFrame],
    ctx: ValidationContext,
) -> list[Notice]:
    trips = feed.get("trips")
    if trips is None or trips.is_empty():
        return []

    if "service_id" not in trips.columns:
        return []

    # Collect valid service_id values from both parent tables
    valid_ids: set[str] = set()

    calendar = feed.get("calendar")
    if calendar is not None and "service_id" in calendar.columns:
        cal_ids = (
            calendar.select("service_id")
            .filter(pl.col("service_id").is_not_null() & (pl.col("service_id") != ""))
            .to_series()
            .to_list()
        )
        valid_ids.update(cal_ids)

    calendar_dates = feed.get("calendar_dates")
    if calendar_dates is not None and "service_id" in calendar_dates.columns:
        cd_ids = (
            calendar_dates.select("service_id")
            .filter(pl.col("service_id").is_not_null() & (pl.col("service_id") != ""))
            .to_series()
            .to_list()
        )
        valid_ids.update(cd_ids)

    # Filter trips to rows with non-null, non-empty service_id
    candidates = trips.filter(
        pl.col("service_id").is_not_null() & (pl.col("service_id") != "")
    )

    if candidates.is_empty():
        return []

    # Find violations: rows whose service_id is not in valid_ids
    violations = candidates.filter(~pl.col("service_id").is_in(list(valid_ids)))

    # Emit one notice per violating row
    notices: list[Notice] = []
    for row in violations.iter_rows(named=True):
        notices.append(
            Notice(
                code="foreign_key_violation",
                severity=Severity.ERROR,
                fields={
                    "childFilename": "trips.txt",
                    "childFieldName": "service_id",
                    "parentFilename": "calendar.txt or calendar_dates.txt",
                    "parentFieldName": "service_id",
                    "fieldValue": row["service_id"],
                    "csvRowNumber": row["csvRowNumber"],
                },
            )
        )
    return notices
