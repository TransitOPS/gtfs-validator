"""Validate timeframes.service_id references calendar or calendar_dates."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from gtfs_validator.notices import Notice, Severity

if TYPE_CHECKING:
    from gtfs_validator.context import ValidationContext


def validate_timeframe_service_id_foreign_key(
    feed: dict[str, pl.DataFrame],
    ctx: ValidationContext,
) -> list[Notice]:
    if "timeframes" not in feed or feed["timeframes"].is_empty():
        return []

    tf = feed["timeframes"]

    if "service_id" not in tf.columns:
        return []

    # Build union of valid service_id values from both parent tables.
    valid_ids: set[str] = set()

    calendar = feed.get("calendar")
    if calendar is not None and not calendar.is_empty() and "service_id" in calendar.columns:
        valid_ids.update(
            calendar.select("service_id")
            .filter(pl.col("service_id").is_not_null() & (pl.col("service_id") != ""))
            .to_series()
            .to_list()
        )

    calendar_dates = feed.get("calendar_dates")
    if calendar_dates is not None and not calendar_dates.is_empty() and "service_id" in calendar_dates.columns:
        valid_ids.update(
            calendar_dates.select("service_id")
            .filter(pl.col("service_id").is_not_null() & (pl.col("service_id") != ""))
            .to_series()
            .to_list()
        )

    # Candidate rows: timeframes rows with non-null, non-empty service_id
    candidates = tf.filter(
        pl.col("service_id").is_not_null() & (pl.col("service_id") != "")
    )

    if candidates.is_empty():
        return []

    # Violations: rows whose service_id is not in the union set
    violations = candidates.filter(~pl.col("service_id").is_in(list(valid_ids)))

    notices: list[Notice] = []
    for row in violations.iter_rows(named=True):
        notices.append(
            Notice(
                code="foreign_key_violation",
                severity=Severity.ERROR,
                fields={
                    "childFilename": "timeframes.txt",
                    "childFieldName": "service_id",
                    "parentFilename": "calendar.txt or calendar_dates.txt",
                    "parentFieldName": "service_id",
                    "fieldValue": row["service_id"],
                    "csvRowNumber": row["csvRowNumber"],
                },
            )
        )

    return notices
