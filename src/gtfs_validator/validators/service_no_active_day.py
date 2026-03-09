"""Validator: ServiceHasNoActiveDayOfTheWeekValidator."""

from __future__ import annotations

import polars as pl

from gtfs_validator.context import ValidationContext
from gtfs_validator.notices import Notice, Severity

_DAY_COLUMNS: list[str] = [
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"
]
_NOT_AVAILABLE: int = 0


def _all_days_inactive() -> pl.Expr:
    """Boolean expression: True when every day column equals NOT_AVAILABLE (0).

    fill_null(0) replicates Java's enum default of NOT_AVAILABLE for missing
    required fields. Nulls in day columns should already produce a load-time
    notice; treating them as 0 here is a safe defensive fallback that does not
    change observable output.
    """
    return pl.all_horizontal(
        [pl.col(day).fill_null(_NOT_AVAILABLE).eq(_NOT_AVAILABLE) for day in _DAY_COLUMNS]
    )


def validate_service_no_active_day(
    feed: dict[str, pl.DataFrame],
    ctx: ValidationContext,
) -> list[Notice]:
    """Emit a WARNING for each calendar row where all day-of-week columns are 0."""
    if "calendar" not in feed or feed["calendar"].is_empty():
        return []

    df = feed["calendar"]

    offending = df.filter(
        pl.col("service_id").is_not_null() & _all_days_inactive()
    )

    if offending.is_empty():
        return []

    notices: list[Notice] = []
    for row in offending.select(["service_id"]).iter_rows(named=True):
        notices.append(
            Notice(
                code="service_has_no_active_day_of_the_week",
                severity=Severity.WARNING,
                fields={
                    "serviceId": row["service_id"],
                },
            )
        )
    return notices
