"""Validator: MissingCalendarAndCalendarDateValidator."""

from __future__ import annotations

import polars as pl

from gtfs_validator.context import ValidationContext
from gtfs_validator.notices import Notice, Severity


def validate_missing_calendar_and_calendar_date(
    feed: dict[str, pl.DataFrame],
    ctx: ValidationContext,
) -> list[Notice]:
    """Emit an error when neither calendar.txt nor calendar_dates.txt is present."""
    calendar_missing = "calendar" not in feed
    calendar_dates_missing = "calendar_dates" not in feed

    if calendar_missing and calendar_dates_missing:
        return [
            Notice(
                code="missing_calendar_and_calendar_date_files",
                severity=Severity.ERROR,
                fields={},
            )
        ]

    return []
