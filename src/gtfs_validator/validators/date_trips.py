"""DateTripsValidator: checks that trip coverage is active for the next 7 days."""

from __future__ import annotations

from datetime import timedelta

import polars as pl

from gtfs_validator.calendar_utils import build_service_date_map
from gtfs_validator.context import ValidationContext
from gtfs_validator.notices import Notice, Severity
from gtfs_validator.trip_calendar_utils import (
    compute_majority_service_coverage,
    count_trips_per_service_date,
)


def validate_date_trips(
    feed: dict[str, pl.DataFrame],
    ctx: ValidationContext,
) -> list[Notice]:
    """Validate that the majority service window covers the next 7 days."""
    # 1. Build service date map (reuse existing helper)
    service_date_map = build_service_date_map(feed)

    # 2. Count trips per service date
    date_trip_counts = count_trips_per_service_date(feed, service_date_map)

    # 3. Compute majority service window
    coverage = compute_majority_service_coverage(date_trip_counts)
    if coverage is None:
        return []

    majority_start, majority_end = coverage

    # 4. Check 7-day coverage
    validation_date = ctx.date_for_validation
    required_end = validation_date + timedelta(days=7)

    # No notice if: majority_start <= validation_date AND majority_end >= required_end
    if majority_start <= validation_date and majority_end >= required_end:
        return []

    # 5. Emit notice
    return [
        Notice(
            code="trip_coverage_not_active_for_next_7_days",
            severity=Severity.WARNING,
            fields={
                "current_date": validation_date.strftime("%Y%m%d"),
                "service_window_start_date": majority_start.strftime("%Y%m%d"),
                "service_window_end_date": majority_end.strftime("%Y%m%d"),
            },
        )
    ]
