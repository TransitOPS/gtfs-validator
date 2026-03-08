"""Validator: expired_calendar — warns when all service dates are in the past."""

from __future__ import annotations

import polars as pl

from gtfs_validator.calendar_utils import build_service_date_map
from gtfs_validator.context import ValidationContext
from gtfs_validator.notices import Notice, Severity


def validate_expired_calendar(
    feed: dict[str, pl.DataFrame],
    ctx: ValidationContext,
) -> list[Notice]:
    """Emit a warning for each service ID whose active dates are all in the past."""
    service_date_map = build_service_date_map(feed)

    if not service_date_map:
        return []

    validation_date = ctx.date_for_validation

    # Determine whether calendar.txt is present and non-empty
    calendar_present = "calendar" in feed and not feed["calendar"].is_empty()

    # Identify expired service IDs
    expired_services: dict[str, None] = {}  # ordered set
    non_expired_exists = False

    for service_id, dates in service_date_map.items():
        if not dates:
            continue  # empty set -- skip
        if max(dates) < validation_date:
            expired_services[service_id] = None
        else:
            non_expired_exists = True

    if not expired_services:
        return []

    notices: list[Notice] = []

    if calendar_present:
        # Branch A: calendar.txt is non-empty
        calendar_df = feed["calendar"]
        cal_service_ids = calendar_df["service_id"].to_list()
        cal_row_map: dict[str, int] = {}
        for i, sid in enumerate(cal_service_ids):
            if sid not in cal_row_map:
                cal_row_map[sid] = i + 2  # 0-based index + header + 1-based

        for service_id in expired_services:
            if service_id in cal_row_map:
                notices.append(
                    Notice(
                        code="expired_calendar",
                        severity=Severity.WARNING,
                        fields={
                            "csv_row_number": cal_row_map[service_id],
                            "service_id": service_id,
                        },
                    )
                )
    else:
        # Branch B: calendar.txt is empty (all services from calendar_dates.txt only)
        if non_expired_exists:
            return []

        cd_df = feed["calendar_dates"]
        cd_service_ids = cd_df["service_id"].to_list()
        cd_row_map: dict[str, int] = {}
        for i, sid in enumerate(cd_service_ids):
            row_num = i + 2
            if sid not in cd_row_map or row_num < cd_row_map[sid]:
                cd_row_map[sid] = row_num

        for service_id in expired_services:
            if service_id in cd_row_map:
                notices.append(
                    Notice(
                        code="expired_calendar",
                        severity=Severity.WARNING,
                        fields={
                            "csv_row_number": cd_row_map[service_id],
                            "service_id": service_id,
                        },
                    )
                )

    notices.sort(key=lambda n: n.fields["csv_row_number"])
    return notices
