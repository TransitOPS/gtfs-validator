"""Booking rules entity validator — row-level checks on booking_rules.txt."""

from __future__ import annotations

import polars as pl

from gtfs_validator.context import ValidationContext
from gtfs_validator.notices import Notice, Severity

# Booking type enum values
REALTIME = 0
SAMEDAY = 1
PRIORDAY = 2

# Fields forbidden per booking type
_REALTIME_FORBIDDEN = [
    "prior_notice_duration_min",
    "prior_notice_duration_max",
    "prior_notice_last_day",
    "prior_notice_last_time",
    "prior_notice_start_day",
    "prior_notice_start_time",
    "prior_notice_service_id",
]

_SAMEDAY_FORBIDDEN = [
    "prior_notice_last_day",
    "prior_notice_last_time",
    "prior_notice_service_id",
]

_PRIORDAY_FORBIDDEN = [
    "prior_notice_duration_min",
    "prior_notice_duration_max",
]


def _check_forbidden_fields(
    row: dict[str, object], forbidden_fields: list[str]
) -> list[str]:
    """Return list of field names that are non-null in the row."""
    return [f for f in forbidden_fields if row[f] is not None]


def validate_booking_rules_entity(
    feed: dict[str, pl.DataFrame],
    ctx: ValidationContext,
) -> list[Notice]:
    """Validate booking_rules rows for field-presence constraints by booking_type."""
    if "booking_rules" not in feed:
        return []
    df = feed["booking_rules"]
    if df.is_empty():
        return []

    notices: list[Notice] = []

    for row in df.iter_rows(named=True):
        csv_row_number = row["csv_row_number"]
        booking_rule_id = row["booking_rule_id"]
        booking_type = row["booking_type"]

        # --- Type-specific forbidden-field checks ---
        if booking_type == REALTIME:
            offending = _check_forbidden_fields(row, _REALTIME_FORBIDDEN)
            if offending:
                notices.append(Notice(
                    code="forbidden_real_time_booking_field_value",
                    severity=Severity.ERROR,
                    fields={
                        "csv_row_number": csv_row_number,
                        "booking_rule_id": booking_rule_id,
                        "field_names": ",".join(offending),
                    },
                ))

        elif booking_type == SAMEDAY:
            offending = _check_forbidden_fields(row, _SAMEDAY_FORBIDDEN)
            if offending:
                notices.append(Notice(
                    code="forbidden_same_day_booking_field_value",
                    severity=Severity.ERROR,
                    fields={
                        "csv_row_number": csv_row_number,
                        "booking_rule_id": booking_rule_id,
                        "field_names": ",".join(offending),
                    },
                ))

            # Missing prior_notice_duration_min for SAMEDAY
            if row["prior_notice_duration_min"] is None:
                notices.append(Notice(
                    code="missing_prior_notice_duration_min",
                    severity=Severity.ERROR,
                    fields={
                        "csv_row_number": csv_row_number,
                        "booking_rule_id": booking_rule_id,
                    },
                ))

        elif booking_type == PRIORDAY:
            offending = _check_forbidden_fields(row, _PRIORDAY_FORBIDDEN)
            if offending:
                notices.append(Notice(
                    code="forbidden_prior_day_booking_field_value",
                    severity=Severity.ERROR,
                    fields={
                        "csv_row_number": csv_row_number,
                        "booking_rule_id": booking_rule_id,
                        "field_names": ",".join(offending),
                    },
                ))

            # Missing required PRIORDAY fields
            if row["prior_notice_last_day"] is None:
                notices.append(Notice(
                    code="missing_prior_notice_last_day",
                    severity=Severity.ERROR,
                    fields={
                        "csv_row_number": csv_row_number,
                        "booking_rule_id": booking_rule_id,
                    },
                ))
            if row["prior_notice_last_time"] is None:
                notices.append(Notice(
                    code="missing_prior_notice_last_time",
                    severity=Severity.ERROR,
                    fields={
                        "csv_row_number": csv_row_number,
                        "booking_rule_id": booking_rule_id,
                    },
                ))

        # --- Type-independent checks (run for ALL booking_type values) ---

        # Duration min vs max range check
        dur_min = row["prior_notice_duration_min"]
        dur_max = row["prior_notice_duration_max"]
        if dur_min is not None and dur_max is not None:
            if dur_max < dur_min:
                notices.append(Notice(
                    code="invalid_prior_notice_duration_min",
                    severity=Severity.ERROR,
                    fields={
                        "csv_row_number": csv_row_number,
                        "booking_rule_id": booking_rule_id,
                        "prior_notice_duration_min": dur_min,
                        "prior_notice_duration_max": dur_max,
                    },
                ))

        # start_day forbidden with duration_max
        start_day = row["prior_notice_start_day"]
        if dur_max is not None and start_day is not None:
            notices.append(Notice(
                code="forbidden_prior_notice_start_day",
                severity=Severity.ERROR,
                fields={
                    "csv_row_number": csv_row_number,
                    "booking_rule_id": booking_rule_id,
                    "prior_notice_start_day": start_day,
                    "prior_notice_duration_max": dur_max,
                },
            ))

        # last_day vs start_day range
        last_day = row["prior_notice_last_day"]
        if last_day is not None and start_day is not None:
            if last_day > start_day:
                notices.append(Notice(
                    code="prior_notice_last_day_after_start_day",
                    severity=Severity.ERROR,
                    fields={
                        "csv_row_number": csv_row_number,
                        "prior_notice_last_day": last_day,
                        "prior_notice_start_day": start_day,
                    },
                ))

        # start_time / start_day co-dependency
        start_time = row["prior_notice_start_time"]
        has_start_time = start_time is not None
        has_start_day = start_day is not None

        if has_start_time and not has_start_day:
            notices.append(Notice(
                code="forbidden_prior_notice_start_time",
                severity=Severity.ERROR,
                fields={
                    "csv_row_number": csv_row_number,
                    "booking_rule_id": booking_rule_id,
                    "prior_notice_start_time": str(start_time),
                },
            ))

        if has_start_day and not has_start_time:
            notices.append(Notice(
                code="missing_prior_notice_start_time",
                severity=Severity.ERROR,
                fields={
                    "csv_row_number": csv_row_number,
                    "booking_rule_id": booking_rule_id,
                    "prior_notice_start_day": start_day,
                },
            ))

    return notices
