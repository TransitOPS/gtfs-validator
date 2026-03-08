"""Validator: PickupBookingRuleIdValidator.

Detects stop_times rows where pickup_type or drop_off_type is 2 (MUST_PHONE)
and a time window is set, but the corresponding booking rule ID is missing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from gtfs_validator.notices import Notice, Severity

if TYPE_CHECKING:
    from gtfs_validator.context import ValidationContext


def validate_pickup_booking_rule_id(
    feed: dict[str, pl.DataFrame],
    ctx: ValidationContext,
) -> list[Notice]:
    """Emit missing_pickup_drop_off_booking_rule_id (WARNING) for qualifying stop_times rows."""
    if "booking_rules" not in feed:
        return []
    if "stop_times" not in feed:
        return []

    df = feed["stop_times"]
    cols = set(df.columns)

    # Guard: at least one of pickup_type / drop_off_type must exist
    if "pickup_type" not in cols and "drop_off_type" not in cols:
        return []

    notices: list[Notice] = []

    # Sub-check A: missing pickup_booking_rule_id
    if "pickup_type" in cols:
        # Window column absent → treat as all-null → no rows match
        if "start_pickup_drop_off_window" in cols:
            pickup_violations = df.filter(
                pl.col("pickup_type").eq(2)
                & pl.col("start_pickup_drop_off_window").is_not_null()
                & (
                    pl.col("pickup_booking_rule_id").is_null()
                    if "pickup_booking_rule_id" in cols
                    else pl.lit(True)
                )
            )
            for row in pickup_violations.iter_rows(named=True):
                notices.append(
                    Notice(
                        code="missing_pickup_drop_off_booking_rule_id",
                        severity=Severity.WARNING,
                        fields={
                            "csv_row_number": row["csv_row_number"],
                            "pickup_type": 2,
                            "drop_off_type": row.get("drop_off_type"),
                        },
                    )
                )

    # Sub-check B: missing drop_off_booking_rule_id
    if "drop_off_type" in cols:
        if "end_pickup_drop_off_window" in cols:
            dropoff_violations = df.filter(
                pl.col("drop_off_type").eq(2)
                & pl.col("end_pickup_drop_off_window").is_not_null()
                & (
                    pl.col("drop_off_booking_rule_id").is_null()
                    if "drop_off_booking_rule_id" in cols
                    else pl.lit(True)
                )
            )
            for row in dropoff_violations.iter_rows(named=True):
                notices.append(
                    Notice(
                        code="missing_pickup_drop_off_booking_rule_id",
                        severity=Severity.WARNING,
                        fields={
                            "csv_row_number": row["csv_row_number"],
                            "pickup_type": row.get("pickup_type"),
                            "drop_off_type": 2,
                        },
                    )
                )

    return notices
