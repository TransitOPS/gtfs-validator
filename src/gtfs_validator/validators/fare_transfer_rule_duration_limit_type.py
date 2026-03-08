"""Validator: fare_transfer_rule_duration_limit_type.

Checks that duration_limit and duration_limit_type are always provided
together in fare_transfer_rules.txt — each is meaningless without the other.
"""

from __future__ import annotations

import polars as pl

from gtfs_validator.context import ValidationContext
from gtfs_validator.notices import Notice, Severity


def validate_fare_transfer_rule_duration_limit_type(
    feed: dict[str, pl.DataFrame],
    ctx: ValidationContext,
) -> list[Notice]:
    """Validate that duration_limit and duration_limit_type appear together.

    Emits an ERROR for any row that has one field set but not the other.
    """
    if "fare_transfer_rules" not in feed or feed["fare_transfer_rules"].is_empty():
        return []

    df = feed["fare_transfer_rules"]
    notices: list[Notice] = []

    # Check 1: duration_limit present, duration_limit_type absent.
    has_limit_no_type = df.filter(
        pl.col("duration_limit").is_not_null() & pl.col("duration_limit_type").is_null()
    )
    for row in has_limit_no_type.iter_rows(named=True):
        notices.append(
            Notice(
                code="fare_transfer_rule_duration_limit_without_type",
                severity=Severity.ERROR,
                fields={"csv_row_number": row["csv_row_number"]},
            )
        )

    # Check 2: duration_limit_type present, duration_limit absent.
    has_type_no_limit = df.filter(
        pl.col("duration_limit_type").is_not_null() & pl.col("duration_limit").is_null()
    )
    for row in has_type_no_limit.iter_rows(named=True):
        notices.append(
            Notice(
                code="fare_transfer_rule_duration_limit_type_without_duration_limit",
                severity=Severity.ERROR,
                fields={"csv_row_number": row["csv_row_number"]},
            )
        )

    return notices
