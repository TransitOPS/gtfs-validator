"""Validator: FareTransferRuleTransferCountValidator."""

from __future__ import annotations

import polars as pl

from gtfs_validator.context import ValidationContext
from gtfs_validator.notices import Notice, Severity


def validate_fare_transfer_rule_transfer_count(
    feed: dict[str, pl.DataFrame],
    ctx: ValidationContext,
) -> list[Notice]:
    """Validate transfer_count constraints on fare_transfer_rules rows.

    - When from_leg_group_id == to_leg_group_id (self-loop), transfer_count is
      required and must be -1 or >= 1.
    - When the two leg group IDs differ (or either is absent), transfer_count
      must be absent.
    """
    if "fare_transfer_rules" not in feed or feed["fare_transfer_rules"].is_empty():
        return []

    df = feed["fare_transfer_rules"]
    notices: list[Notice] = []

    # --- Branch A: both leg group IDs are present and equal (self-loop) ---
    self_loop = df.filter(
        pl.col("from_leg_group_id").is_not_null()
        & pl.col("to_leg_group_id").is_not_null()
        & (pl.col("from_leg_group_id") == pl.col("to_leg_group_id"))
    )

    # Branch A, sub-case 1: transfer_count present but invalid (< -1 or == 0)
    invalid_count = self_loop.filter(
        pl.col("transfer_count").is_not_null()
        & ((pl.col("transfer_count") < -1) | (pl.col("transfer_count") == 0))
    )
    for row in invalid_count.iter_rows(named=True):
        notices.append(
            Notice(
                code="fare_transfer_rule_invalid_transfer_count",
                severity=Severity.ERROR,
                fields={
                    "csv_row_number": row["csv_row_number"],
                    "transfer_count": row["transfer_count"],
                },
            )
        )

    # Branch A, sub-case 2: transfer_count absent (required when IDs are equal)
    missing_count = self_loop.filter(pl.col("transfer_count").is_null())
    for row in missing_count.iter_rows(named=True):
        notices.append(
            Notice(
                code="fare_transfer_rule_without_transfer_count",
                severity=Severity.ERROR,
                fields={"csv_row_number": row["csv_row_number"]},
            )
        )

    # --- Branch B: leg group IDs not equal, or at least one absent ---
    not_self_loop = df.filter(
        pl.col("from_leg_group_id").is_null()
        | pl.col("to_leg_group_id").is_null()
        | (pl.col("from_leg_group_id") != pl.col("to_leg_group_id"))
    )

    # Branch B: transfer_count present (forbidden in this case)
    forbidden_count = not_self_loop.filter(pl.col("transfer_count").is_not_null())
    for row in forbidden_count.iter_rows(named=True):
        notices.append(
            Notice(
                code="fare_transfer_rule_with_forbidden_transfer_count",
                severity=Severity.ERROR,
                fields={"csv_row_number": row["csv_row_number"]},
            )
        )

    return notices
