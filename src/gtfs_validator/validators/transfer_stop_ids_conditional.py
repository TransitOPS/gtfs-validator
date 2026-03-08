"""Validator: conditionally required stop IDs on transfers."""

from __future__ import annotations

import polars as pl

from gtfs_validator.context import ValidationContext
from gtfs_validator.notices import Notice, Severity


def validate_transfer_stop_ids_conditional(
    feed: dict[str, pl.DataFrame],
    ctx: ValidationContext,
) -> list[Notice]:
    """Emit missing_required_field notices for transfers missing from_stop_id or
    to_stop_id when transfer_type is present and not an in-seat type (4 or 5)."""

    if "transfers" not in feed:
        return []

    transfers = feed["transfers"]

    if transfers.is_empty():
        return []

    required_cols = {"transfer_type", "from_stop_id", "to_stop_id", "csv_row_number"}
    if not required_cols.issubset(transfers.columns):
        return []

    # Filter once to rows subject to the stop ID requirement:
    # transfer_type must be non-null AND not an in-seat type (4 or 5).
    non_in_seat = transfers.filter(
        pl.col("transfer_type").is_not_null()
        & ~pl.col("transfer_type").is_in([4, 5])
    )

    notices: list[Notice] = []

    # Check each stop ID field independently.
    # from_stop_id is checked before to_stop_id to preserve Java field order.
    for field_name in ("from_stop_id", "to_stop_id"):
        missing = non_in_seat.filter(pl.col(field_name).is_null())
        for row in missing.select(["csv_row_number"]).iter_rows(named=True):
            notices.append(
                Notice(
                    code="missing_required_field",
                    severity=Severity.ERROR,
                    fields={
                        "filename": "transfers.txt",
                        "csv_row_number": row["csv_row_number"],
                        "field_name": field_name,
                    },
                )
            )

    return notices
