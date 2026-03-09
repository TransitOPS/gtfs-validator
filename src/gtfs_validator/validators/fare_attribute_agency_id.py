"""Validator: fare_attribute_agency_id.

Checks that every fare_attributes.txt row references an agency_id.
- Multi-agency feed: missing agency_id is an ERROR.
- Single-agency feed: missing agency_id is a WARNING.
"""

from __future__ import annotations

import polars as pl

from gtfs_validator.context import ValidationContext
from gtfs_validator.notices import Notice, Severity


def validate_fare_attribute_agency_id(
    feed: dict[str, pl.DataFrame],
    ctx: ValidationContext,
) -> list[Notice]:
    """Return notices for fare_attributes rows with missing agency_id."""
    # 1. Skip if agency.txt is absent or empty
    if "agency" not in feed or feed["agency"].is_empty():
        return []

    # 2. Skip if fare_attributes.txt is absent or empty
    if "fare_attributes" not in feed or feed["fare_attributes"].is_empty():
        return []

    agency_df = feed["agency"]
    fare_df = feed["fare_attributes"]

    # 3. Count total agencies to determine error vs. warning branch
    total_agencies = len(agency_df)

    # 4. Attach 0-based row index for csv_row_number
    fare_df = fare_df.with_row_index("_row_idx")

    # 5. Filter to rows where agency_id is null or empty string
    missing = fare_df.filter(
        pl.col("agency_id").is_null() | (pl.col("agency_id") == "")
    )

    if missing.is_empty():
        return []

    # 6. Emit one notice per offending row
    notices: list[Notice] = []
    if total_agencies > 1:
        # Multi-agency: missing agency_id is an ERROR
        for row in missing.iter_rows(named=True):
            notices.append(
                Notice(
                    code="missing_required_agency_id",
                    severity=Severity.ERROR,
                    fields={
                        "filename": "fare_attributes.txt",
                        "csv_row_number": row["_row_idx"],
                        "agency_name": None,
                    },
                )
            )
    else:
        # Single agency: missing agency_id is a WARNING
        for row in missing.iter_rows(named=True):
            notices.append(
                Notice(
                    code="missing_recommended_field",
                    severity=Severity.WARNING,
                    fields={
                        "filename": "fare_attributes.txt",
                        "csv_row_number": row["_row_idx"],
                        "field_name": "agency_id",
                    },
                )
            )

    return notices
