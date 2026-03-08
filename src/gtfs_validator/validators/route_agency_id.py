"""Validator: route_agency_id.

Checks that every routes.txt row references an agency_id.
- Multi-agency feed: missing agency_id is an ERROR.
- Single-agency feed: missing agency_id is a WARNING.
"""

from __future__ import annotations

import polars as pl

from gtfs_validator.context import ValidationContext
from gtfs_validator.notices import Notice, Severity


def validate_route_agency_id(
    feed: dict[str, pl.DataFrame],
    ctx: ValidationContext,
) -> list[Notice]:
    """Return notices for routes rows with missing agency_id."""
    # 1. Skip if agency.txt is absent or empty
    if "agency" not in feed or feed["agency"].is_empty():
        return []

    # 2. Skip if routes.txt is absent or empty
    if "routes" not in feed or feed["routes"].is_empty():
        return []

    agency_df = feed["agency"]
    routes_df = feed["routes"]

    # 3. Count total agencies to determine error vs. warning branch
    total_agencies = len(agency_df)

    # 4. Filter to rows where agency_id is null or empty string
    missing = routes_df.filter(
        pl.col("agency_id").is_null() | (pl.col("agency_id") == "")
    )

    if missing.is_empty():
        return []

    # 5. Emit one notice per offending row
    notices: list[Notice] = []
    if total_agencies > 1:
        # Multi-agency: missing agency_id is an ERROR
        for row in missing.iter_rows(named=True):
            notices.append(
                Notice(
                    code="missing_required_agency_id",
                    severity=Severity.ERROR,
                    fields={
                        "filename": "routes.txt",
                        "csv_row_number": row["csv_row_number"],
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
                        "filename": "routes.txt",
                        "csv_row_number": row["csv_row_number"],
                        "field_name": "agency_id",
                    },
                )
            )

    return notices
