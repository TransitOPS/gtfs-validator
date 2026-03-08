"""Validator: warn when fare_media_name is absent for types that require a name."""

from __future__ import annotations

import polars as pl

from gtfs_validator.context import ValidationContext
from gtfs_validator.notices import Notice, Severity


def validate_fare_media_name(
    feed: dict[str, pl.DataFrame],
    ctx: ValidationContext,
) -> list[Notice]:
    """Emit missing_recommended_field for TRANSIT_CARD/MOBILE_APP rows with no name."""
    if "fare_media" not in feed or feed["fare_media"].is_empty():
        return []

    df = feed["fare_media"].with_row_index("_row_idx")

    # Types that require a name: TRANSIT_CARD=2, MOBILE_APP=4
    requires_name = pl.col("fare_media_type").is_in([2, 4])

    # Treat both null and empty string as "absent"
    name_absent = pl.col("fare_media_name").is_null() | (pl.col("fare_media_name") == "")

    violations = df.filter(requires_name & name_absent)

    return [
        Notice(
            code="missing_recommended_field",
            severity=Severity.WARNING,
            fields={
                "filename": "fare_media.txt",
                "csv_row_number": row["_row_idx"],
                "field_name": "fare_media_name",
            },
        )
        for row in violations.iter_rows(named=True)
    ]
