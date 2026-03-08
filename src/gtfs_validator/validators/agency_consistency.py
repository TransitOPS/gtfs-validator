"""Agency consistency validator.

Checks agency_id presence rules (single vs multi-agency),
timezone consistency, and language consistency across all agencies.
"""

from __future__ import annotations

import polars as pl

from gtfs_validator.context import ValidationContext
from gtfs_validator.notices import Notice, Severity


def validate_agency_consistency(
    feed: dict[str, pl.DataFrame],
    ctx: ValidationContext,
) -> list[Notice]:
    """Validate agency-level consistency rules."""
    notices: list[Notice] = []

    # Guard: skip if agency table missing or empty
    if "agency" not in feed:
        return notices
    df = feed["agency"]
    if df.is_empty():
        return notices

    row_count = len(df)

    # --- Single-agency path ---
    if row_count == 1:
        val = df["agency_id"][0]
        if val is None or val == "":
            notices.append(Notice(
                code="missing_recommended_field",
                severity=Severity.WARNING,
                fields={
                    "filename": "agency.txt",
                    "csv_row_number": 1,
                    "field_name": "agency_id",
                },
            ))
        return notices

    # --- Multi-agency path ---

    # Add a 0-based row index column for notice reporting
    df = df.with_row_index("_row_idx")

    # Step 2a: Required agency_id check
    missing_id = df.filter(
        pl.col("agency_id").is_null() | (pl.col("agency_id") == "")
    )
    for row in missing_id.iter_rows(named=True):
        notices.append(Notice(
            code="missing_required_agency_id",
            severity=Severity.ERROR,
            fields={
                "filename": "agency.txt",
                "csv_row_number": row["_row_idx"],
                "agency_name": row["agency_name"],
            },
        ))

    # Step 2b: Timezone consistency
    expected_tz = df["agency_timezone"][0]
    tz_mismatch = df.filter(
        (pl.col("_row_idx") > 0)
        & (pl.col("agency_timezone") != expected_tz)
    )
    for row in tz_mismatch.iter_rows(named=True):
        notices.append(Notice(
            code="inconsistent_agency_timezone",
            severity=Severity.ERROR,
            fields={
                "csv_row_number": row["_row_idx"],
                "expected": expected_tz,
                "actual": row["agency_timezone"],
            },
        ))

    # Step 2c: Language consistency
    non_null_lang = df.filter(
        pl.col("agency_lang").is_not_null()
        & (pl.col("agency_lang") != "")
    )
    if len(non_null_lang) > 0:
        expected_lang = non_null_lang["agency_lang"][0]
        lang_mismatch = non_null_lang.filter(
            pl.col("agency_lang") != expected_lang
        )
        for row in lang_mismatch.iter_rows(named=True):
            notices.append(Notice(
                code="inconsistent_agency_lang",
                severity=Severity.WARNING,
                fields={
                    "csv_row_number": row["_row_idx"],
                    "expected": expected_lang,
                    "actual": row["agency_lang"],
                },
            ))

    return notices
