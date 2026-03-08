"""Validator: warn when feed_info.txt provides only one of feed_start_date / feed_end_date."""

from __future__ import annotations

import polars as pl

from gtfs_validator.context import ValidationContext
from gtfs_validator.notices import Notice, Severity


def validate_feed_service_date(
    feed: dict[str, pl.DataFrame],
    ctx: ValidationContext,
) -> list[Notice]:
    """Warn when feed_info.txt provides only one of feed_start_date / feed_end_date."""
    if "feed_info" not in feed or feed["feed_info"].is_empty():
        return []

    notices: list[Notice] = []

    for row in feed["feed_info"].iter_rows(named=True):
        has_start = row["feed_start_date"] is not None
        has_end   = row["feed_end_date"]   is not None

        if has_start and not has_end:
            notices.append(
                Notice(
                    code="missing_feed_info_date",
                    severity=Severity.WARNING,
                    fields={
                        "csv_row_number": row["csv_row_number"],
                        "field_name": "feed_end_date",
                    },
                )
            )
        elif not has_start and has_end:
            notices.append(
                Notice(
                    code="missing_feed_info_date",
                    severity=Severity.WARNING,
                    fields={
                        "csv_row_number": row["csv_row_number"],
                        "field_name": "feed_start_date",
                    },
                )
            )

    return notices
