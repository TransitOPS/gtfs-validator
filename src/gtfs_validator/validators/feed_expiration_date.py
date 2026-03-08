"""Validator: feed expiration date (7-day and 30-day warnings)."""

from __future__ import annotations

import datetime
from datetime import timedelta

import polars as pl

from gtfs_validator.context import ValidationContext
from gtfs_validator.notices import Notice, Severity


def validate_feed_expiration_date(
    feed: dict[str, pl.DataFrame],
    ctx: ValidationContext,
) -> list[Notice]:
    """Emit warnings when feed_end_date is near or past expiration.

    Emits ``feed_expiration_date_7_days`` when the feed expires in fewer
    than 7 days (or has already expired), and ``feed_expiration_date_30_days``
    when it expires in 7–29 days inclusive.  The two notices are mutually
    exclusive per row.
    """
    if "feed_info" not in feed or feed["feed_info"].is_empty():
        return []

    current_date = ctx.date_for_validation
    plus_7 = current_date + timedelta(days=7)
    plus_30 = current_date + timedelta(days=30)

    df = feed["feed_info"].filter(pl.col("feed_end_date").is_not_null())

    def _parse_date(value: datetime.date | str) -> datetime.date:
        if isinstance(value, datetime.date):
            return value
        return datetime.datetime.strptime(value, "%Y%m%d").date()

    notices: list[Notice] = []
    for row in df.iter_rows(named=True):
        end_date = _parse_date(row["feed_end_date"])  # parse YYYYMMDD string or date
        row_number = row["csv_row_number"]  # int

        if end_date < plus_7:
            notices.append(
                Notice(
                    code="feed_expiration_date_7_days",
                    severity=Severity.WARNING,
                    fields={
                        "csv_row_number": row_number,
                        "current_date": current_date.strftime("%Y%m%d"),
                        "feed_end_date": end_date.strftime("%Y%m%d"),
                        "suggested_expiration_date": plus_7.strftime("%Y%m%d"),
                    },
                )
            )
            continue  # 30-day check is skipped for this row

        if end_date < plus_30:
            notices.append(
                Notice(
                    code="feed_expiration_date_30_days",
                    severity=Severity.WARNING,
                    fields={
                        "csv_row_number": row_number,
                        "current_date": current_date.strftime("%Y%m%d"),
                        "feed_end_date": end_date.strftime("%Y%m%d"),
                        "suggested_expiration_date": plus_30.strftime("%Y%m%d"),
                    },
                )
            )

    return notices
