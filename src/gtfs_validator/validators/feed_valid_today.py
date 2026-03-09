"""Validator: check whether the feed covers the current date."""

from __future__ import annotations

import datetime
import polars as pl

from gtfs_validator.context import ValidationContext
from gtfs_validator.notices import Notice, Severity


def _parse_date(value: datetime.date | str) -> datetime.date:
    """Return a datetime.date from either a date object or a YYYYMMDD string."""
    if isinstance(value, datetime.date):
        return value
    return datetime.datetime.strptime(value, "%Y%m%d").date()


def validate_feed_valid_today(
    feed: dict[str, pl.DataFrame],
    ctx: ValidationContext,
) -> list[Notice]:
    """Emit info notice when the feed covers only future service."""
    if "feed_info" not in feed or feed["feed_info"].is_empty():
        return []

    df = feed["feed_info"]

    if "feed_start_date" not in df.columns:
        return []

    min_start_raw = df.select(pl.col("feed_start_date").min()).item()

    if min_start_raw is None:
        return []

    min_start = _parse_date(min_start_raw)
    current_date = ctx.date_for_validation

    if min_start > current_date:
        return [
            Notice(
                code="future_feed",
                severity=Severity.INFO,
                fields={
                    "feed_start_date": min_start.strftime("%Y%m%d"),
                    "current_date": current_date.strftime("%Y%m%d"),
                },
            )
        ]

    return []
