"""Validator: FeedContactValidator — checks feed_info.txt contact fields."""

from __future__ import annotations

import polars as pl

from gtfs_validator.context import ValidationContext
from gtfs_validator.notices import Notice, Severity


def _is_blank(col: str) -> pl.Expr:
    """Return a Polars expression that is True when the column value is null or whitespace-only."""
    return pl.col(col).is_null() | (pl.col(col).str.strip_chars() == "")


def _is_blank_or_absent(col: str, df: pl.DataFrame) -> pl.Expr:
    """Return a Polars expression that is True when the column is absent or blank."""
    if col not in df.columns:
        return pl.lit(True)
    return _is_blank(col)


def validate_feed_contact(
    feed: dict[str, pl.DataFrame],
    ctx: ValidationContext,
) -> list[Notice]:
    """Warn when feed_info.txt has neither feed_contact_email nor feed_contact_url."""
    if "feed_info" not in feed or feed["feed_info"].is_empty():
        return []

    df = feed["feed_info"]
    notices: list[Notice] = []

    violating = df.filter(
        _is_blank_or_absent("feed_contact_email", df)
        & _is_blank_or_absent("feed_contact_url", df)
    )

    for row in violating.iter_rows(named=True):
        notices.append(
            Notice(
                code="missing_feed_contact_email_and_url",
                severity=Severity.WARNING,
                fields={"csv_row_number": row["csv_row_number"]},
            )
        )

    return notices
