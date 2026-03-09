"""Validator: MatchingFeedAndAgencyLangValidator.

Compares feed_info.feed_lang against agency.agency_lang for each agency row.
When feed_lang is not "mul", all agencies with a specified agency_lang should
match feed_lang.
"""

from __future__ import annotations

import polars as pl

from gtfs_validator.context import ValidationContext
from gtfs_validator.notices import Notice, Severity


def validate_matching_feed_and_agency_lang(
    feed: dict[str, pl.DataFrame],
    ctx: ValidationContext,
) -> list[Notice]:
    feed_info = feed.get("feed_info")
    if feed_info is None or feed_info.height == 0:
        return []

    agency = feed.get("agency")
    if agency is None or agency.height == 0:
        return []

    # Extract feed_lang from the first (and only) row
    feed_lang_col = feed_info.get_column("feed_lang")
    feed_lang_val = feed_lang_col[0]

    # Guard: feed_lang is null or empty
    if feed_lang_val is None or feed_lang_val == "":
        return []

    # Guard: feed_lang is "mul" (case-insensitive)
    if feed_lang_val.lower() == "mul":
        return []

    # Guard: agency_lang column must exist
    if "agency_lang" not in agency.columns:
        return []

    # Filter agencies where agency_lang is non-null, non-empty,
    # and does not match feed_lang (case-insensitive)
    mismatched = agency.filter(
        pl.col("agency_lang").is_not_null()
        & (pl.col("agency_lang") != "")
        & (pl.col("agency_lang").str.to_lowercase() != feed_lang_val.lower())
    )

    notices: list[Notice] = []
    for row in mismatched.iter_rows(named=True):
        notices.append(
            Notice(
                code="feed_info_lang_and_agency_lang_mismatch",
                severity=Severity.WARNING,
                fields={
                    "csvRowNumber": row["csvRowNumber"],
                    "agencyId": row.get("agency_id", ""),
                    "agencyName": row.get("agency_name", ""),
                    "agencyLang": row["agency_lang"],
                    "feedLang": feed_lang_val,
                },
            )
        )

    return notices
