"""Validator: MissingFeedInfoValidator."""

from __future__ import annotations

from typing import TYPE_CHECKING

from gtfs_validator.notices import Notice, Severity

if TYPE_CHECKING:
    import polars as pl

    from gtfs_validator.context import ValidationContext


def validate_missing_feed_info(
    feed: dict[str, pl.DataFrame],
    ctx: ValidationContext,
) -> list[Notice]:
    """Emit a notice when feed_info.txt is absent.

    If translations.txt is also absent, emits a WARNING (recommended file).
    If translations.txt is present, emits an ERROR (required because
    translations depend on feed_lang from feed_info.txt).
    """
    if "feed_info" in feed:
        return []

    if "translations" not in feed:
        return [
            Notice(
                code="missing_recommended_file",
                severity=Severity.WARNING,
                fields={"filename": "feed_info.txt"},
            )
        ]
    else:
        return [
            Notice(
                code="missing_required_file",
                severity=Severity.ERROR,
                fields={"filename": "feed_info.txt"},
            )
        ]
