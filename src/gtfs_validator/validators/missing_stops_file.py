"""Validator: MissingStopsFileValidator."""

from __future__ import annotations

from typing import TYPE_CHECKING

from gtfs_validator.notices import Notice, Severity

if TYPE_CHECKING:
    import polars as pl

    from gtfs_validator.context import ValidationContext


def validate_missing_stops_file(
    feed: dict[str, pl.DataFrame],
    ctx: ValidationContext,
) -> list[Notice]:
    """Emit an error when neither stops.txt nor locations.geojson is present."""
    stops_present = "stops" in feed
    geojson_present = "locations_geojson" in feed
    if not stops_present and not geojson_present:
        return [
            Notice(
                code="missing_required_file",
                severity=Severity.ERROR,
                fields={"filename": "stops.txt"},
            )
        ]
    return []
