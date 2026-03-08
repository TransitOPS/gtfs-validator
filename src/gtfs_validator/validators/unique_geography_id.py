"""Validator: check that geography IDs are unique across stops, location_groups, and GeoJSON features."""

from __future__ import annotations

from typing import Any

import polars as pl

from gtfs_validator.context import ValidationContext
from gtfs_validator.notices import Notice, Severity


def validate_unique_geography_id(
    feed: dict[str, pl.DataFrame],
    ctx: ValidationContext,
) -> list[Notice]:
    """Check that geography IDs are unique across stops.txt, location_groups.txt, and locations.geojson.

    Emits an ERROR notice when the same ID appears in more than one of:
    - stops.stop_id
    - location_groups.location_group_id
    - locations.geojson feature.id

    Same-source duplicates are ignored; they are handled by load-time PK checks.
    """
    # Collect all ID entries from each source
    entries: list[dict[str, Any]] = []

    # From stops.txt
    stops_df = feed.get("stops")
    if stops_df is not None and not stops_df.is_empty() and "stop_id" in stops_df.columns:
        for row in stops_df.select("stop_id", "csv_row_number").iter_rows(named=True):
            if row["stop_id"] is not None:
                entries.append({"id": row["stop_id"], "source": "stops.txt", "row_number": row["csv_row_number"]})

    # From location_groups.txt
    lg_df = feed.get("location_groups")
    if lg_df is not None and not lg_df.is_empty() and "location_group_id" in lg_df.columns:
        for row in lg_df.select("location_group_id", "csv_row_number").iter_rows(named=True):
            if row["location_group_id"] is not None:
                entries.append({"id": row["location_group_id"], "source": "location_groups.txt", "row_number": row["csv_row_number"]})

    # From locations.geojson (list of dicts, not a DataFrame)
    geojson_features: list[dict[str, Any]] = feed.get("locations_geojson") or []  # type: ignore[assignment]
    for feature in geojson_features:
        fid = feature.get("id")
        if fid is not None:
            entries.append({"id": fid, "source": "locations.geojson", "row_number": feature.get("index")})

    if not entries:
        return []

    # Group by ID and detect cross-source collisions
    id_groups: dict[str, list[dict[str, Any]]] = {}
    for entry in entries:
        id_groups.setdefault(entry["id"], []).append(entry)

    notices: list[Notice] = []
    for geo_id, group in id_groups.items():
        unique_sources = {e["source"] for e in group}
        if len(unique_sources) <= 1:
            continue  # same-source duplicates are handled by load-time PK checks

        # Capture only the first occurrence per source (findFirst semantics)
        csv_row_number_stops: int | None = None
        csv_row_number_location_groups: int | None = None
        feature_index: int | None = None

        for entry in group:
            src = entry["source"]
            if src == "stops.txt" and csv_row_number_stops is None:
                csv_row_number_stops = entry["row_number"]
            elif src == "location_groups.txt" and csv_row_number_location_groups is None:
                csv_row_number_location_groups = entry["row_number"]
            elif src == "locations.geojson" and feature_index is None:
                feature_index = entry["row_number"]

        notices.append(
            Notice(
                code="duplicate_geography_id",
                severity=Severity.ERROR,
                fields={
                    "geography_id": geo_id,
                    "csv_row_number_stops": csv_row_number_stops,
                    "csv_row_number_location_groups": csv_row_number_location_groups,
                    "feature_index": feature_index,
                },
            )
        )

    return notices
