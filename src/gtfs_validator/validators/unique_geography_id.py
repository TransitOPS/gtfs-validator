"""Validator: check that geography IDs are unique across stops, location_groups, and GeoJSON features."""

from __future__ import annotations

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
    """
    # Collect all ID entries from each table
    entries: list[dict] = []
    
    # From stops.txt
    if "stops" in feed and not feed["stops"].is_empty():
        stops = feed["stops"]
        if "stop_id" in stops.columns:
            for row in stops.select("stop_id", "csv_row_number").iter_rows(named=True):
                if row["stop_id"] is not None:
                    entries.append({
                        "id": row["stop_id"],
                        "source": "stops.txt",
                        "row_number": row["csv_row_number"],
                    })
    
    # From location_groups.txt
    if "location_groups" in feed and not feed["location_groups"].is_empty():
        loc_groups = feed["location_groups"]
        if "location_group_id" in loc_groups.columns:
            for row in loc_groups.select("location_group_id", "csv_row_number").iter_rows(named=True):
                if row["location_group_id"] is not None:
                    entries.append({
                        "id": row["location_group_id"],
                        "source": "location_groups.txt",
                        "row_number": row["csv_row_number"],
                    })
    
    # From locations.geojson (if present)
    if "geojson_features" in feed and not feed["geojson_features"].is_empty():
        features = feed["geojson_features"]
        if "feature_id" in features.columns and "feature_index" in features.columns:
            for row in features.select("feature_id", "feature_index").iter_rows(named=True):
                if row["feature_id"] is not None:
                    entries.append({
                        "id": row["feature_id"],
                        "source": "locations.geojson",
                        "row_number": row["feature_index"],
                    })
    
    if not entries:
        return []
    
    # Group by ID and check for duplicates across different sources
    notices: list[Notice] = []
    id_groups: dict[str, list[dict]] = {}
    
    for entry in entries:
        id_val = entry["id"]
        if id_val not in id_groups:
            id_groups[id_val] = []
        id_groups[id_val].append(entry)
    
    for geo_id, group in id_groups.items():
        # Check if this ID appears in more than one source
        sources = {e["source"] for e in group}
        if len(sources) > 1:
            # Get row numbers for each source (may be None if source not present)
            csv_row_number_stops = None
            csv_row_number_location_groups = None
            feature_index = None
            
            for entry in group:
                if entry["source"] == "stops.txt":
                    csv_row_number_stops = entry["row_number"]
                elif entry["source"] == "location_groups.txt":
                    csv_row_number_location_groups = entry["row_number"]
                elif entry["source"] == "locations.geojson":
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