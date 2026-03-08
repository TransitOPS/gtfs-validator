"""Validator: pathway_endpoint_type."""

from __future__ import annotations

import polars as pl

from gtfs_validator.context import ValidationContext
from gtfs_validator.notices import Notice, Severity

_STATION: int = 1
_STOP: int = 0


def validate_pathway_endpoint_type(
    feed: dict[str, pl.DataFrame],
    ctx: ValidationContext,
) -> list[Notice]:
    """Error when a pathway endpoint is a STATION or a STOP that has boarding-area children."""
    pathways = feed.get("pathways")
    if pathways is None or pathways.is_empty():
        return []

    stops = feed.get("stops")
    if stops is None or stops.is_empty():
        return []

    # Build stop_id -> location_type mapping (None when location_type is null)
    stop_type: dict[str, int | None] = {}
    for row in stops.select(["stop_id", "location_type"]).iter_rows(named=True):
        stop_type[row["stop_id"]] = row["location_type"]

    # Build set of stop_ids that appear as a parent_station of any child
    parents_with_children: set[str] = set(
        stops.filter(pl.col("parent_station").is_not_null())["parent_station"].to_list()
    )

    notices: list[Notice] = []
    for row in pathways.select(
        ["csv_row_number", "pathway_id", "from_stop_id", "to_stop_id"]
    ).iter_rows(named=True):
        for field_name, stop_id in [
            ("from_stop_id", row["from_stop_id"]),
            ("to_stop_id", row["to_stop_id"]),
        ]:
            loc_type = stop_type.get(stop_id)
            if loc_type is None:
                continue  # broken FK — skip silently
            if loc_type == _STATION:
                notices.append(
                    Notice(
                        code="pathway_to_wrong_location_type",
                        severity=Severity.ERROR,
                        fields={
                            "csvRowNumber": row["csv_row_number"],
                            "pathwayId": row["pathway_id"],
                            "fieldName": field_name,
                            "stopId": stop_id,
                        },
                    )
                )
            elif loc_type == _STOP and stop_id in parents_with_children:
                notices.append(
                    Notice(
                        code="pathway_to_platform_with_boarding_areas",
                        severity=Severity.ERROR,
                        fields={
                            "csvRowNumber": row["csv_row_number"],
                            "pathwayId": row["pathway_id"],
                            "fieldName": field_name,
                            "stopId": stop_id,
                        },
                    )
                )

    return notices
