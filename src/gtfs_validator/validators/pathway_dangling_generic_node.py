"""Validator: pathway_dangling_generic_node."""

from __future__ import annotations

import polars as pl

from gtfs_validator.context import ValidationContext
from gtfs_validator.notices import Notice, Severity

_GENERIC_NODE: int = 3


def validate_pathway_dangling_generic_node(
    feed: dict[str, pl.DataFrame],
    ctx: ValidationContext,
) -> list[Notice]:
    """Warn when a GENERIC_NODE stop has exactly one distinct neighbor in the pathway graph."""
    stops = feed.get("stops")
    if stops is None or stops.is_empty():
        return []

    generic_nodes = stops.filter(pl.col("location_type") == _GENERIC_NODE)
    if generic_nodes.is_empty():
        return []

    # Initialise neighbor sets for every GENERIC_NODE stop_id
    neighbors: dict[str, set[str]] = {
        row["stop_id"]: set()
        for row in generic_nodes.iter_rows(named=True)
    }

    # Walk every pathway row and accumulate undirected neighbors
    pathways = feed.get("pathways")
    if pathways is not None and not pathways.is_empty():
        for row in pathways.select(["from_stop_id", "to_stop_id"]).iter_rows(named=True):
            from_id = row["from_stop_id"]
            to_id = row["to_stop_id"]
            if from_id in neighbors:
                neighbors[from_id].add(to_id)
            if to_id in neighbors:
                neighbors[to_id].add(from_id)

    # Emit a notice for each GENERIC_NODE that has exactly one distinct neighbor
    notices: list[Notice] = []
    for row in generic_nodes.iter_rows(named=True):
        if len(neighbors[row["stop_id"]]) == 1:
            notices.append(
                Notice(
                    code="pathway_dangling_generic_node",
                    severity=Severity.WARNING,
                    fields={
                        "csvRowNumber": row["csvRowNumber"],
                        "stopId": row["stop_id"],
                        "stopName": row.get("stop_name"),
                        "parentStation": row.get("parent_station"),
                    },
                )
            )

    return notices
