"""Validator: trips sharing a block_id must have consistent route_type."""

from __future__ import annotations

import polars as pl

from gtfs_validator.context import ValidationContext
from gtfs_validator.notices import Notice, Severity


def validate_inconsistent_route_type_for_block_id(
    feed: dict[str, pl.DataFrame],
    ctx: ValidationContext,
) -> list[Notice]:
    trips = feed.get("trips")
    routes = feed.get("routes")
    if trips is None or routes is None:
        return []
    if trips.is_empty() or routes.is_empty():
        return []
    if "block_id" not in trips.columns:
        return []

    # Join trips with routes to attach route_type
    joined = trips.select("block_id", "route_id").join(
        routes.select("route_id", "route_type"),
        on="route_id",
        how="inner",
    )

    # Filter out rows where block_id is null or empty
    filtered = joined.filter(
        pl.col("block_id").is_not_null() & (pl.col("block_id") != "")
    )

    if filtered.is_empty():
        return []

    # Group by block_id and aggregate distinct route info
    agg = filtered.group_by("block_id").agg(
        pl.col("route_type").n_unique().alias("n_types"),
        pl.col("route_id").unique().alias("route_ids"),
        pl.col("route_type").unique().alias("route_types"),
    )

    # Keep only blocks with more than one distinct route_type
    bad = agg.filter(pl.col("n_types") > 1)

    # Emit one notice per inconsistent block
    notices = []
    for row in bad.iter_rows(named=True):
        notices.append(
            Notice(
                code="inconsistent_route_type_for_block_id",
                severity=Severity.WARNING,
                fields={
                    "block_id": row["block_id"],
                    "route_ids": ", ".join(
                        str(r) for r in sorted(row["route_ids"])
                    ),
                    "route_types": ", ".join(
                        str(t) for t in sorted(row["route_types"])
                    ),
                },
            )
        )
    return notices
