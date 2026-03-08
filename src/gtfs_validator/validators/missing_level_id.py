"""Validator: MissingLevelIdValidator."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from gtfs_validator.notices import Notice, Severity

if TYPE_CHECKING:
    from gtfs_validator.context import ValidationContext


def validate_missing_level_id(
    feed: dict[str, pl.DataFrame],
    ctx: ValidationContext,
) -> list[Notice]:
    """Emit a notice for stops that serve as elevator pathway endpoints but lack level_id.

    Elevator pathways have pathway_mode == 5. Each stop that appears as either
    from_stop_id or to_stop_id of an elevator pathway must have level_id defined.
    One notice is emitted per distinct stop, regardless of how many elevator pathways
    reference it.
    """
    if "pathways" not in feed or "stops" not in feed:
        return []

    pathways = feed["pathways"]
    stops = feed["stops"]

    # Step 1: filter to elevator pathways only (pathway_mode == 5)
    elevators = pathways.filter(pl.col("pathway_mode") == 5)
    if elevators.is_empty():
        return []

    # Step 2: collect unique stop IDs from both endpoint columns, deduplicated
    from_ids = elevators.select(pl.col("from_stop_id").alias("stop_id"))
    to_ids = elevators.select(pl.col("to_stop_id").alias("stop_id"))
    elevator_stop_ids = pl.concat([from_ids, to_ids]).unique()

    # Step 3: determine which stop columns to select (level_id may be absent)
    has_level_id = "level_id" in stops.columns
    has_stop_name = "stop_name" in stops.columns

    stop_cols = ["stop_id", "csvRowNumber"]
    if has_level_id:
        stop_cols.append("level_id")
    if has_stop_name:
        stop_cols.append("stop_name")

    # Step 4: inner-join against stops to get stop metadata
    # (stops not in stops.txt are silently skipped — dangling FK is
    #  handled by the FK validator separately)
    joined = elevator_stop_ids.join(
        stops.select(stop_cols),
        on="stop_id",
        how="inner",
    )

    # Step 5: keep only stops where level_id is null (or column absent entirely)
    if has_level_id:
        missing = joined.filter(
            pl.col("level_id").is_null() | (pl.col("level_id") == "")
        )
    else:
        # level_id column entirely absent — all joined stops are missing it
        missing = joined

    # Step 6: emit one notice per remaining row
    notices: list[Notice] = []
    for row in missing.iter_rows(named=True):
        stop_name = row.get("stop_name") if has_stop_name else None
        notices.append(
            Notice(
                code="missing_level_id",
                severity=Severity.ERROR,
                fields={
                    "csvRowNumber": row["csvRowNumber"],
                    "stopId": row["stop_id"],
                    "stopName": stop_name,
                },
            )
        )
    return notices
