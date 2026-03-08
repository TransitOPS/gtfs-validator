"""Validator: pathway_stop_access."""

from __future__ import annotations

import polars as pl

from gtfs_validator.context import ValidationContext
from gtfs_validator.notices import Notice, Severity


def validate_pathway_stop_access(
    feed: dict[str, pl.DataFrame],
    ctx: ValidationContext,
) -> list[Notice]:
    """Error when a pathway endpoint references a stop with access outside of station pathways."""
    if "stops" not in feed:
        return []

    stops = feed["stops"]
    if "stop_access" not in stops.columns:
        return []

    if "pathways" not in feed or feed["pathways"].is_empty():
        return []

    pathways = feed["pathways"]

    # Phase 1 — Build the external-access stop map
    external_df = stops.filter(pl.col("stop_access") == 1).select(
        ["stop_id", "platform_code"]
    )

    if external_df.is_empty():
        return []

    external_map: dict[str, str | None] = dict(
        zip(
            external_df["stop_id"].to_list(),
            external_df["platform_code"].to_list(),
        )
    )

    # Phase 2 — Scan pathway rows
    notices: list[Notice] = []
    for row in pathways.select(
        ["csv_row_number", "pathway_id", "from_stop_id", "to_stop_id"]
    ).iter_rows(named=True):
        emitted: set[str] = set()

        for field_name in ("from_stop_id", "to_stop_id"):
            stop_id = row[field_name]
            if stop_id is None or stop_id == "":
                continue
            if stop_id not in external_map:
                continue
            if stop_id in emitted:
                continue  # dedup: same stop as both endpoints
            emitted.add(stop_id)
            notices.append(
                Notice(
                    code="pathway_to_stop_with_access_outside_of_station_pathways",
                    severity=Severity.ERROR,
                    fields={
                        "csvRowNumber": row["csv_row_number"],
                        "platformCode": external_map[stop_id],
                        "pathwayId": row["pathway_id"],
                        "stopId": stop_id,
                    },
                )
            )

    return notices
