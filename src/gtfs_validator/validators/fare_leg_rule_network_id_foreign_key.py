"""Validate fare_leg_rules.network_id references routes or networks."""

from __future__ import annotations

import polars as pl

from gtfs_validator.context import ValidationContext
from gtfs_validator.notices import Notice, Severity


def validate_fare_leg_rule_network_id_foreign_key(
    feed: dict[str, pl.DataFrame],
    ctx: ValidationContext,
) -> list[Notice]:
    fare_leg_rules = feed.get("fare_leg_rules")
    if fare_leg_rules is None or fare_leg_rules.is_empty():
        return []

    if "network_id" not in fare_leg_rules.columns:
        return []

    # Collect valid network_id values from both parent tables
    valid_ids: set[str] = set()

    routes = feed.get("routes")
    if routes is not None and "network_id" in routes.columns:
        route_ids = (
            routes.select("network_id")
            .filter(pl.col("network_id").is_not_null() & (pl.col("network_id") != ""))
            .to_series()
            .to_list()
        )
        valid_ids.update(route_ids)

    networks = feed.get("networks")
    if networks is not None and "network_id" in networks.columns:
        net_ids = (
            networks.select("network_id")
            .filter(pl.col("network_id").is_not_null() & (pl.col("network_id") != ""))
            .to_series()
            .to_list()
        )
        valid_ids.update(net_ids)

    # Filter to rows with non-null, non-empty network_id
    candidates = fare_leg_rules.filter(
        pl.col("network_id").is_not_null() & (pl.col("network_id") != "")
    )

    if candidates.is_empty():
        return []

    # Find violations
    violations = candidates.filter(~pl.col("network_id").is_in(list(valid_ids)))

    notices: list[Notice] = []
    for row in violations.iter_rows(named=True):
        notices.append(
            Notice(
                code="foreign_key_violation",
                severity=Severity.ERROR,
                fields={
                    "childFilename": "fare_leg_rules.txt",
                    "childFieldName": "network_id",
                    "parentFilename": "routes.txt or networks.txt",
                    "parentFieldName": "network_id",
                    "fieldValue": row["network_id"],
                    "csvRowNumber": row["csvRowNumber"],
                },
            )
        )
    return notices
