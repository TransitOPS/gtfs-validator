"""Validator: NetworkIdConsistencyValidator.

Detects when routes.txt declares a network_id column while route_networks.txt
or networks.txt is also present, because network membership must be specified
in exactly one place.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from gtfs_validator.notices import Notice, Severity

if TYPE_CHECKING:
    from gtfs_validator.context import ValidationContext


def validate_network_id_consistency(
    feed: dict[str, pl.DataFrame],
    ctx: ValidationContext,
) -> list[Notice]:
    """Emit an error when network_id in routes.txt conflicts with route_networks.txt or networks.txt."""
    routes = feed.get("routes")
    has_network_id_col = routes is not None and "network_id" in routes.columns

    if not has_network_id_col:
        return []

    notices: list[Notice] = []

    if "route_networks" in feed:
        notices.append(
            Notice(
                code="route_networks_specified_in_more_than_one_file",
                severity=Severity.ERROR,
                fields={
                    "file_name_a": "routes.txt",
                    "file_name_b": "route_networks.txt",
                    "field_name": "network_id",
                },
            )
        )

    if "networks" in feed:
        notices.append(
            Notice(
                code="route_networks_specified_in_more_than_one_file",
                severity=Severity.ERROR,
                fields={
                    "file_name_a": "routes.txt",
                    "file_name_b": "networks.txt",
                    "field_name": "network_id",
                },
            )
        )

    return notices
