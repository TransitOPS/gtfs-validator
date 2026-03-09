"""Validator: StopAccessValidator.

Checks that stop_access is only specified for stops (location_type == 0) that
have a parent_station. Emits errors when stop_access is set on:
- A stop (location_type == 0) with no parent_station.
- Any non-stop location type (1, 2, 3, or 4).
"""

from __future__ import annotations

import polars as pl

from gtfs_validator.context import ValidationContext
from gtfs_validator.notices import Notice, Severity

_STOP_LOCATION_TYPE: int = 0


def validate_stop_access(
    feed: dict[str, pl.DataFrame],
    ctx: ValidationContext,
) -> list[Notice]:
    """Validate stop_access field usage in stops.txt."""
    stops = feed.get("stops")
    if stops is None:
        return []
    if stops.is_empty():
        return []
    if "stop_access" not in stops.columns:
        return []

    # Pre-filter: only rows where stop_access is set (non-null)
    with_access = stops.filter(pl.col("stop_access").is_not_null())
    if with_access.is_empty():
        return []

    notices: list[Notice] = []

    # --- Pass 1: stop_access_specified_for_stop_with_no_parent_station ---
    # Condition: location_type == 0 (STOP) AND parent_station is null/absent
    if "parent_station" in stops.columns:
        no_parent_filter = (pl.col("location_type") == _STOP_LOCATION_TYPE) & (
            pl.col("parent_station").is_null() | (pl.col("parent_station") == "")
        )
    else:
        # parent_station column absent entirely — all STOP rows lack a parent
        no_parent_filter = pl.col("location_type") == _STOP_LOCATION_TYPE

    no_parent = with_access.filter(no_parent_filter)
    for row in no_parent.iter_rows(named=True):
        notices.append(
            Notice(
                code="stop_access_specified_for_stop_with_no_parent_station",
                severity=Severity.ERROR,
                fields={
                    "csv_row_number": row["csv_row_number"],
                    "stop_id": row["stop_id"],
                    "stop_name": row.get("stop_name"),
                    "stop_access": row["stop_access"],
                    "location_type": row["location_type"],
                },
            )
        )

    # --- Pass 2: stop_access_specified_for_incorrect_location ---
    # Condition: location_type != 0 (any non-STOP type)
    wrong_type = with_access.filter(pl.col("location_type") != _STOP_LOCATION_TYPE)
    for row in wrong_type.iter_rows(named=True):
        notices.append(
            Notice(
                code="stop_access_specified_for_incorrect_location",
                severity=Severity.ERROR,
                fields={
                    "csv_row_number": row["csv_row_number"],
                    "stop_id": row["stop_id"],
                    "stop_name": row.get("stop_name"),
                    "stop_access": row["stop_access"],
                    "location_type": row["location_type"],
                },
            )
        )

    return notices
