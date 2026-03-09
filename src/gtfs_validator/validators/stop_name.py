"""Validator: StopNameValidator.

Checks that stops, stations, and entrances (location_type 0, 1, 2) have a
``stop_name``, and that ``stop_desc`` does not duplicate ``stop_name``
(case-insensitive).
"""

from __future__ import annotations

import polars as pl

from gtfs_validator.context import ValidationContext
from gtfs_validator.notices import Notice, Severity

_REQUIRED_NAME_LOCATION_TYPES: set[int] = {0, 1, 2}  # STOP, STATION, ENTRANCE


def validate_stop_name(
    feed: dict[str, pl.DataFrame],
    ctx: ValidationContext,
) -> list[Notice]:
    """Validate stop_name field usage in stops.txt."""
    if "stops" not in feed:
        return []

    stops = feed["stops"]

    if stops.is_empty():
        return []

    if "stop_name" not in stops.columns and "location_type" not in stops.columns:
        return []

    notices: list[Notice] = []

    # --- Check A: missing_stop_name ---
    # Only run when both stop_name and location_type columns are present.
    if "stop_name" in stops.columns and "location_type" in stops.columns:
        missing_name = stops.filter(
            pl.col("location_type").is_in(list(_REQUIRED_NAME_LOCATION_TYPES))
            & (pl.col("stop_name").is_null() | (pl.col("stop_name") == ""))
        )
        for row in missing_name.iter_rows(named=True):
            notices.append(
                Notice(
                    code="missing_stop_name",
                    severity=Severity.ERROR,
                    fields={
                        "csv_row_number": row["csv_row_number"],
                        "stop_id": row["stop_id"],
                        "location_type": row["location_type"],
                    },
                )
            )

    # --- Check B: same_name_and_description_for_stop ---
    # Only run when both stop_name and stop_desc columns are present.
    if "stop_name" in stops.columns and "stop_desc" in stops.columns:
        same_name_desc = stops.filter(
            pl.col("stop_name").is_not_null()
            & (pl.col("stop_name") != "")
            & pl.col("stop_desc").is_not_null()
            & (pl.col("stop_desc") != "")
            & (
                pl.col("stop_name").str.to_lowercase()
                == pl.col("stop_desc").str.to_lowercase()
            )
        )
        for row in same_name_desc.iter_rows(named=True):
            notices.append(
                Notice(
                    code="same_name_and_description_for_stop",
                    severity=Severity.WARNING,
                    fields={
                        "csv_row_number": row["csv_row_number"],
                        "stop_id": row["stop_id"],
                        "stop_desc": row["stop_desc"],
                    },
                )
            )

    return notices
