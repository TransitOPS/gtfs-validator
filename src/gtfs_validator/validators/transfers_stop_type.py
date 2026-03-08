"""Validator: TransfersStopTypeValidator.

Checks that from_stop_id and to_stop_id in transfers.txt reference stops
whose location_type is 0 (STOP) or 1 (STATION).
"""

from __future__ import annotations

import polars as pl

from gtfs_validator.context import ValidationContext
from gtfs_validator.notices import Notice, Severity

VALID_TRANSFER_LOCATION_TYPES: frozenset[int] = frozenset({0, 1})  # STOP, STATION

LOCATION_TYPE_NAMES: dict[int, str] = {
    0: "STOP",
    1: "STATION",
    2: "ENTRANCE",
    3: "GENERIC_NODE",
    4: "BOARDING_AREA",
}

_REQUIRED_TRANSFER_COLUMNS: frozenset[str] = frozenset(
    {"from_stop_id", "to_stop_id", "csv_row_number"}
)

_DIRECTIONS: list[tuple[str, str]] = [
    ("from_stop_id", "from_stop_id"),
    ("to_stop_id", "to_stop_id"),
]


def validate_transfers_stop_type(
    feed: dict[str, pl.DataFrame],
    ctx: ValidationContext,
) -> list[Notice]:
    """Emit ERROR for each transfer stop whose location_type is not 0 or 1."""
    # Guard: transfers table must exist and be non-empty with required columns.
    if "transfers" not in feed:
        return []
    transfers = feed["transfers"]
    if transfers.is_empty():
        return []
    if not _REQUIRED_TRANSFER_COLUMNS.issubset(transfers.columns):
        return []

    # Build stop_id -> location_type lookup.
    stops_by_id: dict[str, int | None] = {}
    stops_df = feed.get("stops")
    if (
        stops_df is not None
        and "stop_id" in stops_df.columns
        and "location_type" in stops_df.columns
    ):
        for row in stops_df.select(["stop_id", "location_type"]).iter_rows(named=True):
            if row["stop_id"] is not None:
                stops_by_id[row["stop_id"]] = row["location_type"]

    notices: list[Notice] = []

    for row in transfers.iter_rows(named=True):
        csv_row_number = row["csv_row_number"]

        for stop_field, stop_id_field_name in _DIRECTIONS:
            stop_id = row.get(stop_field)

            if stop_id is None or stop_id not in stops_by_id:
                continue  # FK absence handled elsewhere

            location_type = stops_by_id[stop_id]

            if location_type is None or location_type not in VALID_TRANSFER_LOCATION_TYPES:
                notices.append(
                    Notice(
                        code="transfer_with_invalid_stop_location_type",
                        severity=Severity.ERROR,
                        fields={
                            "csv_row_number": csv_row_number,
                            "stop_id_field_name": stop_id_field_name,
                            "stop_id": stop_id,
                            "location_type_value": location_type,
                            "location_type_name": (
                                LOCATION_TYPE_NAMES.get(location_type, str(location_type))
                                if location_type is not None
                                else "UNKNOWN"
                            ),
                        },
                    )
                )

    return notices
