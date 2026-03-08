"""Validator: in-seat transfer type checks (transfer_type 4 and 5)."""

from __future__ import annotations

import polars as pl

from gtfs_validator.context import ValidationContext
from gtfs_validator.notices import Notice, Severity


def validate_transfers_in_seat_transfer_type(
    feed: dict[str, pl.DataFrame],
    ctx: ValidationContext,
) -> list[Notice]:
    """Validate in-seat transfers (transfer_type=4 or 5).

    Checks:
    - from_trip_id and to_trip_id must be present
    - Referenced stops must not be stations (location_type=1)
    - from_stop_id must be the last stop in the from-trip
    - to_stop_id must be the first stop in the to-trip
    """
    transfers = feed.get("transfers")
    if transfers is None:
        return []
    if transfers.is_empty():
        return []
    if "transfer_type" not in transfers.columns:
        return []

    required_cols = {
        "transfer_type",
        "from_trip_id",
        "to_trip_id",
        "from_stop_id",
        "to_stop_id",
        "csv_row_number",
    }
    if not required_cols.issubset(set(transfers.columns)):
        return []

    # Build stops lookup: stop_id -> location_type (int or None)
    stops_by_id: dict[str, int | None] = {}
    stops_df = feed.get("stops")
    if (
        stops_df is not None
        and "stop_id" in stops_df.columns
        and "location_type" in stops_df.columns
    ):
        for row in stops_df.select(["stop_id", "location_type"]).iter_rows(named=True):
            stops_by_id[row["stop_id"]] = row["location_type"]

    # Build trip -> ordered stop list: trip_id -> list[stop_id] sorted by stop_sequence
    trip_stops: dict[str, list[str]] = {}
    stop_times_df = feed.get("stop_times")
    if stop_times_df is not None:
        required_st_cols = {"trip_id", "stop_id", "stop_sequence"}
        if required_st_cols.issubset(stop_times_df.columns):
            sorted_st = stop_times_df.sort("stop_sequence").select(["trip_id", "stop_id"])
            for row in sorted_st.iter_rows(named=True):
                trip_stops.setdefault(row["trip_id"], []).append(row["stop_id"])

    # Filter to in-seat transfer rows only
    in_seat = transfers.filter(
        pl.col("transfer_type").is_not_null() & pl.col("transfer_type").is_in([4, 5])
    )
    if in_seat.is_empty():
        return []

    # Direction config: (trip_id_field, stop_id_field, expected_position)
    DIRECTIONS = [
        ("from_trip_id", "from_stop_id", "last"),
        ("to_trip_id", "to_stop_id", "first"),
    ]

    notices: list[Notice] = []

    for row in in_seat.iter_rows(named=True):
        csv_row_number = row["csv_row_number"]

        for trip_field, stop_field, position in DIRECTIONS:
            trip_id = row.get(trip_field)
            stop_id = row.get(stop_field)

            # Check 1: trip_id is required for in-seat transfers
            if trip_id is None:
                notices.append(
                    Notice(
                        code="missing_required_field",
                        severity=Severity.ERROR,
                        fields={
                            "filename": "transfers.txt",
                            "csv_row_number": csv_row_number,
                            "field_name": trip_field,
                        },
                    )
                )

            # Stop checks: only if stop_id is non-null and resolves in stops
            if stop_id is None or stop_id not in stops_by_id:
                continue  # FK validation deferred

            location_type = stops_by_id[stop_id]

            # Check 2: STATION (location_type=1) is forbidden for in-seat transfers
            if location_type == 1:
                notices.append(
                    Notice(
                        code="transfer_with_invalid_stop_location_type",
                        severity=Severity.ERROR,
                        fields={
                            "csv_row_number": csv_row_number,
                            "stop_id_field_name": stop_field,
                            "stop_id": stop_id,
                            "location_type_value": 1,
                            "location_type_name": "STATION",
                        },
                    )
                )

            # Check 3: verify stop appears in trip's stop-times
            if trip_id is None or trip_id not in trip_stops:
                continue  # cross-reference deferred

            stops_in_trip = trip_stops[trip_id]
            if stop_id not in stops_in_trip:
                continue  # FK absence deferred

            # Check 4: positional check
            expected_stop = stops_in_trip[-1] if position == "last" else stops_in_trip[0]
            if expected_stop != stop_id:
                notices.append(
                    Notice(
                        code="transfer_with_suspicious_mid_trip_in_seat",
                        severity=Severity.WARNING,
                        fields={
                            "csv_row_number": csv_row_number,
                            "trip_id_field_name": trip_field,
                            "trip_id": trip_id,
                            "stop_id_field_name": stop_field,
                            "stop_id": stop_id,
                        },
                    )
                )

    return notices
