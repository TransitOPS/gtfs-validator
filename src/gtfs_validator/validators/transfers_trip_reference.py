"""Validator: TransfersTripReferenceValidator.

Checks that from_trip_id / to_trip_id in transfers.txt are consistent with
the from_route_id / to_route_id and from_stop_id / to_stop_id fields:

  - transfer_with_invalid_trip_and_route: route field does not match the
    trip's actual route_id from trips.txt.
  - transfer_with_invalid_trip_and_stop: stop field is not served by the
    trip (considering station expansion for location_type=1 stops).
"""

from __future__ import annotations

import polars as pl

from gtfs_validator.context import ValidationContext
from gtfs_validator.notices import Notice, Severity

_TRANSFER_DIRECTIONS: list[dict[str, str]] = [
    {
        "trip_field": "from_trip_id",
        "route_field": "from_route_id",
        "stop_field": "from_stop_id",
    },
    {
        "trip_field": "to_trip_id",
        "route_field": "to_route_id",
        "stop_field": "to_stop_id",
    },
]


def _expand_stop(
    stop_id: str,
    stops_location: dict[str, int],
    children_by_station: dict[str, set[str]],
) -> set[str]:
    """Return the set of candidate stop IDs to check against trip stop-times.

    - STOP (location_type=0): returns {stop_id}
    - STATION (location_type=1): returns the set of child stops
    - Other / unknown: returns empty set (FK or stop-type validator handles it)
    """
    loc_type = stops_location.get(stop_id)
    if loc_type is None:
        return set()  # not found; FK validator handles this
    if loc_type == 0:  # STOP
        return {stop_id}
    if loc_type == 1:  # STATION
        return children_by_station.get(stop_id, set())
    return set()  # ENTRANCE / GENERIC_NODE / BOARDING_AREA


def validate_transfers_trip_reference(
    feed: dict[str, pl.DataFrame],
    ctx: ValidationContext,
) -> list[Notice]:
    """Emit ERROR when a transfer's trip/route or trip/stop pairing is invalid."""
    if "transfers" not in feed:
        return []
    transfers = feed["transfers"]
    if transfers.is_empty():
        return []
    required_cols = {
        "from_trip_id",
        "to_trip_id",
        "from_route_id",
        "to_route_id",
        "from_stop_id",
        "to_stop_id",
        "csv_row_number",
    }
    if not required_cols.issubset(transfers.columns):
        return []

    # trips lookup: trip_id -> route_id
    trips_by_id: dict[str, str] = {}
    trips_df = feed.get("trips")
    if trips_df is not None and {"trip_id", "route_id"}.issubset(trips_df.columns):
        for row in trips_df.select(["trip_id", "route_id"]).iter_rows(named=True):
            if row["trip_id"] is not None:
                trips_by_id[row["trip_id"]] = row["route_id"] or ""

    # stop_times lookup: trip_id -> set of stop_ids served
    stop_times_by_trip: dict[str, set[str]] = {}
    stop_times_df = feed.get("stop_times")
    if stop_times_df is not None and {"trip_id", "stop_id"}.issubset(
        stop_times_df.columns
    ):
        for row in stop_times_df.select(["trip_id", "stop_id"]).iter_rows(named=True):
            if row["trip_id"] is not None and row["stop_id"] is not None:
                stop_times_by_trip.setdefault(row["trip_id"], set()).add(row["stop_id"])

    # stops lookups: stop_id -> location_type, parent_station -> set of child stop_ids
    stops_location: dict[str, int] = {}
    children_by_station: dict[str, set[str]] = {}
    stops_df = feed.get("stops")
    if stops_df is not None and {
        "stop_id",
        "location_type",
        "parent_station",
    }.issubset(stops_df.columns):
        for row in stops_df.select(
            ["stop_id", "location_type", "parent_station"]
        ).iter_rows(named=True):
            sid = row["stop_id"]
            if sid is not None:
                stops_location[sid] = (
                    row["location_type"] if row["location_type"] is not None else 0
                )
                if row["parent_station"]:
                    children_by_station.setdefault(row["parent_station"], set()).add(
                        sid
                    )

    notices: list[Notice] = []

    for row in transfers.iter_rows(named=True):
        csv_row_number = row["csv_row_number"]

        for direction in _TRANSFER_DIRECTIONS:
            trip_field = direction["trip_field"]
            route_field = direction["route_field"]
            stop_field = direction["stop_field"]

            trip_id = row.get(trip_field)
            if trip_id is None:
                continue

            if trip_id not in trips_by_id:
                continue  # FK validator owns this error

            expected_route_id = trips_by_id[trip_id]

            # Route check
            route_id = row.get(route_field)
            if route_id is not None and route_id != expected_route_id:
                notices.append(
                    Notice(
                        code="transfer_with_invalid_trip_and_route",
                        severity=Severity.ERROR,
                        fields={
                            "csv_row_number": csv_row_number,
                            "trip_field_name": trip_field,
                            "trip_id": trip_id,
                            "route_field_name": route_field,
                            "route_id": route_id,
                            "expected_route_id": expected_route_id,
                        },
                    )
                )

            # Stop check
            stop_id = row.get(stop_field)
            if stop_id is not None:
                if stop_id not in stops_location:
                    continue  # FK validator owns this error
                candidates = _expand_stop(stop_id, stops_location, children_by_station)
                trip_stops = stop_times_by_trip.get(trip_id, set())
                if candidates.isdisjoint(trip_stops):
                    notices.append(
                        Notice(
                            code="transfer_with_invalid_trip_and_stop",
                            severity=Severity.ERROR,
                            fields={
                                "csv_row_number": csv_row_number,
                                "trip_field_name": trip_field,
                                "trip_id": trip_id,
                                "stop_field_name": stop_field,
                                "stop_id": stop_id,
                            },
                        )
                    )

    return notices
