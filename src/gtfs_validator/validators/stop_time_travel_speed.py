"""Validator: StopTimeTravelSpeedValidator.

Checks that the implied travel speed between consecutive stop times does not
exceed the maximum speed for the route type.  Two notice codes are emitted:

- ``fast_travel_between_consecutive_stops``: adjacent pair exceeds the limit.
- ``fast_travel_between_far_stops``: any two stops (accumulated consecutive
  distances > 10 km) exceeds the limit.  At most one per trip.
"""

from __future__ import annotations

import math
from typing import Optional

import polars as pl

from gtfs_validator.context import ValidationContext
from gtfs_validator.notices import Notice, Severity

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

MAX_SPEED_KPH: dict[int, float] = {
    0: 100.0,   # LIGHT_RAIL
    1: 150.0,   # SUBWAY
    2: 500.0,   # RAIL
    3: 150.0,   # BUS
    4: 80.0,    # FERRY
    5: 30.0,    # CABLE_TRAM
    6: 50.0,    # AERIAL_LIFT
    7: 50.0,    # FUNICULAR
    11: 150.0,  # TROLLEYBUS
    12: 150.0,  # MONORAIL
}
DEFAULT_MAX_SPEED_KPH: float = 200.0
FAR_STOP_DISTANCE_THRESHOLD_KM: float = 10.0
MIN_EFFECTIVE_TIME_SECS: int = 60
NUM_SECONDS_PER_HOUR: int = 3600
MAX_PARENT_HOPS: int = 3

_REQUIRED_TABLES = ["stop_times", "trips", "routes", "stops"]


# ---------------------------------------------------------------------------
# Pure helper functions (exported for testing)
# ---------------------------------------------------------------------------


def _time_to_seconds(t: str) -> Optional[int]:
    """Convert a GTFS time string (e.g. ``"25:30:00"``) to seconds since midnight.

    GTFS times may exceed 24 hours.  Returns ``None`` for falsy/empty values.
    """
    if not t:
        return None
    h, m, s = t.split(":")
    return int(h) * 3600 + int(m) * 60 + int(s)


def get_max_speed_kph(route_type: int) -> float:
    """Return the maximum allowed speed in km/h for *route_type*."""
    return MAX_SPEED_KPH.get(route_type, DEFAULT_MAX_SPEED_KPH)


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return the great-circle distance in km between two lat/lon points."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    )
    return R * 2 * math.asin(math.sqrt(a))


def resolve_stop_latlng(
    stop_id: str,
    stops_index: dict[str, dict],
) -> Optional[tuple[float, float]]:
    """Return ``(lat, lon)`` for *stop_id*, following ``parent_station`` up to
    ``MAX_PARENT_HOPS`` times.  Returns ``None`` when coordinates cannot be
    resolved.
    """
    current_id = stop_id
    for _ in range(MAX_PARENT_HOPS):
        row = stops_index.get(current_id)
        if row is None:
            return None
        lat = row.get("stop_lat")
        lon = row.get("stop_lon")
        if lat is not None and lon is not None:
            return (lat, lon)
        parent = row.get("parent_station")
        if not parent:
            return None
        current_id = parent
    return None


def get_time_between_stops_secs(departure_secs: int, arrival_secs: int) -> int:
    """Return effective travel time in seconds, applying minute-resolution
    buffering and a minimum of 60 s for non-positive raw durations.
    """
    raw = arrival_secs - departure_secs
    if raw <= 0:
        return MIN_EFFECTIVE_TIME_SECS
    if arrival_secs % 60 == 0 and departure_secs % 60 == 0:
        return raw + MIN_EFFECTIVE_TIME_SECS
    return raw


def get_speed_kph(distance_km: float, departure_secs: int, arrival_secs: int) -> float:
    """Return implied speed in km/h."""
    return distance_km * NUM_SECONDS_PER_HOUR / get_time_between_stops_secs(departure_secs, arrival_secs)


def secs_to_hhmmss(secs: int) -> str:
    """Convert integer seconds-since-midnight to ``"HH:MM:SS"``."""
    h = secs // 3600
    m = (secs % 3600) // 60
    s = secs % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _build_notice_fields(
    trip_csv_row_number: int,
    trip_id: str,
    route_id: str,
    speed_kph: float,
    distance_km: float,
    start: dict,
    end: dict,
    stops_index: dict[str, dict],
) -> dict:
    start_stop = stops_index.get(start["stop_id"]) or {}
    end_stop = stops_index.get(end["stop_id"]) or {}
    return {
        "trip_csv_row_number": trip_csv_row_number,
        "trip_id": trip_id,
        "route_id": route_id,
        "speed_kph": round(speed_kph, 2),
        "distance_km": round(distance_km, 4),
        "csv_row_number1": start["csv_row_number"],
        "stop_sequence1": start["stop_sequence"],
        "stop_id1": start["stop_id"],
        "stop_name1": start_stop.get("stop_name") or "",
        "departure_time1": secs_to_hhmmss(start["departure_time"]),
        "csv_row_number2": end["csv_row_number"],
        "stop_sequence2": end["stop_sequence"],
        "stop_id2": end["stop_id"],
        "stop_name2": end_stop.get("stop_name") or "",
        "arrival_time2": secs_to_hhmmss(end["arrival_time"]),
    }


def _check_consecutive_stops(
    rows: list[dict],
    distances_km: list[float],
    max_speed: float,
    trip_id: str,
    route_id: str,
    trip_csv_row_number: int,
    stops_index: dict[str, dict],
) -> list[Notice]:
    notices: list[Notice] = []
    start = rows[0]
    for i in range(1, len(rows)):
        end = rows[i]

        start_latlng = resolve_stop_latlng(start["stop_id"], stops_index)
        end_latlng = resolve_stop_latlng(end["stop_id"], stops_index)

        if start_latlng is None or end_latlng is None:
            # Do NOT advance start; continue with same start
            continue

        distance_km = haversine_km(*start_latlng, *end_latlng)

        # Both times must be present
        if start["departure_time"] is None or end["arrival_time"] is None:
            start = end
            continue

        speed = get_speed_kph(distance_km, start["departure_time"], end["arrival_time"])

        if speed > max_speed:
            notices.append(
                Notice(
                    code="fast_travel_between_consecutive_stops",
                    severity=Severity.WARNING,
                    fields=_build_notice_fields(
                        trip_csv_row_number,
                        trip_id,
                        route_id,
                        speed,
                        distance_km,
                        start,
                        end,
                        stops_index,
                    ),
                )
            )

        start = end

    return notices


def _check_far_stops(
    rows: list[dict],
    distances_km: list[float],
    max_speed: float,
    trip_id: str,
    route_id: str,
    trip_csv_row_number: int,
    stops_index: dict[str, dict],
) -> list[Notice]:
    n = len(rows)
    for end_idx in range(n):
        end = rows[end_idx]

        if end["arrival_time"] is None:
            continue

        if stops_index.get(end["stop_id"]) is None:
            return []  # abort entire far-stop check for this trip

        distance_to_end = 0.0

        for start_idx in range(end_idx - 1, -1, -1):
            distance_to_end += distances_km[start_idx]

            start = rows[start_idx]

            if start["departure_time"] is None:
                continue

            speed = get_speed_kph(distance_to_end, start["departure_time"], end["arrival_time"])

            if speed <= max_speed:
                continue

            if distance_to_end <= FAR_STOP_DISTANCE_THRESHOLD_KM:
                continue  # near pair — handled by consecutive check

            if stops_index.get(start["stop_id"]) is None:
                return []  # abort entire far-stop check for this trip

            return [
                Notice(
                    code="fast_travel_between_far_stops",
                    severity=Severity.WARNING,
                    fields=_build_notice_fields(
                        trip_csv_row_number,
                        trip_id,
                        route_id,
                        speed,
                        distance_to_end,
                        start,
                        end,
                        stops_index,
                    ),
                )
            ]

    return []


# ---------------------------------------------------------------------------
# Main validator
# ---------------------------------------------------------------------------


def validate_stop_time_travel_speed(
    feed: dict[str, pl.DataFrame],
    ctx: ValidationContext,
) -> list[Notice]:
    """Validate that stop-time travel speeds do not exceed per-route-type limits."""
    for table in _REQUIRED_TABLES:
        if table not in feed or feed[table].is_empty():
            return []

    stop_times = feed["stop_times"]
    trips = feed["trips"]
    routes = feed["routes"]
    stops = feed["stops"]

    # Build stops lookup dict keyed on stop_id
    stops_cols = ["stop_id", "stop_lat", "stop_lon", "stop_name"]
    available_stops_cols = [c for c in stops_cols if c in stops.columns]
    if "parent_station" in stops.columns:
        available_stops_cols.append("parent_station")

    stops_index: dict[str, dict] = {
        row["stop_id"]: row
        for row in stops.select(available_stops_cols).iter_rows(named=True)
    }

    # Join stop_times -> trips -> routes to get route_type and trip metadata
    trips_slim = trips.select([
        pl.col("trip_id"),
        pl.col("route_id"),
        pl.col("csv_row_number").alias("trip_csv_row_number"),
    ])
    routes_slim = routes.select([
        pl.col("route_id"),
        pl.col("route_type"),
    ])
    joined = (
        stop_times
        .join(trips_slim, on="trip_id", how="left")
        .join(routes_slim, on="route_id", how="left")
        .filter(pl.col("route_type").is_not_null())
        .sort(["trip_id", "stop_sequence"])
    )

    if joined.is_empty():
        return []

    # Convert GTFS time strings (stored as Utf8) to integer seconds-since-midnight
    # so that arithmetic works correctly.  Columns may be absent if the feed omits them.
    # Skip conversion if the column is already a numeric type (e.g. in tests).
    for time_col in ("arrival_time", "departure_time"):
        if time_col in joined.columns and joined[time_col].dtype == pl.Utf8:
            joined = joined.with_columns(
                pl.col(time_col)
                .map_elements(_time_to_seconds, return_dtype=pl.Int64)
                .alias(time_col)
            )

    notices: list[Notice] = []

    for trip_df in joined.partition_by("trip_id", maintain_order=True):
        rows = list(trip_df.iter_rows(named=True))
        if len(rows) < 2:
            continue

        route_type = rows[0]["route_type"]
        max_speed = get_max_speed_kph(route_type)
        trip_id = rows[0]["trip_id"]
        route_id = rows[0]["route_id"]
        trip_csv_row_number = rows[0]["trip_csv_row_number"]

        # Pre-compute consecutive distances for far-stop check
        distances_km: list[float] = []
        for i in range(len(rows) - 1):
            start_latlng = resolve_stop_latlng(rows[i]["stop_id"], stops_index)
            end_latlng = resolve_stop_latlng(rows[i + 1]["stop_id"], stops_index)
            if start_latlng is None or end_latlng is None:
                distances_km.append(0.0)
            else:
                distances_km.append(haversine_km(*start_latlng, *end_latlng))

        notices.extend(
            _check_consecutive_stops(
                rows, distances_km, max_speed,
                trip_id, route_id, trip_csv_row_number,
                stops_index,
            )
        )

        notices.extend(
            _check_far_stops(
                rows, distances_km, max_speed,
                trip_id, route_id, trip_csv_row_number,
                stops_index,
            )
        )

    return notices
