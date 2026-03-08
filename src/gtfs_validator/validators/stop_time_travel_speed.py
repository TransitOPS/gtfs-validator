"""Validator: StopTimeTravelSpeedValidator.

Checks that the implied travel speed between consecutive stop times does not
exceed the maximum speed for the route type. Two notice codes are emitted:

- ``fast_travel_between_consecutive_stops``: adjacent pair exceeds the limit.
- ``fast_travel_between_far_stops``: any two stops (accumulated consecutive
  distances > 10 km) exceeds the limit. At most one per trip.
"""

from __future__ import annotations

import math
from bisect import bisect_left
from collections.abc import Mapping

import polars as pl

from gtfs_validator.context import (
    ValidationContext,
    build_stop_location_cache,
)
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
_ResolvedLatLngByStopId = Mapping[str, tuple[float, float] | None]
_StopNameByStopId = Mapping[str, str | None]


# ---------------------------------------------------------------------------
# Pure helper functions (exported for testing)
# ---------------------------------------------------------------------------


def _time_to_seconds(t: str) -> int | None:
    """Convert a GTFS time string (e.g. ``"25:30:00"``) to seconds since midnight.

    GTFS times may exceed 24 hours. Returns ``None`` for falsy/empty values.
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
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )
    return R * 2 * math.asin(math.sqrt(a))


def resolve_stop_latlng(
    stop_id: str,
    stops_index: dict[str, dict[str, object]],
) -> tuple[float, float] | None:
    """Return ``(lat, lon)`` for *stop_id*, following ``parent_station`` up to
    ``MAX_PARENT_HOPS`` times. Returns ``None`` when coordinates cannot be
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
            return (float(str(lat)), float(str(lon)))
        parent = row.get("parent_station")
        if not parent:
            return None
        current_id = str(parent)
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
    return (
        distance_km
        * NUM_SECONDS_PER_HOUR
        / get_time_between_stops_secs(departure_secs, arrival_secs)
    )


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
    start_idx: int,
    end_idx: int,
    csv_row_numbers: list[int],
    stop_sequences: list[int],
    stop_ids: list[str],
    departure_times: list[int | None],
    arrival_times: list[int | None],
    stop_name_by_stop_id: Mapping[str, str | None],
) -> dict[str, object]:
    departure_time = departure_times[start_idx]
    arrival_time = arrival_times[end_idx]
    assert departure_time is not None
    assert arrival_time is not None

    start_stop_id = stop_ids[start_idx]
    end_stop_id = stop_ids[end_idx]
    return {
        "trip_csv_row_number": trip_csv_row_number,
        "trip_id": trip_id,
        "route_id": route_id,
        "speed_kph": round(speed_kph, 2),
        "distance_km": round(distance_km, 4),
        "csv_row_number1": csv_row_numbers[start_idx],
        "stop_sequence1": stop_sequences[start_idx],
        "stop_id1": start_stop_id,
        "stop_name1": stop_name_by_stop_id.get(start_stop_id) or "",
        "departure_time1": secs_to_hhmmss(departure_time),
        "csv_row_number2": csv_row_numbers[end_idx],
        "stop_sequence2": stop_sequences[end_idx],
        "stop_id2": end_stop_id,
        "stop_name2": stop_name_by_stop_id.get(end_stop_id) or "",
        "arrival_time2": secs_to_hhmmss(arrival_time),
    }


def _prepare_stop_lookups(
    stops: pl.DataFrame,
    ctx: ValidationContext,
) -> tuple[_ResolvedLatLngByStopId, _StopNameByStopId, frozenset[str]]:
    cache = ctx.stop_location_cache
    if cache is None:
        cache = build_stop_location_cache(stops)
    return (
        cache.resolved_latlng_by_stop_id,
        cache.stop_name_by_stop_id,
        cache.existing_stop_ids,
    )


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

    resolved_latlng_by_stop_id, stop_name_by_stop_id, existing_stop_ids = (
        _prepare_stop_lookups(stops, ctx)
    )

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

    for time_col in ("arrival_time", "departure_time"):
        if time_col in joined.columns and joined[time_col].dtype == pl.Utf8:
            joined = joined.with_columns(
                pl.col(time_col)
                .map_elements(_time_to_seconds, return_dtype=pl.Int64)
                .alias(time_col)
            )

    trip_ids = joined["trip_id"].to_list()
    route_ids = joined["route_id"].to_list()
    route_types = joined["route_type"].to_list()
    trip_csv_row_numbers = joined["trip_csv_row_number"].to_list()
    stop_ids = joined["stop_id"].to_list()
    stop_sequences = joined["stop_sequence"].to_list()
    csv_row_numbers = joined["csv_row_number"].to_list()
    arrival_times = joined["arrival_time"].to_list()
    departure_times = joined["departure_time"].to_list()

    notices: list[Notice] = []
    total_rows = len(trip_ids)
    trip_start = 0

    while trip_start < total_rows:
        trip_end = trip_start + 1
        trip_id = trip_ids[trip_start]
        while trip_end < total_rows and trip_ids[trip_end] == trip_id:
            trip_end += 1

        if trip_end - trip_start < 2:
            trip_start = trip_end
            continue

        route_type = route_types[trip_start]
        if route_type is None:
            trip_start = trip_end
            continue

        max_speed = get_max_speed_kph(int(route_type))
        route_id = route_ids[trip_start]
        trip_csv_row_number = trip_csv_row_numbers[trip_start]

        # Consecutive-stop check with exact bridging behavior.
        start_idx = trip_start
        start_latlng = resolved_latlng_by_stop_id.get(stop_ids[start_idx])
        for end_idx in range(trip_start + 1, trip_end):
            end_latlng = resolved_latlng_by_stop_id.get(stop_ids[end_idx])

            if start_latlng is None or end_latlng is None:
                continue

            distance_km = haversine_km(*start_latlng, *end_latlng)

            departure_secs = departure_times[start_idx]
            arrival_secs = arrival_times[end_idx]
            if departure_secs is None or arrival_secs is None:
                start_idx = end_idx
                start_latlng = end_latlng
                continue

            speed = get_speed_kph(distance_km, departure_secs, arrival_secs)
            if speed > max_speed:
                notices.append(
                    Notice(
                        code="fast_travel_between_consecutive_stops",
                        severity=Severity.WARNING,
                        fields=_build_notice_fields(
                            int(trip_csv_row_number),
                            str(trip_id),
                            str(route_id),
                            speed,
                            distance_km,
                            start_idx,
                            end_idx,
                            csv_row_numbers,
                            stop_sequences,
                            stop_ids,
                            departure_times,
                            arrival_times,
                            stop_name_by_stop_id,
                        ),
                    )
                )

            start_idx = end_idx
            start_latlng = end_latlng

        # Precompute per-trip consecutive segment distances and prefix sums.
        trip_len = trip_end - trip_start
        segment_distances_km: list[float] = [0.0] * (trip_len - 1)
        prefix_distances_km: list[float] = [0.0] * trip_len
        total_trip_distance_km = 0.0

        for rel_idx in range(trip_len - 1):
            left_idx = trip_start + rel_idx
            right_idx = left_idx + 1
            left_latlng = resolved_latlng_by_stop_id.get(stop_ids[left_idx])
            right_latlng = resolved_latlng_by_stop_id.get(stop_ids[right_idx])
            if left_latlng is None or right_latlng is None:
                segment_distance_km = 0.0
            else:
                segment_distance_km = haversine_km(*left_latlng, *right_latlng)
            segment_distances_km[rel_idx] = segment_distance_km
            total_trip_distance_km += segment_distance_km
            prefix_distances_km[rel_idx + 1] = total_trip_distance_km

        # Cheap skip: no far-stop pair can exceed the distance threshold.
        if total_trip_distance_km > FAR_STOP_DISTANCE_THRESHOLD_KM:
            far_notice: Notice | None = None
            abort_far_check = False

            for end_rel_idx in range(trip_len):
                end_idx = trip_start + end_rel_idx
                arrival_secs = arrival_times[end_idx]
                if arrival_secs is None:
                    continue

                end_stop_id = stop_ids[end_idx]
                if end_stop_id not in existing_stop_ids:
                    abort_far_check = True
                    break

                threshold = (
                    prefix_distances_km[end_rel_idx]
                    - FAR_STOP_DISTANCE_THRESHOLD_KM
                )
                farthest_near_start_idx = bisect_left(
                    prefix_distances_km,
                    threshold,
                    0,
                    end_rel_idx,
                ) - 1

                for start_rel_idx in range(farthest_near_start_idx, -1, -1):
                    start_idx = trip_start + start_rel_idx
                    departure_secs = departure_times[start_idx]
                    if departure_secs is None:
                        continue

                    distance_to_end = (
                        prefix_distances_km[end_rel_idx]
                        - prefix_distances_km[start_rel_idx]
                    )
                    raw = arrival_secs - departure_secs
                    if raw <= 0:
                        effective_secs = MIN_EFFECTIVE_TIME_SECS
                    elif arrival_secs % 60 == 0 and departure_secs % 60 == 0:
                        effective_secs = raw + MIN_EFFECTIVE_TIME_SECS
                    else:
                        effective_secs = raw

                    speed = distance_to_end * NUM_SECONDS_PER_HOUR / effective_secs
                    if speed <= max_speed:
                        continue

                    start_stop_id = stop_ids[start_idx]
                    if start_stop_id not in existing_stop_ids:
                        abort_far_check = True
                        break

                    far_notice = Notice(
                        code="fast_travel_between_far_stops",
                        severity=Severity.WARNING,
                        fields=_build_notice_fields(
                            int(trip_csv_row_number),
                            str(trip_id),
                            str(route_id),
                            speed,
                            distance_to_end,
                            start_idx,
                            end_idx,
                            csv_row_numbers,
                            stop_sequences,
                            stop_ids,
                            departure_times,
                            arrival_times,
                            stop_name_by_stop_id,
                        ),
                    )
                    break

                if abort_far_check or far_notice is not None:
                    break

            if far_notice is not None and not abort_far_check:
                notices.append(far_notice)

        trip_start = trip_end

    return notices
