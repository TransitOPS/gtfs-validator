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

import numpy as np
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


def _parse_gtfs_time_expr(col_name: str) -> pl.Expr:
    """Vectorized Polars expression: GTFS time string → seconds since midnight.

    Handles times > 24h (e.g. ``"25:30:00"``). Nulls and non-matching values
    produce null output.
    """
    g = pl.col(col_name).str.extract_groups(r"^(\d+):(\d{2}):(\d{2})$")
    return (
        g.struct.field("1").cast(pl.Int64) * 3600
        + g.struct.field("2").cast(pl.Int64) * 60
        + g.struct.field("3").cast(pl.Int64)
    ).alias(col_name)


def _haversine_km_vectorized(
    lat1: np.ndarray,
    lon1: np.ndarray,
    lat2: np.ndarray,
    lon2: np.ndarray,
) -> np.ndarray:
    """Vectorized haversine distance in km. NaN inputs propagate as NaN."""
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2
    )
    return R * 2 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


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


def _build_consecutive_notice(
    row: dict,
    stop_name_by_stop_id: Mapping[str, str | None],
) -> Notice:
    """Build a consecutive-stop speed notice from a Polars row dict."""
    return Notice(
        code="fast_travel_between_consecutive_stops",
        severity=Severity.WARNING,
        fields={
            "trip_csv_row_number": int(row["trip_csv_row_number"]),
            "trip_id": str(row["trip_id"]),
            "route_id": str(row["route_id"]),
            "speed_kph": round(float(row["speed_kph"]), 2),
            "distance_km": round(float(row["consec_distance_km"]), 4),
            "csv_row_number1": row["prev_csv_row_number"],
            "stop_sequence1": row["prev_stop_sequence"],
            "stop_id1": row["prev_stop_id"],
            "stop_name1": stop_name_by_stop_id.get(row["prev_stop_id"]) or "",
            "departure_time1": secs_to_hhmmss(int(row["prev_departure_time"])),
            "csv_row_number2": row["csv_row_number"],
            "stop_sequence2": row["stop_sequence"],
            "stop_id2": row["stop_id"],
            "stop_name2": stop_name_by_stop_id.get(row["stop_id"]) or "",
            "arrival_time2": secs_to_hhmmss(int(row["arrival_time"])),
        },
    )


def _haversine_km_expr(
    lat1: str, lon1: str, lat2: str, lon2: str,
) -> pl.Expr:
    """Polars expression: haversine distance in km between two lat/lon column pairs."""
    R = 6371.0
    dlat = (pl.col(lat2) - pl.col(lat1)).radians() / 2
    dlon = (pl.col(lon2) - pl.col(lon1)).radians() / 2
    a = (
        dlat.sin() ** 2
        + pl.col(lat1).radians().cos() * pl.col(lat2).radians().cos() * dlon.sin() ** 2
    )
    return R * 2 * a.sqrt().arcsin()


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

    time_cols_to_parse = [
        c for c in ("arrival_time", "departure_time")
        if c in joined.columns and joined[c].dtype == pl.Utf8
    ]
    if time_cols_to_parse:
        joined = joined.with_columns([_parse_gtfs_time_expr(c) for c in time_cols_to_parse])

    # --- Add resolved lat/lon via join ---
    latlng_rows = [
        {"stop_id": sid, "_rlat": ll[0], "_rlon": ll[1]}
        for sid, ll in resolved_latlng_by_stop_id.items()
        if ll is not None
    ]
    if latlng_rows:
        latlng_df = pl.DataFrame(latlng_rows).with_columns(
            pl.col("_rlat").cast(pl.Float64), pl.col("_rlon").cast(pl.Float64),
        )
        joined = joined.join(latlng_df, on="stop_id", how="left")
    else:
        joined = joined.with_columns(
            pl.lit(None, dtype=pl.Float64).alias("_rlat"),
            pl.lit(None, dtype=pl.Float64).alias("_rlon"),
        )

    # --- Add max speed per route_type ---
    speed_rows = [{"route_type": k, "max_speed_kph": v} for k, v in MAX_SPEED_KPH.items()]
    speed_df = pl.DataFrame(speed_rows).with_columns(
        pl.col("route_type").cast(joined["route_type"].dtype),
    )
    joined = joined.join(speed_df, on="route_type", how="left").with_columns(
        pl.col("max_speed_kph").fill_null(DEFAULT_MAX_SPEED_KPH),
    )

    # ===================================================================
    # Consecutive-stop check (Polars-native)
    # ===================================================================
    # Filter to rows with resolved coordinates, then pair adjacent valid
    # stops within each trip via shift().
    valid = joined.filter(pl.col("_rlat").is_not_null())
    valid = valid.with_columns([
        pl.col("_rlat").shift(1).over("trip_id").alias("prev_lat"),
        pl.col("_rlon").shift(1).over("trip_id").alias("prev_lon"),
        pl.col("departure_time").shift(1).over("trip_id").alias("prev_departure_time"),
        pl.col("csv_row_number").shift(1).over("trip_id").alias("prev_csv_row_number"),
        pl.col("stop_sequence").shift(1).over("trip_id").alias("prev_stop_sequence"),
        pl.col("stop_id").shift(1).over("trip_id").alias("prev_stop_id"),
    ])
    # Drop first row of each trip (no predecessor).
    valid = valid.filter(pl.col("prev_lat").is_not_null())

    # Haversine distance and speed as Polars expressions.
    dist_km_expr = _haversine_km_expr("prev_lat", "prev_lon", "_rlat", "_rlon")

    raw_time = pl.col("arrival_time") - pl.col("prev_departure_time")
    is_minute = (pl.col("arrival_time") % 60 == 0) & (pl.col("prev_departure_time") % 60 == 0)
    effective_time = (
        pl.when(raw_time <= 0).then(pl.lit(MIN_EFFECTIVE_TIME_SECS))
        .when(is_minute).then(raw_time + MIN_EFFECTIVE_TIME_SECS)
        .otherwise(raw_time)
    )
    speed_expr = dist_km_expr * NUM_SECONDS_PER_HOUR / effective_time

    valid = valid.with_columns([
        dist_km_expr.alias("consec_distance_km"),
        speed_expr.alias("speed_kph"),
    ])

    violations = valid.filter(
        pl.col("speed_kph").is_not_null()
        & pl.col("arrival_time").is_not_null()
        & pl.col("prev_departure_time").is_not_null()
        & (pl.col("speed_kph") > pl.col("max_speed_kph"))
    )

    notices: list[Notice] = [
        _build_consecutive_notice(row, stop_name_by_stop_id)
        for row in violations.to_dicts()
    ]

    # ===================================================================
    # Far-stop check (only for qualifying trips with total distance > 10 km)
    # ===================================================================
    # Compute segment distances for ALL rows (NaN coords → 0 km).
    joined = joined.with_columns([
        pl.col("_rlat").shift(1).over("trip_id").alias("_seg_prev_lat"),
        pl.col("_rlon").shift(1).over("trip_id").alias("_seg_prev_lon"),
    ])
    seg_dist_expr = _haversine_km_expr("_seg_prev_lat", "_seg_prev_lon", "_rlat", "_rlon")
    joined = joined.with_columns(
        seg_dist_expr.fill_null(0.0).alias("_seg_dist_km"),
    )
    joined = joined.with_columns(
        pl.col("_seg_dist_km").cum_sum().over("trip_id").alias("_prefix_dist_km"),
    )

    # Find trips exceeding the distance threshold.
    trip_totals = joined.group_by("trip_id").agg(
        pl.col("_seg_dist_km").sum().alias("_total_dist_km"),
    ).filter(pl.col("_total_dist_km") > FAR_STOP_DISTANCE_THRESHOLD_KM)

    if not trip_totals.is_empty():
        qualifying_ids = set(trip_totals["trip_id"].to_list())
        far_df = joined.filter(pl.col("trip_id").is_in(list(qualifying_ids)))

        # Materialize only the columns needed for far-stop iteration.
        far_trip_ids = far_df["trip_id"].to_list()
        far_stop_ids = far_df["stop_id"].to_list()
        far_csv_rows = far_df["csv_row_number"].to_list()
        far_stop_seqs = far_df["stop_sequence"].to_list()
        far_arrival = far_df["arrival_time"].to_list()
        far_departure = far_df["departure_time"].to_list()
        far_route_ids = far_df["route_id"].to_list()
        far_route_types = far_df["route_type"].to_list()
        far_trip_csv_rows = far_df["trip_csv_row_number"].to_list()
        far_max_speeds = far_df["max_speed_kph"].to_list()
        far_prefix = far_df["_prefix_dist_km"].to_list()

        total_far = len(far_trip_ids)
        trip_start = 0

        while trip_start < total_far:
            trip_end = trip_start + 1
            trip_id = far_trip_ids[trip_start]
            while trip_end < total_far and far_trip_ids[trip_end] == trip_id:
                trip_end += 1

            trip_len = trip_end - trip_start
            if trip_len < 2:
                trip_start = trip_end
                continue

            max_speed = float(far_max_speeds[trip_start])
            prefix_distances_km = [far_prefix[trip_start + i] for i in range(trip_len)]
            total_trip_distance_km = prefix_distances_km[-1]

            if total_trip_distance_km <= FAR_STOP_DISTANCE_THRESHOLD_KM:
                trip_start = trip_end
                continue

            far_notice: Notice | None = None
            abort_far_check = False

            for end_rel_idx in range(trip_len):
                end_idx = trip_start + end_rel_idx
                arrival_secs = far_arrival[end_idx]
                if arrival_secs is None:
                    continue

                end_stop_id = far_stop_ids[end_idx]
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
                    departure_secs = far_departure[start_idx]
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

                    start_stop_id = far_stop_ids[start_idx]
                    if start_stop_id not in existing_stop_ids:
                        abort_far_check = True
                        break

                    far_notice = Notice(
                        code="fast_travel_between_far_stops",
                        severity=Severity.WARNING,
                        fields=_build_notice_fields(
                            int(far_trip_csv_rows[trip_start]),
                            str(trip_id),
                            str(far_route_ids[trip_start]),
                            speed,
                            distance_to_end,
                            start_idx,
                            end_idx,
                            far_csv_rows,
                            far_stop_seqs,
                            far_stop_ids,
                            far_departure,
                            far_arrival,
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
