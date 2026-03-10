"""Validator: ShapeToStopMatchingValidator.

Checks that stops on each trip are close enough to the trip's shape,
that matches are in the correct sequence, and that user-distance-based
matching is consistent.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

import polars as pl

from gtfs_validator.context import ValidationContext
from gtfs_validator.notices import Notice, Severity
from gtfs_validator.validators.shape_to_stop_matching_util import (
    CandidateMatch,
    RAIL_ROUTE_TYPE,
    MatchSettings,
    Problem,
    _VEC_MIN_SEGMENTS,
    _VEC_MIN_WORK_ITEMS,
    build_shape_arrays,
    build_shape_points,
    build_shape_spatial_index,
    build_stop_points,
    compute_trip_hash,
    match_using_geo_distance,
    match_using_user_distance,
    resolve_stop_name,
)

_REQUIRED_TABLES = ["stops", "trips", "routes", "stop_times", "shapes"]


@dataclass(frozen=True)
class TripInfo:
    trip_id: str
    route_id: str
    csv_row_number: int


def _problems_to_notices(
    problems: list[Problem],
    trip: TripInfo,
    shape_id: str,
    stops_by_id: dict,
    reported_stop_ids: set[str],
    notice_code: str,
) -> list[Notice]:
    """Convert Problem objects to Notice objects, applying deduplication rules."""
    notices: list[Notice] = []

    for problem in problems:
        kind = problem.kind
        data = problem.data

        if kind in ("too_far", "too_far_user_distance"):
            stop_point = data["stop_point"]
            match = data["match"]
            stop_time_row = stop_point.stop_time_row
            stop_id = stop_time_row.get("stop_id") or ""

            # Rule 1: skip empty/null stop_id
            if not stop_id:
                continue

            # Rule 2: deduplication by stop_id per shape
            if stop_id in reported_stop_ids:
                continue
            reported_stop_ids.add(stop_id)

            stop_name = resolve_stop_name(stop_id, stops_by_id)
            # Determine the correct notice code
            if kind == "too_far_user_distance":
                code = "stop_too_far_from_shape_using_user_distance"
            else:
                code = notice_code

            notices.append(
                Notice(
                    code=code,
                    severity=Severity.WARNING,
                    fields={
                        "trip_csv_row_number": trip.csv_row_number,
                        "shape_id": shape_id,
                        "trip_id": trip.trip_id,
                        "stop_time_csv_row_number": stop_time_row["csv_row_number"],
                        "stop_id": stop_id,
                        "stop_name": stop_name,
                        "match": {"lat": match.lat, "lng": match.lon},
                        "geo_distance_to_shape": match.geo_distance_to_shape,
                    },
                )
            )

        elif kind == "too_many_matches":
            stop_point = data["stop_point"]
            match = data["match"]
            match_count = data["match_count"]
            stop_time_row = stop_point.stop_time_row
            stop_id = stop_time_row.get("stop_id") or ""

            # Rule 1: skip empty/null stop_id
            if not stop_id:
                continue

            stop_name = resolve_stop_name(stop_id, stops_by_id)

            notices.append(
                Notice(
                    code="stop_has_too_many_matches_for_shape",
                    severity=Severity.WARNING,
                    fields={
                        "trip_csv_row_number": trip.csv_row_number,
                        "shape_id": shape_id,
                        "trip_id": trip.trip_id,
                        "stop_time_csv_row_number": stop_time_row["csv_row_number"],
                        "stop_id": stop_id,
                        "stop_name": stop_name,
                        "match": {"lat": match.lat, "lng": match.lon},
                        "match_count": match_count,
                    },
                )
            )

        elif kind == "out_of_order":
            stop_point1 = data["stop_point1"]
            match1 = data["match1"]
            stop_point2 = data["stop_point2"]
            match2 = data["match2"]

            stop_time_row1 = stop_point1.stop_time_row
            stop_time_row2 = stop_point2.stop_time_row

            stop_id1 = stop_time_row1.get("stop_id") or ""
            stop_id2 = stop_time_row2.get("stop_id") or ""

            # Rule 1: skip if either stop_id is empty/null
            if not stop_id1 or not stop_id2:
                continue

            stop_name1 = resolve_stop_name(stop_id1, stops_by_id)
            stop_name2 = resolve_stop_name(stop_id2, stops_by_id)

            notices.append(
                Notice(
                    code="stops_match_shape_out_of_order",
                    severity=Severity.WARNING,
                    fields={
                        "trip_csv_row_number": trip.csv_row_number,
                        "shape_id": shape_id,
                        "trip_id": trip.trip_id,
                        "stop_time_csv_row_number1": stop_time_row1["csv_row_number"],
                        "stop_id1": stop_id1,
                        "stop_name1": stop_name1,
                        "match1": {"lat": match1.lat, "lng": match1.lon},
                        "stop_time_csv_row_number2": stop_time_row2["csv_row_number"],
                        "stop_id2": stop_id2,
                        "stop_name2": stop_name2,
                        "match2": {"lat": match2.lat, "lng": match2.lon},
                    },
                )
            )

    return notices


def validate_shape_to_stop_matching(
    feed: dict[str, pl.DataFrame],
    ctx: ValidationContext,
    settings: Optional[MatchSettings] = None,
) -> list[Notice]:
    """Validate that stops on each trip are close to the trip's shape."""
    for table in _REQUIRED_TABLES:
        if table not in feed or feed[table].is_empty():
            return []

    if settings is None:
        settings = MatchSettings()

    stops_df = feed["stops"]
    trips_df = feed["trips"]
    routes_df = feed["routes"]
    stop_times_df = feed["stop_times"]
    shapes_df = feed["shapes"]

    # Use pre-resolved coordinates from context cache when available.
    resolved_latlng = (
        ctx.stop_location_cache.resolved_latlng_by_stop_id
        if ctx.stop_location_cache is not None
        else None
    )

    # Select only needed columns for stops (graceful about optional columns)
    stops_cols = ["stop_id", "stop_name", "stop_lat", "stop_lon"]
    optional_stops = ["parent_station", "location_type"]
    available_stops_cols = stops_cols.copy()
    for col in optional_stops:
        if col in stops_df.columns:
            available_stops_cols.append(col)

    stops_by_id: dict[str, dict] = {
        row["stop_id"]: row
        for row in stops_df.select(available_stops_cols).iter_rows(named=True)
    }

    route_type_by_id: dict[str, int] = {
        row["route_id"]: row["route_type"]
        for row in routes_df.select(["route_id", "route_type"]).iter_rows(named=True)
    }

    # Group trips by shape_id first so downstream row processing can filter.
    trip_cols = ["trip_id", "route_id", "shape_id", "csv_row_number"]
    trips_by_shape_id: dict[str, list[TripInfo]] = defaultdict(list)
    relevant_trip_ids: set[str] = set()
    for row in trips_df.select(trip_cols).iter_rows(named=True):
        shape_id = row.get("shape_id")
        if shape_id:
            trip = TripInfo(
                trip_id=row["trip_id"],
                route_id=row["route_id"],
                csv_row_number=row["csv_row_number"],
            )
            trips_by_shape_id[shape_id].append(trip)
            relevant_trip_ids.add(trip.trip_id)

    # Group stop_times by trip_id
    st_cols = ["trip_id", "stop_id", "stop_sequence", "csv_row_number"]
    optional_st = ["shape_dist_traveled"]
    available_st_cols = st_cols.copy()
    for col in optional_st:
        if col in stop_times_df.columns:
            available_st_cols.append(col)

    # Pre-sort in Polars (vectorized, GIL-released) so Python loops need no further sorting.
    st_by_trip_id: dict[str, list[dict]] = defaultdict(list)
    for row in (
        stop_times_df.select(available_st_cols)
        .sort(["trip_id", "stop_sequence"])
        .iter_rows(named=True)
    ):
        if row["trip_id"] in relevant_trip_ids:
            st_by_trip_id[row["trip_id"]].append(row)

    # Group shapes by shape_id
    shape_cols = ["shape_id", "shape_pt_lat", "shape_pt_lon", "shape_pt_sequence"]
    optional_shape = ["shape_dist_traveled"]
    available_shape_cols = shape_cols.copy()
    for col in optional_shape:
        if col in shapes_df.columns:
            available_shape_cols.append(col)

    relevant_shape_ids = set(trips_by_shape_id.keys())
    # Pre-sort by shape_pt_sequence in Polars so build_shape_points needs no further sorting.
    shapes_groups: dict[str, list[dict]] = defaultdict(list)
    for row in (
        shapes_df.select(available_shape_cols)
        .sort(["shape_id", "shape_pt_sequence"])
        .iter_rows(named=True)
    ):
        shape_id = row["shape_id"]
        if shape_id in relevant_shape_ids:
            shapes_groups[shape_id].append(row)

    notices: list[Notice] = []

    for shape_id, shape_rows in shapes_groups.items():
        trips_for_shape = trips_by_shape_id.get(shape_id, [])
        if not trips_for_shape:
            continue

        shape_points = build_shape_points(shape_rows)
        if not shape_points:
            continue

        num_segments = max(0, len(shape_points) - 1)
        max_trip_stops = max(
            (len(st_by_trip_id.get(trip.trip_id, [])) for trip in trips_for_shape),
            default=0,
        )
        should_build_shape_arrays = (
            num_segments >= _VEC_MIN_SEGMENTS
            and (num_segments * max_trip_stops) >= _VEC_MIN_WORK_ITEMS
        )
        shape_arrays = build_shape_arrays(shape_points) if should_build_shape_arrays else None
        shape_spatial_index = build_shape_spatial_index(shape_points)
        shape_has_user_dist = shape_points[-1].user_distance > 0.0
        reported_stop_ids: set[str] = set()
        # Two-stage dedup: quick signature first, full trip hash only on collisions.
        dedup_quick_state: dict[
            tuple[int, str, str, float, float], list[dict] | set[tuple]
        ] = {}
        candidates_cache: dict[tuple[str, float, float, float], list[CandidateMatch]] = {}
        closest_cache: dict[tuple[str, float, float], CandidateMatch] = {}
        need_trip_dedup = len(trips_for_shape) > 1

        for trip in trips_for_shape:
            stop_times = st_by_trip_id.get(trip.trip_id, [])
            if not stop_times:
                continue

            if need_trip_dedup:
                first = stop_times[0]
                last = stop_times[-1]
                quick_sig = (
                    len(stop_times),
                    str(first.get("stop_id") or ""),
                    str(last.get("stop_id") or ""),
                    float(first.get("shape_dist_traveled") or 0.0),
                    float(last.get("shape_dist_traveled") or 0.0),
                )
                state = dedup_quick_state.get(quick_sig)
                if state is None:
                    dedup_quick_state[quick_sig] = stop_times
                elif isinstance(state, list):
                    first_hash = compute_trip_hash(state)
                    cur_hash = compute_trip_hash(stop_times)
                    hash_bucket = {first_hash}
                    dedup_quick_state[quick_sig] = hash_bucket
                    if cur_hash in hash_bucket:
                        continue
                    hash_bucket.add(cur_hash)
                else:
                    cur_hash = compute_trip_hash(stop_times)
                    if cur_hash in state:
                        continue
                    state.add(cur_hash)

            route_type = route_type_by_id.get(trip.route_id)
            if route_type is None:
                continue

            is_large_route = route_type == RAIL_ROUTE_TYPE
            stop_points = build_stop_points(stop_times, stops_by_id, is_large_route, resolved_latlng)

            # Geo-distance matching (always)
            geo_problems = match_using_geo_distance(
                stop_points,
                shape_points,
                settings,
                shape_arrays,
                shape_spatial_index,
                candidates_cache=candidates_cache,
                closest_cache=closest_cache,
            )
            notices.extend(
                _problems_to_notices(
                    geo_problems,
                    trip,
                    shape_id,
                    stops_by_id,
                    reported_stop_ids,
                    notice_code="stop_too_far_from_shape",
                )
            )

            # User-distance matching (conditional)
            stops_have_user_dist = stop_points[-1].user_distance > 0.0
            if stops_have_user_dist and shape_has_user_dist:
                user_problems = match_using_user_distance(
                    stop_points,
                    shape_points,
                    settings,
                    shape_arrays,
                    shape_spatial_index,
                    candidates_cache=candidates_cache,
                )
                notices.extend(
                    _problems_to_notices(
                        user_problems,
                        trip,
                        shape_id,
                        stops_by_id,
                        reported_stop_ids,
                        notice_code="stop_too_far_from_shape_using_user_distance",
                    )
                )

    return notices
