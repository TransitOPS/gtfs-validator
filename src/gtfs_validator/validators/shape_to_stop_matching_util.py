"""Geometry and algorithm helpers for ShapeToStopMatchingValidator.

All functions are pure Python (no Polars) for clarity and testability.
"""

from __future__ import annotations

import hashlib
import math
import struct
from dataclasses import dataclass, field
from math import asin, atan2, cos, degrees, radians, sin, sqrt
from typing import Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EARTH_RADIUS_METERS: float = 6_371_010.0
RAIL_ROUTE_TYPE: int = 2
DEFAULT_MAX_DISTANCE_METERS: float = 100.0
DEFAULT_LARGE_STATION_MULTIPLIER: float = 4.0
DEFAULT_POTENTIAL_MATCHES_THRESHOLD: int = 20


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


@dataclass
class MatchSettings:
    max_distance_meters: float = DEFAULT_MAX_DISTANCE_METERS
    large_station_multiplier: float = DEFAULT_LARGE_STATION_MULTIPLIER
    potential_matches_threshold: int = DEFAULT_POTENTIAL_MATCHES_THRESHOLD


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ShapePoint:
    geo_distance: float  # cumulative great-circle distance from shape start (metres)
    user_distance: float  # running max of shape_dist_traveled (0.0 if null)
    lat: float
    lon: float
    uv: tuple[float, float, float] = field(default_factory=lambda: (0.0, 0.0, 0.0))  # precomputed unit vector


@dataclass
class StopPoint:
    lat: float
    lon: float
    user_distance: float  # stop_time.shape_dist_traveled (0.0 if null)
    stop_time_row: dict  # original stop_time row dict
    is_large_station: bool  # True only for first/last stop on RAIL routes


@dataclass
class CandidateMatch:
    geo_distance_to_shape: float  # metres from stop to matched point on shape
    shape_geo_distance: float  # ShapePoint.geo_distance at the match location
    lat: float  # matched point lat
    lon: float  # matched point lon


@dataclass
class Problem:
    kind: str  # "too_far" | "too_many_matches" | "out_of_order"
    data: dict


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def geo_distance_meters(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Haversine great-circle distance in metres."""
    phi1, phi2 = radians(lat1), radians(lat2)
    dphi = phi2 - phi1
    dlambda = radians(lon2 - lon1)
    a = sin(dphi / 2) ** 2 + cos(phi1) * cos(phi2) * sin(dlambda / 2) ** 2
    return 2 * atan2(sqrt(a), sqrt(max(0.0, 1.0 - a))) * EARTH_RADIUS_METERS


def latlng_to_unit_vector(lat_deg: float, lon_deg: float) -> tuple[float, float, float]:
    """Convert geographic coordinates to a unit 3-vector on the unit sphere."""
    lat = radians(lat_deg)
    lon = radians(lon_deg)
    return (cos(lat) * cos(lon), cos(lat) * sin(lon), sin(lat))


def unit_vector_to_latlng(v: tuple[float, float, float]) -> tuple[float, float]:
    """Convert a unit 3-vector back to (lat_deg, lon_deg)."""
    lat = degrees(asin(max(-1.0, min(1.0, v[2]))))
    lon = degrees(atan2(v[1], v[0]))
    return (lat, lon)


def _dot(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(
    a: tuple[float, float, float], b: tuple[float, float, float]
) -> tuple[float, float, float]:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _norm(v: tuple[float, float, float]) -> float:
    return sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)


def _normalize(v: tuple[float, float, float]) -> tuple[float, float, float]:
    n = _norm(v)
    if n == 0.0:
        return v
    return (v[0] / n, v[1] / n, v[2] / n)


def _sub(
    a: tuple[float, float, float], b: tuple[float, float, float]
) -> tuple[float, float, float]:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _scale(v: tuple[float, float, float], s: float) -> tuple[float, float, float]:
    return (v[0] * s, v[1] * s, v[2] * s)


def closest_point_on_edge(
    p: tuple[float, float, float],
    a: tuple[float, float, float],
    b: tuple[float, float, float],
) -> tuple[float, float]:
    """Return (lat, lon) of the closest point on the great-circle arc [a, b] to p.

    All inputs are unit 3-vectors.
    """
    # Degenerate arc: a == b
    n = _cross(a, b)
    n_norm = _norm(n)
    _EPSILON = 1e-10
    if n_norm < _EPSILON:
        return unit_vector_to_latlng(a)

    # Project p onto the plane perpendicular to n
    d = _dot(p, n)
    # p_proj = p - dot(p, n)*n (then normalised)
    n_unit = _normalize(n)
    scaled_n = _scale(n_unit, _dot(p, n_unit))
    p_proj_raw = _sub(p, scaled_n)
    p_proj_norm = _norm(p_proj_raw)
    if p_proj_norm < _EPSILON:
        # p is along the arc's normal (pole of the arc's great circle); pick closer endpoint
        if _dot(a, p) >= _dot(b, p):
            return unit_vector_to_latlng(a)
        return unit_vector_to_latlng(b)

    p_proj = _normalize(p_proj_raw)

    # Check if p_proj lies within the arc [a, b]
    # p_proj is within arc if cross(a, p_proj)·n >= 0 and cross(p_proj, b)·n >= 0
    cross_ap = _cross(a, p_proj)
    cross_pb = _cross(p_proj, b)
    if _dot(cross_ap, n) >= 0 and _dot(cross_pb, n) >= 0:
        return unit_vector_to_latlng(p_proj)

    # Otherwise return whichever endpoint is closer (larger dot product = smaller angle)
    if _dot(a, p) >= _dot(b, p):
        return unit_vector_to_latlng(a)
    return unit_vector_to_latlng(b)


# ---------------------------------------------------------------------------
# Shape / stop building
# ---------------------------------------------------------------------------


def build_shape_points(shape_rows: list[dict]) -> list[ShapePoint]:
    """Build a list of ShapePoint from raw shape row dicts, sorted by sequence."""
    sorted_rows = sorted(shape_rows, key=lambda r: r["shape_pt_sequence"])
    geo_dist = 0.0
    user_dist = 0.0
    result: list[ShapePoint] = []
    prev_lat: Optional[float] = None
    prev_lon: Optional[float] = None

    for row in sorted_rows:
        lat = float(row["shape_pt_lat"])
        lon = float(row["shape_pt_lon"])
        if prev_lat is not None and prev_lon is not None:
            geo_dist += geo_distance_meters(prev_lat, prev_lon, lat, lon)
        raw_user = row.get("shape_dist_traveled") or 0.0
        user_dist = max(user_dist, float(raw_user))
        result.append(
            ShapePoint(
                geo_distance=geo_dist,
                user_distance=user_dist,
                lat=lat,
                lon=lon,
                uv=latlng_to_unit_vector(lat, lon),
            )
        )
        prev_lat, prev_lon = lat, lon

    return result


def resolve_stop_location(stop_id: str, stops_by_id: dict) -> tuple[float, float]:
    """Resolve a stop's coordinates, walking up to parent_station if needed."""
    stop = stops_by_id.get(stop_id)
    for _ in range(4):  # original + up to 3 parent hops
        if stop is None:
            return (0.0, 0.0)
        lat = stop.get("stop_lat")
        lon = stop.get("stop_lon")
        if lat is not None and lon is not None:
            return (float(lat), float(lon))
        parent_id = stop.get("parent_station")
        if not parent_id:
            return (0.0, 0.0)
        stop = stops_by_id.get(parent_id)
    return (0.0, 0.0)


def resolve_stop_name(stop_id: str, stops_by_id: dict) -> str:
    """Return the stop_name for stop_id, or empty string if not found."""
    stop = stops_by_id.get(stop_id)
    if stop is None:
        return ""
    return stop.get("stop_name") or ""


def build_stop_points(
    stop_times: list[dict],
    stops_by_id: dict,
    is_large_route: bool,
) -> list[StopPoint]:
    """Build a list of StopPoint from sorted stop_times dicts."""
    sorted_times = sorted(stop_times, key=lambda r: r["stop_sequence"])
    n = len(sorted_times)
    result: list[StopPoint] = []
    for i, st in enumerate(sorted_times):
        lat, lon = resolve_stop_location(st["stop_id"], stops_by_id)
        user_dist = st.get("shape_dist_traveled") or 0.0
        is_large = is_large_route and (i == 0 or i == n - 1)
        result.append(
            StopPoint(
                lat=lat,
                lon=lon,
                user_distance=float(user_dist),
                stop_time_row=st,
                is_large_station=is_large,
            )
        )
    return result


def compute_trip_hash(stop_times: list[dict]) -> bytes:
    """Stable hash over (stop_id, shape_dist_traveled) sequence."""
    h = hashlib.sha256()
    h.update(struct.pack(">I", len(stop_times)))
    for st in stop_times:
        sid = (st.get("stop_id") or "").encode("utf-8")
        h.update(struct.pack(">I", len(sid)))
        h.update(sid)
        dist = st.get("shape_dist_traveled") or 0.0
        h.update(struct.pack(">d", float(dist)))
    return h.digest()


# ---------------------------------------------------------------------------
# Matching helpers
# ---------------------------------------------------------------------------


def _seg_fraction(
    p: tuple[float, float, float],
    a: tuple[float, float, float],
    b: tuple[float, float, float],
) -> float:
    """Estimate fraction along segment [a, b] for the closest point to p.

    Uses dot products on the unit sphere as a proxy for arc length.
    """
    # Project p onto the plane of the arc
    n = _cross(a, b)
    n_norm = _norm(n)
    _EPSILON = 1e-10
    if n_norm < _EPSILON:
        return 0.0

    n_unit = _normalize(n)
    scaled_n = _scale(n_unit, _dot(p, n_unit))
    p_proj_raw = _sub(p, scaled_n)
    p_proj_norm = _norm(p_proj_raw)
    if p_proj_norm < _EPSILON:
        return 0.0

    p_proj = _normalize(p_proj_raw)

    # Check that p_proj is within the arc
    cross_ap = _cross(a, p_proj)
    cross_pb = _cross(p_proj, b)
    if _dot(cross_ap, n) >= 0 and _dot(cross_pb, n) >= 0:
        # Fraction: angle(a, p_proj) / angle(a, b)
        dot_ab = max(-1.0, min(1.0, _dot(a, b)))
        angle_ab = math.acos(dot_ab)
        if angle_ab < _EPSILON:
            return 0.0
        dot_ap = max(-1.0, min(1.0, _dot(a, p_proj)))
        angle_ap = math.acos(dot_ap)
        return max(0.0, min(1.0, angle_ap / angle_ab))

    # p_proj outside arc — check endpoints
    if _dot(a, p) >= _dot(b, p):
        return 0.0
    return 1.0


def find_potential_matches(
    stop: StopPoint,
    shape_points: list[ShapePoint],
    max_dist: float,
) -> list[CandidateMatch]:
    """Find candidate matches as local minima within max_dist along the shape."""
    if len(shape_points) == 1:
        sp = shape_points[0]
        dist = geo_distance_meters(stop.lat, stop.lon, sp.lat, sp.lon)
        if dist <= max_dist:
            return [
                CandidateMatch(
                    geo_distance_to_shape=dist,
                    shape_geo_distance=sp.geo_distance,
                    lat=sp.lat,
                    lon=sp.lon,
                )
            ]
        return []

    p = latlng_to_unit_vector(stop.lat, stop.lon)

    # Precompute per-stop bounding-box slack to quickly skip distant segments.
    _METERS_PER_DEG = 111_320.0
    lat_slack = max_dist / _METERS_PER_DEG
    lon_slack = max_dist / (_METERS_PER_DEG * max(cos(radians(abs(stop.lat))), 0.01))

    in_close_run = False
    run_best_dist = math.inf
    run_best_match: Optional[CandidateMatch] = None
    matches: list[CandidateMatch] = []

    for i in range(len(shape_points) - 1):
        a = shape_points[i]
        b = shape_points[i + 1]

        # Fast bbox rejection before expensive spherical geometry.
        if (
            stop.lat < min(a.lat, b.lat) - lat_slack
            or stop.lat > max(a.lat, b.lat) + lat_slack
            or stop.lon < min(a.lon, b.lon) - lon_slack
            or stop.lon > max(a.lon, b.lon) + lon_slack
        ):
            if in_close_run:
                assert run_best_match is not None
                matches.append(run_best_match)
                in_close_run = False
                run_best_dist = math.inf
                run_best_match = None
            continue

        av = a.uv
        bv = b.uv
        closest_lat, closest_lon = closest_point_on_edge(p, av, bv)
        dist = geo_distance_meters(stop.lat, stop.lon, closest_lat, closest_lon)

        # Interpolate geo_distance at matched point
        frac = _seg_fraction(p, av, bv)
        matched_geo_dist = a.geo_distance + frac * (b.geo_distance - a.geo_distance)

        candidate = CandidateMatch(
            geo_distance_to_shape=dist,
            shape_geo_distance=matched_geo_dist,
            lat=closest_lat,
            lon=closest_lon,
        )

        if dist <= max_dist:
            if not in_close_run:
                in_close_run = True
                run_best_dist = dist
                run_best_match = candidate
            elif dist < run_best_dist:
                run_best_dist = dist
                run_best_match = candidate
        else:
            if in_close_run:
                assert run_best_match is not None
                matches.append(run_best_match)
                in_close_run = False
                run_best_dist = math.inf
                run_best_match = None

    if in_close_run and run_best_match is not None:
        matches.append(run_best_match)

    return matches


def find_closest_on_shape(
    stop: StopPoint,
    shape_points: list[ShapePoint],
) -> CandidateMatch:
    """Scan all edges and return globally closest point (no distance limit)."""
    if len(shape_points) == 1:
        sp = shape_points[0]
        dist = geo_distance_meters(stop.lat, stop.lon, sp.lat, sp.lon)
        return CandidateMatch(
            geo_distance_to_shape=dist,
            shape_geo_distance=sp.geo_distance,
            lat=sp.lat,
            lon=sp.lon,
        )

    p = latlng_to_unit_vector(stop.lat, stop.lon)
    best_dist = math.inf
    best_match: Optional[CandidateMatch] = None

    for i in range(len(shape_points) - 1):
        a = shape_points[i]
        b = shape_points[i + 1]
        av = a.uv
        bv = b.uv
        closest_lat, closest_lon = closest_point_on_edge(p, av, bv)
        dist = geo_distance_meters(stop.lat, stop.lon, closest_lat, closest_lon)
        if dist < best_dist:
            best_dist = dist
            frac = _seg_fraction(p, av, bv)
            matched_geo_dist = a.geo_distance + frac * (b.geo_distance - a.geo_distance)
            best_match = CandidateMatch(
                geo_distance_to_shape=dist,
                shape_geo_distance=matched_geo_dist,
                lat=closest_lat,
                lon=closest_lon,
            )

    assert best_match is not None
    return best_match


def find_best_assignment(
    candidates_per_stop: list[list[CandidateMatch]],
    stop_points: list[StopPoint],
) -> tuple[list[Optional[CandidateMatch]], list[tuple[int, CandidateMatch, CandidateMatch]]]:
    """DP assignment: pick one candidate per stop so shape_geo_distance is non-decreasing
    and total geo_distance_to_shape is minimised.

    Returns (assignment, out_of_order_problems).
    out_of_order_problems: list of (stop_idx, match_of_this_stop, match_of_prev_stop)
    """
    n = len(candidates_per_stop)
    if n == 0:
        return [], []

    # dp[j] = (min_cost, prev_candidate_idx or -1, prev_stop_idx or -1)
    # We keep per-stop tables.
    INF = math.inf

    # dp_table[i][j] = (cost, prev_j)
    dp_table: list[list[tuple[float, int]]] = []

    for i, candidates in enumerate(candidates_per_stop):
        row: list[tuple[float, int]] = []
        if i == 0:
            for c in candidates:
                row.append((c.geo_distance_to_shape, -1))
        else:
            prev_candidates = candidates_per_stop[i - 1]
            prev_row = dp_table[i - 1]
            for j, c in enumerate(candidates):
                best_cost = INF
                best_prev = -1
                for k, pc in enumerate(prev_candidates):
                    if pc.shape_geo_distance <= c.shape_geo_distance:
                        cost = prev_row[k][0] + c.geo_distance_to_shape
                        if cost < best_cost:
                            best_cost = cost
                            best_prev = k
                row.append((best_cost, best_prev))
        dp_table.append(row)

    # Backtrack from the last stop
    assignment: list[Optional[CandidateMatch]] = [None] * n
    out_of_order: list[tuple[int, CandidateMatch, CandidateMatch]] = []

    # Find the best ending candidate for the last stop
    last_row = dp_table[n - 1]
    last_candidates = candidates_per_stop[n - 1]

    # For stops that have no valid predecessor, we still need to pick the best local candidate.
    # Find overall best (lowest cost that is finite, else pick lowest geo_dist_to_shape)
    best_j = -1
    best_cost = INF
    for j, (cost, _) in enumerate(last_row):
        if cost < best_cost:
            best_cost = cost
            best_j = j

    if best_j == -1:
        # All paths had INF cost — just pick index 0
        best_j = 0

    # Backtrack
    cur_j = best_j
    for i in range(n - 1, -1, -1):
        candidates = candidates_per_stop[i]
        if cur_j >= len(candidates):
            cur_j = 0
        assignment[i] = candidates[cur_j]
        prev_j = dp_table[i][cur_j][1]
        if i > 0:
            if prev_j == -1:
                # No valid predecessor — out-of-order
                # Pick the best local candidate for stop i-1 as the "prev match"
                prev_candidates = candidates_per_stop[i - 1]
                prev_row = dp_table[i - 1]
                best_prev = min(range(len(prev_candidates)), key=lambda k: prev_candidates[k].geo_distance_to_shape)
                out_of_order.append((i, candidates[cur_j], prev_candidates[best_prev]))
                cur_j = best_prev
            else:
                cur_j = prev_j

    return assignment, out_of_order


def interpolate_user_distance(
    user_dist: float,
    shape_points: list[ShapePoint],
    search_from: int,
) -> tuple[CandidateMatch, int]:
    """Find the segment where user_dist falls, starting from search_from.

    Interpolates lat/lon using 3D unit-vector linear interpolation (slerp-like).
    Returns (match, new_search_from).
    """
    n = len(shape_points)
    if n == 1:
        sp = shape_points[0]
        return (
            CandidateMatch(
                geo_distance_to_shape=0.0,
                shape_geo_distance=sp.geo_distance,
                lat=sp.lat,
                lon=sp.lon,
            ),
            0,
        )

    # Clamp search_from
    search_from = max(0, min(search_from, n - 2))

    # Find the segment
    seg_idx = search_from
    for i in range(search_from, n - 1):
        if shape_points[i].user_distance <= user_dist <= shape_points[i + 1].user_distance:
            seg_idx = i
            break
        # If we've passed user_dist, clamp to last valid segment
        if shape_points[i + 1].user_distance > user_dist:
            seg_idx = i
            break
        seg_idx = i  # keep advancing

    a = shape_points[seg_idx]
    b = shape_points[seg_idx + 1] if seg_idx + 1 < n else shape_points[seg_idx]

    if seg_idx + 1 >= n or a is b:
        match = CandidateMatch(
            geo_distance_to_shape=0.0,
            shape_geo_distance=a.geo_distance,
            lat=a.lat,
            lon=a.lon,
        )
        return match, seg_idx

    # Interpolation fraction based on user_distance
    denom = b.user_distance - a.user_distance
    if denom <= 0.0:
        frac = 0.0
    else:
        frac = max(0.0, min(1.0, (user_dist - a.user_distance) / denom))

    # Interpolate in 3D unit-vector space
    av = latlng_to_unit_vector(a.lat, a.lon)
    bv = latlng_to_unit_vector(b.lat, b.lon)
    # Linear interpolation in 3D then normalize
    interp = (
        av[0] + frac * (bv[0] - av[0]),
        av[1] + frac * (bv[1] - av[1]),
        av[2] + frac * (bv[2] - av[2]),
    )
    interp_norm = _normalize(interp)
    interp_lat, interp_lon = unit_vector_to_latlng(interp_norm)

    matched_geo_dist = a.geo_distance + frac * (b.geo_distance - a.geo_distance)

    match = CandidateMatch(
        geo_distance_to_shape=0.0,  # will be checked later against actual stop position
        shape_geo_distance=matched_geo_dist,
        lat=interp_lat,
        lon=interp_lon,
    )
    return match, seg_idx


# ---------------------------------------------------------------------------
# High-level matching
# ---------------------------------------------------------------------------


def match_using_geo_distance(
    stop_points: list[StopPoint],
    shape_points: list[ShapePoint],
    settings: MatchSettings,
) -> list[Problem]:
    """Geo-distance matching: find candidates, assign, detect issues."""
    problems: list[Problem] = []
    candidates_per_stop: list[list[CandidateMatch]] = []
    # Maps index in candidates_per_stop -> index in stop_points (skips too-far stops)
    candidate_stop_idx: list[int] = []

    for orig_idx, stop in enumerate(stop_points):
        max_dist = settings.max_distance_meters
        if stop.is_large_station:
            max_dist *= settings.large_station_multiplier

        candidates = find_potential_matches(stop, shape_points, max_dist)

        if len(candidates) > settings.potential_matches_threshold:
            # Find closest among candidates for reporting
            best = min(candidates, key=lambda c: c.geo_distance_to_shape)
            problems.append(
                Problem(
                    kind="too_many_matches",
                    data={
                        "stop_point": stop,
                        "match": best,
                        "match_count": len(candidates),
                    },
                )
            )

        if not candidates:
            closest = find_closest_on_shape(stop, shape_points)
            problems.append(
                Problem(
                    kind="too_far",
                    data={
                        "stop_point": stop,
                        "match": closest,
                    },
                )
            )
            # Do not add this stop to candidates_per_stop; continue checking remaining stops.
            continue

        candidates_per_stop.append(candidates)
        candidate_stop_idx.append(orig_idx)

    assignment, out_of_order = find_best_assignment(candidates_per_stop, stop_points)

    for cand_idx, match1, match2 in out_of_order:
        # cand_idx and cand_idx-1 are indices into candidates_per_stop; map back to stop_points
        stop1 = stop_points[candidate_stop_idx[cand_idx]]
        stop2 = stop_points[candidate_stop_idx[cand_idx - 1]]
        problems.append(
            Problem(
                kind="out_of_order",
                data={
                    "stop_point1": stop1,
                    "match1": match1,
                    "stop_point2": stop2,
                    "match2": match2,
                },
            )
        )

    return problems


def match_using_user_distance(
    stop_points: list[StopPoint],
    shape_points: list[ShapePoint],
    settings: MatchSettings,
) -> list[Problem]:
    """User-distance matching: interpolate by shape_dist_traveled, detect issues."""
    problems: list[Problem] = []
    candidates_per_stop: list[list[CandidateMatch]] = []
    search_from = 0

    for stop in stop_points:
        if stop.user_distance > 0.0:
            match, search_from = interpolate_user_distance(
                stop.user_distance, shape_points, search_from
            )
            # Update geo_distance_to_shape from actual stop position
            actual_dist = geo_distance_meters(stop.lat, stop.lon, match.lat, match.lon)
            match = CandidateMatch(
                geo_distance_to_shape=actual_dist,
                shape_geo_distance=match.shape_geo_distance,
                lat=match.lat,
                lon=match.lon,
            )
            candidates_per_stop.append([match])
        else:
            max_dist = settings.max_distance_meters
            if stop.is_large_station:
                max_dist *= settings.large_station_multiplier
            candidates_per_stop.append(
                find_potential_matches(stop, shape_points, max_dist)
            )

    assignment, out_of_order = find_best_assignment(candidates_per_stop, stop_points)

    for i, match in enumerate(assignment):
        if match is None:
            continue
        stop = stop_points[i]
        max_dist = settings.max_distance_meters
        if stop.is_large_station:
            max_dist *= settings.large_station_multiplier
        if match.geo_distance_to_shape > max_dist:
            problems.append(
                Problem(
                    kind="too_far_user_distance",
                    data={
                        "stop_point": stop,
                        "match": match,
                    },
                )
            )

    for stop_idx, match1, match2 in out_of_order:
        stop1 = stop_points[stop_idx]
        stop2 = stop_points[stop_idx - 1]
        problems.append(
            Problem(
                kind="out_of_order",
                data={
                    "stop_point1": stop1,
                    "match1": match1,
                    "stop_point2": stop2,
                    "match2": match2,
                },
            )
        )

    return problems
