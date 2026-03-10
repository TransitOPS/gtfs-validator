"""Geometry and algorithm helpers for ShapeToStopMatchingValidator.

Scalar functions are kept for clarity, testability, and single-point edge
cases.  Hot-path callers use the numpy-vectorised variants below.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from math import asin, atan2, cos, degrees, radians, sin, sqrt
from typing import Optional

import numpy as np

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
class ShapeArrays:
    """Numpy array representation of a shape for vectorised geometry."""
    lats: np.ndarray           # (N,) float64
    lons: np.ndarray           # (N,) float64
    geo_distances: np.ndarray  # (N,) float64 cumulative metres
    user_distances: np.ndarray # (N,) float64 cumulative
    uvs: np.ndarray            # (N, 3) float64 unit vectors
    # Precomputed per-segment bounding boxes (N-1,)
    seg_min_lat: np.ndarray
    seg_max_lat: np.ndarray
    seg_min_lon: np.ndarray
    seg_max_lon: np.ndarray


@dataclass
class ShapeSpatialIndex:
    """Grid index mapping lat/lon cells to segment ids."""

    cell_size_deg: float
    segment_cells: dict[tuple[int, int], list[int]]
    fallback_segments: list[int]


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
    """Build a list of ShapePoint from shape row dicts (pre-sorted by shape_pt_sequence)."""
    geo_dist = 0.0
    user_dist = 0.0
    result: list[ShapePoint] = []
    prev_uv: tuple[float, float, float] | None = None

    for row in shape_rows:
        lat = float(row["shape_pt_lat"])
        lon = float(row["shape_pt_lon"])
        cur_uv = latlng_to_unit_vector(lat, lon)
        if prev_uv is not None:
            dot = max(-1.0, min(1.0, _dot(prev_uv, cur_uv)))
            geo_dist += math.acos(dot) * EARTH_RADIUS_METERS
        raw_user = row.get("shape_dist_traveled") or 0.0
        user_dist = max(user_dist, float(raw_user))
        result.append(
            ShapePoint(
                geo_distance=geo_dist,
                user_distance=user_dist,
                lat=lat,
                lon=lon,
                uv=cur_uv,
            )
        )
        prev_uv = cur_uv

    return result


def build_shape_arrays(shape_points: list[ShapePoint]) -> ShapeArrays:
    """Convert a list of ShapePoint into contiguous numpy arrays for vectorised ops."""
    n = len(shape_points)
    lats = np.array([sp.lat for sp in shape_points], dtype=np.float64)
    lons = np.array([sp.lon for sp in shape_points], dtype=np.float64)
    geo_distances = np.array([sp.geo_distance for sp in shape_points], dtype=np.float64)
    user_distances = np.array([sp.user_distance for sp in shape_points], dtype=np.float64)

    # Unit vectors (N, 3)
    lat_r = np.radians(lats)
    lon_r = np.radians(lons)
    cos_lat = np.cos(lat_r)
    uvs = np.column_stack([cos_lat * np.cos(lon_r), cos_lat * np.sin(lon_r), np.sin(lat_r)])

    # Segment bounding boxes (N-1,)
    if n >= 2:
        seg_min_lat = np.minimum(lats[:-1], lats[1:])
        seg_max_lat = np.maximum(lats[:-1], lats[1:])
        seg_min_lon = np.minimum(lons[:-1], lons[1:])
        seg_max_lon = np.maximum(lons[:-1], lons[1:])
    else:
        seg_min_lat = seg_max_lat = seg_min_lon = seg_max_lon = np.empty(0)

    return ShapeArrays(
        lats=lats,
        lons=lons,
        geo_distances=geo_distances,
        user_distances=user_distances,
        uvs=uvs,
        seg_min_lat=seg_min_lat,
        seg_max_lat=seg_max_lat,
        seg_min_lon=seg_min_lon,
        seg_max_lon=seg_max_lon,
    )


# ---------------------------------------------------------------------------
# Vectorised geometry helpers (numpy)
# ---------------------------------------------------------------------------

_EPSILON_VEC = 1e-10
# Minimum number of shape segments to justify numpy overhead.
# Below this, the scalar Python path is faster due to lower per-call overhead.
_VEC_MIN_SEGMENTS = 512
# Minimum per-call geometry workload (segments * stops) to justify numpy path.
_VEC_MIN_WORK_ITEMS = 100_000

# Scalar spatial index tuning.
_SPATIAL_INDEX_MIN_SEGMENTS = 128
_SPATIAL_CELL_SIZE_DEG = 0.01
_SPATIAL_MAX_CELLS_PER_SEG = 256


def _haversine_meters_vec(
    lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray,
) -> np.ndarray:
    """Vectorised haversine distance in metres."""
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = phi2 - phi1
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
    return 2 * np.arctan2(np.sqrt(a), np.sqrt(np.maximum(0.0, 1.0 - a))) * EARTH_RADIUS_METERS


def _closest_points_on_edges_vec(
    p_uv: np.ndarray,
    a_uvs: np.ndarray,
    b_uvs: np.ndarray,
) -> np.ndarray:
    """Vectorised closest-point-on-arc for M edges.

    Parameters
    ----------
    p_uv : (3,) unit vector of the query point
    a_uvs : (M, 3) start-point unit vectors
    b_uvs : (M, 3) end-point unit vectors

    Returns
    -------
    (M, 3) unit vectors of the closest point on each arc
    """
    # Cross product n = a × b  (M, 3)
    n = np.cross(a_uvs, b_uvs)
    n_norm = np.linalg.norm(n, axis=1, keepdims=True)  # (M, 1)
    degenerate = (n_norm.ravel() < _EPSILON_VEC)

    # Normalise n (safe — degenerate arcs handled separately)
    n_safe = np.divide(
        n,
        n_norm,
        out=np.zeros_like(n),
        where=n_norm > _EPSILON_VEC,
    )

    # Project p onto the great-circle plane: p_proj = p - (p·n_unit)*n_unit
    p_dot_n = (p_uv[np.newaxis, :] * n_safe).sum(axis=1, keepdims=True)  # (M, 1)
    p_proj_raw = p_uv[np.newaxis, :] - p_dot_n * n_safe  # (M, 3)
    p_proj_norm = np.linalg.norm(p_proj_raw, axis=1, keepdims=True)  # (M, 1)
    pole = (p_proj_norm.ravel() < _EPSILON_VEC)

    # Normalise p_proj
    p_proj = np.divide(
        p_proj_raw,
        p_proj_norm,
        out=np.zeros_like(p_proj_raw),
        where=p_proj_norm > _EPSILON_VEC,
    )

    # Check if p_proj lies within arc: cross(a, p_proj)·n >= 0 AND cross(p_proj, b)·n >= 0
    cross_ap = np.cross(a_uvs, p_proj)
    cross_pb = np.cross(p_proj, b_uvs)
    dot_ap_n = (cross_ap * n).sum(axis=1)
    dot_pb_n = (cross_pb * n).sum(axis=1)
    within_arc = (dot_ap_n >= 0) & (dot_pb_n >= 0)

    # Dot products for endpoint selection
    dot_a_p = (a_uvs * p_uv[np.newaxis, :]).sum(axis=1)
    dot_b_p = (b_uvs * p_uv[np.newaxis, :]).sum(axis=1)
    a_closer = dot_a_p >= dot_b_p

    # Build result: pick p_proj if within arc, else closer endpoint
    endpoint = np.where(a_closer[:, np.newaxis], a_uvs, b_uvs)
    result = np.where(within_arc[:, np.newaxis], p_proj, endpoint)

    # Handle degenerate arcs (a == b) and pole cases
    result[degenerate] = a_uvs[degenerate]
    pole_mask = pole & ~degenerate
    result[pole_mask] = endpoint[pole_mask]

    return result


def _uv_to_latlng_vec(uvs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert (M, 3) unit vectors to (lats, lons) arrays in degrees."""
    lats = np.degrees(np.arcsin(np.clip(uvs[:, 2], -1.0, 1.0)))
    lons = np.degrees(np.arctan2(uvs[:, 1], uvs[:, 0]))
    return lats, lons


def _seg_fraction_vec(
    p_uv: np.ndarray,
    a_uvs: np.ndarray,
    b_uvs: np.ndarray,
) -> np.ndarray:
    """Vectorised segment fraction for M edges. Returns (M,) array in [0, 1]."""
    n = np.cross(a_uvs, b_uvs)
    n_norm = np.linalg.norm(n, axis=1, keepdims=True)
    degenerate = (n_norm.ravel() < _EPSILON_VEC)

    n_safe = np.divide(
        n,
        n_norm,
        out=np.zeros_like(n),
        where=n_norm > _EPSILON_VEC,
    )
    p_dot_n = (p_uv[np.newaxis, :] * n_safe).sum(axis=1, keepdims=True)
    p_proj_raw = p_uv[np.newaxis, :] - p_dot_n * n_safe
    p_proj_norm = np.linalg.norm(p_proj_raw, axis=1, keepdims=True)
    small_proj = (p_proj_norm.ravel() < _EPSILON_VEC)

    p_proj = np.divide(
        p_proj_raw,
        p_proj_norm,
        out=np.zeros_like(p_proj_raw),
        where=p_proj_norm > _EPSILON_VEC,
    )

    # Check within arc
    cross_ap = np.cross(a_uvs, p_proj)
    cross_pb = np.cross(p_proj, b_uvs)
    dot_ap_n = (cross_ap * n).sum(axis=1)
    dot_pb_n = (cross_pb * n).sum(axis=1)
    within = (dot_ap_n >= 0) & (dot_pb_n >= 0)

    # Angle fractions
    dot_ab = np.clip((a_uvs * b_uvs).sum(axis=1), -1.0, 1.0)
    angle_ab = np.arccos(dot_ab)
    dot_ap = np.clip((a_uvs * p_proj).sum(axis=1), -1.0, 1.0)
    angle_ap = np.arccos(dot_ap)

    ratio = np.divide(
        angle_ap,
        angle_ab,
        out=np.zeros_like(angle_ap),
        where=angle_ab > _EPSILON_VEC,
    )
    frac = np.where(angle_ab > _EPSILON_VEC, np.clip(ratio, 0.0, 1.0), 0.0)

    # Outside arc: 0 or 1 based on closer endpoint
    dot_a_p = (a_uvs * p_uv[np.newaxis, :]).sum(axis=1)
    dot_b_p = (b_uvs * p_uv[np.newaxis, :]).sum(axis=1)
    outside_frac = np.where(dot_a_p >= dot_b_p, 0.0, 1.0)

    result = np.where(within, frac, outside_frac)
    result[degenerate | small_proj] = 0.0
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


def build_shape_spatial_index(shape_points: list[ShapePoint]) -> ShapeSpatialIndex | None:
    """Build a coarse lat/lon grid index over shape segments for scalar matching."""
    num_segments = len(shape_points) - 1
    if num_segments < _SPATIAL_INDEX_MIN_SEGMENTS:
        return None

    cell_size = _SPATIAL_CELL_SIZE_DEG
    inv_cell = 1.0 / cell_size
    segment_cells: dict[tuple[int, int], list[int]] = {}
    fallback_segments: list[int] = []

    for i in range(num_segments):
        a = shape_points[i]
        b = shape_points[i + 1]

        min_lat = a.lat if a.lat <= b.lat else b.lat
        max_lat = b.lat if a.lat <= b.lat else a.lat
        min_lon = a.lon if a.lon <= b.lon else b.lon
        max_lon = b.lon if a.lon <= b.lon else a.lon

        min_lat_cell = int(math.floor(min_lat * inv_cell))
        max_lat_cell = int(math.floor(max_lat * inv_cell))
        min_lon_cell = int(math.floor(min_lon * inv_cell))
        max_lon_cell = int(math.floor(max_lon * inv_cell))

        cell_span = (max_lat_cell - min_lat_cell + 1) * (max_lon_cell - min_lon_cell + 1)
        if cell_span > _SPATIAL_MAX_CELLS_PER_SEG:
            fallback_segments.append(i)
            continue

        for lat_cell in range(min_lat_cell, max_lat_cell + 1):
            for lon_cell in range(min_lon_cell, max_lon_cell + 1):
                key = (lat_cell, lon_cell)
                bucket = segment_cells.get(key)
                if bucket is None:
                    segment_cells[key] = [i]
                else:
                    bucket.append(i)

    return ShapeSpatialIndex(
        cell_size_deg=cell_size,
        segment_cells=segment_cells,
        fallback_segments=fallback_segments,
    )


def _query_shape_spatial_index(
    index: ShapeSpatialIndex,
    stop: StopPoint,
    max_dist: float,
) -> list[int]:
    """Return sorted candidate segment ids that may be within max_dist of stop."""
    meters_per_deg = 111_320.0
    lat_slack = max_dist / meters_per_deg
    lon_slack = max_dist / (meters_per_deg * max(cos(radians(abs(stop.lat))), 0.01))

    min_lat = stop.lat - lat_slack
    max_lat = stop.lat + lat_slack
    min_lon = stop.lon - lon_slack
    max_lon = stop.lon + lon_slack

    inv_cell = 1.0 / index.cell_size_deg
    min_lat_cell = int(math.floor(min_lat * inv_cell))
    max_lat_cell = int(math.floor(max_lat * inv_cell))
    min_lon_cell = int(math.floor(min_lon * inv_cell))
    max_lon_cell = int(math.floor(max_lon * inv_cell))

    candidates: set[int] = set(index.fallback_segments)
    for lat_cell in range(min_lat_cell, max_lat_cell + 1):
        for lon_cell in range(min_lon_cell, max_lon_cell + 1):
            segs = index.segment_cells.get((lat_cell, lon_cell))
            if segs:
                candidates.update(segs)

    if not candidates:
        return []
    return sorted(candidates)


def build_stop_points(
    stop_times: list[dict],
    stops_by_id: dict,
    is_large_route: bool,
    resolved_latlng: Optional[dict] = None,
) -> list[StopPoint]:
    """Build a list of StopPoint from stop_times dicts (pre-sorted by stop_sequence).

    If *resolved_latlng* is provided (e.g. from StopLocationCache), coordinates
    are looked up directly without traversing parent_station chains.
    """
    n = len(stop_times)
    result: list[StopPoint] = []
    for i, st in enumerate(stop_times):
        stop_id = st["stop_id"]
        if resolved_latlng is not None:
            ll = resolved_latlng.get(stop_id)
            lat, lon = (ll[0], ll[1]) if ll is not None else (0.0, 0.0)
        else:
            lat, lon = resolve_stop_location(stop_id, stops_by_id)
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


def compute_trip_hash(stop_times: list[dict]) -> tuple:
    """Stable signature over (stop_id, shape_dist_traveled) sequence for deduplication."""
    return tuple(
        (st.get("stop_id") or "", float(st.get("shape_dist_traveled") or 0.0))
        for st in stop_times
    )


def _candidate_cache_key(stop: StopPoint, max_dist: float) -> tuple[str, float, float, float]:
    """Stable cache key for per-shape stop-to-candidate matching."""
    stop_id = str(stop.stop_time_row.get("stop_id") or "")
    return (
        stop_id,
        round(stop.lat, 7),
        round(stop.lon, 7),
        round(max_dist, 3),
    )


def _closest_cache_key(stop: StopPoint) -> tuple[str, float, float]:
    """Stable cache key for per-shape closest-point lookup."""
    stop_id = str(stop.stop_time_row.get("stop_id") or "")
    return (stop_id, round(stop.lat, 7), round(stop.lon, 7))


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
    shape_arrays: Optional[ShapeArrays] = None,
    spatial_index: ShapeSpatialIndex | None = None,
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

    # Vectorised path for large shapes.
    if shape_arrays is not None and len(shape_points) - 1 >= _VEC_MIN_SEGMENTS:
        return _find_potential_matches_vec(stop, shape_arrays, max_dist)

    # --- Scalar path (inlined for zero overhead) ---
    p = latlng_to_unit_vector(stop.lat, stop.lon)

    _METERS_PER_DEG = 111_320.0
    lat_slack = max_dist / _METERS_PER_DEG
    lon_slack = max_dist / (_METERS_PER_DEG * max(cos(radians(abs(stop.lat))), 0.01))

    in_close_run = False
    run_best_dist = math.inf
    run_best_geo_dist = 0.0
    run_best_lat = 0.0
    run_best_lon = 0.0
    matches: list[CandidateMatch] = []

    seg_ids = (
        _query_shape_spatial_index(spatial_index, stop, max_dist)
        if spatial_index is not None
        else list(range(len(shape_points) - 1))
    )
    prev_seg_id = -1
    for i in seg_ids:
        if in_close_run and prev_seg_id >= 0 and i != prev_seg_id + 1:
            matches.append(
                CandidateMatch(
                    geo_distance_to_shape=run_best_dist,
                    shape_geo_distance=run_best_geo_dist,
                    lat=run_best_lat,
                    lon=run_best_lon,
                )
            )
            in_close_run = False
            run_best_dist = math.inf

        a = shape_points[i]
        b = shape_points[i + 1]
        a_lat = a.lat
        b_lat = b.lat
        a_lon = a.lon
        b_lon = b.lon
        if a_lat <= b_lat:
            min_lat = a_lat
            max_lat = b_lat
        else:
            min_lat = b_lat
            max_lat = a_lat
        if a_lon <= b_lon:
            min_lon = a_lon
            max_lon = b_lon
        else:
            min_lon = b_lon
            max_lon = a_lon

        if (
            stop.lat < min_lat - lat_slack
            or stop.lat > max_lat + lat_slack
            or stop.lon < min_lon - lon_slack
            or stop.lon > max_lon + lon_slack
        ):
            if in_close_run:
                matches.append(
                    CandidateMatch(
                        geo_distance_to_shape=run_best_dist,
                        shape_geo_distance=run_best_geo_dist,
                        lat=run_best_lat,
                        lon=run_best_lon,
                    )
                )
                in_close_run = False
                run_best_dist = math.inf
            continue

        av = a.uv
        bv = b.uv
        closest_lat, closest_lon = closest_point_on_edge(p, av, bv)
        dist = geo_distance_meters(stop.lat, stop.lon, closest_lat, closest_lon)

        if dist <= max_dist:
            frac = _seg_fraction(p, av, bv)
            matched_geo_dist = a.geo_distance + frac * (b.geo_distance - a.geo_distance)
            if not in_close_run:
                in_close_run = True
                run_best_dist = dist
                run_best_geo_dist = matched_geo_dist
                run_best_lat = closest_lat
                run_best_lon = closest_lon
            elif dist < run_best_dist:
                run_best_dist = dist
                run_best_geo_dist = matched_geo_dist
                run_best_lat = closest_lat
                run_best_lon = closest_lon
        else:
            if in_close_run:
                matches.append(
                    CandidateMatch(
                        geo_distance_to_shape=run_best_dist,
                        shape_geo_distance=run_best_geo_dist,
                        lat=run_best_lat,
                        lon=run_best_lon,
                    )
                )
                in_close_run = False
                run_best_dist = math.inf
        prev_seg_id = i

    if in_close_run:
        matches.append(
            CandidateMatch(
                geo_distance_to_shape=run_best_dist,
                shape_geo_distance=run_best_geo_dist,
                lat=run_best_lat,
                lon=run_best_lon,
            )
        )

    return matches


def _find_potential_matches_vec(
    stop: StopPoint,
    sa: ShapeArrays,
    max_dist: float,
) -> list[CandidateMatch]:
    """Numpy-vectorised implementation for large shapes."""
    num_segs = len(sa.lats) - 1

    _METERS_PER_DEG = 111_320.0
    lat_slack = max_dist / _METERS_PER_DEG
    lon_slack = max_dist / (_METERS_PER_DEG * max(cos(radians(abs(stop.lat))), 0.01))

    bbox_pass = (
        (stop.lat >= sa.seg_min_lat - lat_slack)
        & (stop.lat <= sa.seg_max_lat + lat_slack)
        & (stop.lon >= sa.seg_min_lon - lon_slack)
        & (stop.lon <= sa.seg_max_lon + lon_slack)
    )

    pass_idx = np.flatnonzero(bbox_pass)
    if pass_idx.size == 0:
        return []

    p_uv = np.array(latlng_to_unit_vector(stop.lat, stop.lon), dtype=np.float64)
    a_uvs = sa.uvs[pass_idx]
    b_uvs = sa.uvs[pass_idx + 1]
    closest_uvs = _closest_points_on_edges_vec(p_uv, a_uvs, b_uvs)
    closest_lats, closest_lons = _uv_to_latlng_vec(closest_uvs)
    dists = _haversine_meters_vec(
        np.full(pass_idx.size, stop.lat), np.full(pass_idx.size, stop.lon),
        closest_lats, closest_lons,
    )
    fracs = _seg_fraction_vec(p_uv, a_uvs, b_uvs)
    geo_d_a = sa.geo_distances[pass_idx]
    geo_d_b = sa.geo_distances[pass_idx + 1]
    matched_geo_dists = geo_d_a + fracs * (geo_d_b - geo_d_a)

    full_dists = np.full(num_segs, np.inf)
    full_dists[pass_idx] = dists

    is_close = full_dists <= max_dist
    padded = np.concatenate([[False], is_close, [False]])
    diff = np.diff(padded.astype(np.int8))
    run_starts = np.flatnonzero(diff == 1)
    run_ends = np.flatnonzero(diff == -1)

    matches: list[CandidateMatch] = []
    for rs, re in zip(run_starts, run_ends):
        run_dists = full_dists[rs:re]
        best_in_run = rs + int(np.argmin(run_dists))
        pos = np.searchsorted(pass_idx, best_in_run)
        matches.append(
            CandidateMatch(
                geo_distance_to_shape=float(dists[pos]),
                shape_geo_distance=float(matched_geo_dists[pos]),
                lat=float(closest_lats[pos]),
                lon=float(closest_lons[pos]),
            )
        )

    return matches


def find_closest_on_shape(
    stop: StopPoint,
    shape_points: list[ShapePoint],
    shape_arrays: Optional[ShapeArrays] = None,
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

    # Vectorised path for large shapes.
    if shape_arrays is not None and len(shape_points) - 1 >= _VEC_MIN_SEGMENTS:
        return _find_closest_on_shape_vec(stop, shape_arrays)

    # --- Scalar path (inlined for zero overhead) ---
    p = latlng_to_unit_vector(stop.lat, stop.lon)
    best_dist = math.inf
    best_geo_dist = 0.0
    best_lat = 0.0
    best_lon = 0.0

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
            best_geo_dist = matched_geo_dist
            best_lat = closest_lat
            best_lon = closest_lon

    return CandidateMatch(
        geo_distance_to_shape=best_dist,
        shape_geo_distance=best_geo_dist,
        lat=best_lat,
        lon=best_lon,
    )


def _find_closest_on_shape_vec(
    stop: StopPoint,
    sa: ShapeArrays,
) -> CandidateMatch:
    """Numpy-vectorised implementation for large shapes."""
    p_uv = np.array(latlng_to_unit_vector(stop.lat, stop.lon), dtype=np.float64)
    a_uvs = sa.uvs[:-1]
    b_uvs = sa.uvs[1:]
    closest_uvs = _closest_points_on_edges_vec(p_uv, a_uvs, b_uvs)
    closest_lats, closest_lons = _uv_to_latlng_vec(closest_uvs)
    num_segs = len(sa.lats) - 1
    dists = _haversine_meters_vec(
        np.full(num_segs, stop.lat), np.full(num_segs, stop.lon),
        closest_lats, closest_lons,
    )
    best_i = int(np.argmin(dists))
    frac = float(_seg_fraction_vec(p_uv, a_uvs[best_i:best_i+1], b_uvs[best_i:best_i+1])[0])
    matched_geo_dist = sa.geo_distances[best_i] + frac * (sa.geo_distances[best_i + 1] - sa.geo_distances[best_i])
    return CandidateMatch(
        geo_distance_to_shape=float(dists[best_i]),
        shape_geo_distance=float(matched_geo_dist),
        lat=float(closest_lats[best_i]),
        lon=float(closest_lons[best_i]),
    )


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

    # Fast path: all stops have at most one candidate, so DP is unnecessary.
    if all(len(candidates) <= 1 for candidates in candidates_per_stop):
        quick_assignment: list[Optional[CandidateMatch]] = []
        quick_out_of_order: list[tuple[int, CandidateMatch, CandidateMatch]] = []
        prev_match: Optional[CandidateMatch] = None
        for i, candidates in enumerate(candidates_per_stop):
            match = candidates[0] if candidates else None
            quick_assignment.append(match)
            if match is None:
                continue
            if prev_match is not None and prev_match.shape_geo_distance > match.shape_geo_distance:
                quick_out_of_order.append((i, match, prev_match))
            prev_match = match
        return quick_assignment, quick_out_of_order

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
    av = a.uv
    bv = b.uv
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
    shape_arrays: Optional[ShapeArrays] = None,
    spatial_index: ShapeSpatialIndex | None = None,
    candidates_cache: Optional[
        dict[tuple[str, float, float, float], list[CandidateMatch]]
    ] = None,
    closest_cache: Optional[dict[tuple[str, float, float], CandidateMatch]] = None,
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

        candidates_key: Optional[tuple[str, float, float, float]] = None
        candidates: Optional[list[CandidateMatch]] = None
        if candidates_cache is not None:
            candidates_key = _candidate_cache_key(stop, max_dist)
            candidates = candidates_cache.get(candidates_key)
        if candidates is None:
            candidates = find_potential_matches(
                stop,
                shape_points,
                max_dist,
                shape_arrays,
                spatial_index,
            )
            if candidates_cache is not None and candidates_key is not None:
                candidates_cache[candidates_key] = candidates

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
            closest_key: Optional[tuple[str, float, float]] = None
            closest: Optional[CandidateMatch] = None
            if closest_cache is not None:
                closest_key = _closest_cache_key(stop)
                closest = closest_cache.get(closest_key)
            if closest is None:
                closest = find_closest_on_shape(
                    stop,
                    shape_points,
                    shape_arrays,
                )
                if closest_cache is not None and closest_key is not None:
                    closest_cache[closest_key] = closest
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
    shape_arrays: Optional[ShapeArrays] = None,
    spatial_index: ShapeSpatialIndex | None = None,
    candidates_cache: Optional[
        dict[tuple[str, float, float, float], list[CandidateMatch]]
    ] = None,
) -> list[Problem]:
    """User-distance matching: interpolate by shape_dist_traveled, detect issues."""
    problems: list[Problem] = []
    candidates_per_stop: list[list[CandidateMatch]] = []
    search_from = 0
    for stop in stop_points:
        if stop.user_distance > 0.0:
            interp_match, search_from = interpolate_user_distance(
                stop.user_distance, shape_points, search_from
            )
            # Update geo_distance_to_shape from actual stop position
            actual_dist = geo_distance_meters(
                stop.lat, stop.lon, interp_match.lat, interp_match.lon
            )
            user_match = CandidateMatch(
                geo_distance_to_shape=actual_dist,
                shape_geo_distance=interp_match.shape_geo_distance,
                lat=interp_match.lat,
                lon=interp_match.lon,
            )
            candidates_per_stop.append([user_match])
        else:
            max_dist = settings.max_distance_meters
            if stop.is_large_station:
                max_dist *= settings.large_station_multiplier
            candidates_key: Optional[tuple[str, float, float, float]] = None
            candidates: Optional[list[CandidateMatch]] = None
            if candidates_cache is not None:
                candidates_key = _candidate_cache_key(stop, max_dist)
                candidates = candidates_cache.get(candidates_key)
            if candidates is None:
                candidates = find_potential_matches(
                    stop,
                    shape_points,
                    max_dist,
                    shape_arrays,
                    spatial_index,
                )
                if candidates_cache is not None and candidates_key is not None:
                    candidates_cache[candidates_key] = candidates
            candidates_per_stop.append(candidates)

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
