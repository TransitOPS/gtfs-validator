"""Validator: TransferDistanceValidator.

Checks that transfers between stops are not unreasonably far apart.
Emits a WARNING when distance > 10 km, and an INFO when distance > 2 km.
"""

from __future__ import annotations

import math

import polars as pl

from gtfs_validator.context import ValidationContext
from gtfs_validator.notices import Notice, Severity

EARTH_RADIUS_METERS: float = 6_371_010.0


def _haversine_meters(
    lat1: float, lon1: float, lat2: float, lon2: float
) -> float:
    """Great-circle distance in metres using the haversine formula (S2Earth)."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    )
    return 2 * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a))) * EARTH_RADIUS_METERS


def _resolve_stop_latlng(
    stop_id: str,
    stops_index: dict[str, tuple[float | None, float | None, str | None]],
) -> tuple[float, float]:
    """Resolve a stop_id to (lat, lon) by walking the parent_station chain.

    stops_index maps stop_id -> (stop_lat, stop_lon, parent_station).

    Returns (0.0, 0.0) if no coordinates are found within 3 hops, or if the
    stop_id is not present in the index. This matches the Java S2LatLng.CENTER
    fallback behavior exactly (including the known bug where an unresolvable
    stop produces a spurious distance computation from the origin rather than
    being skipped — do not fix this).
    """
    for _ in range(3):
        entry = stops_index.get(stop_id)
        if entry is None:
            return (0.0, 0.0)
        lat, lon, parent = entry
        if lat is not None and lon is not None:
            return (lat, lon)
        if parent is not None:
            stop_id = parent
        else:
            break
    return (0.0, 0.0)


def validate_transfer_distance(
    feed: dict[str, pl.DataFrame],
    ctx: ValidationContext,
) -> list[Notice]:
    """Validate that transfer distances between stops are not too large."""
    # Guard clauses
    if "transfers" not in feed or "stops" not in feed:
        return []

    transfers = feed["transfers"]
    stops = feed["stops"]

    if "from_stop_id" not in transfers.columns or "to_stop_id" not in transfers.columns:
        return []

    if transfers.is_empty():
        return []

    # Build stop coordinate + parent index once
    select_cols = ["stop_id", "stop_lat", "stop_lon"]
    if "parent_station" in stops.columns:
        select_cols.append("parent_station")

    stops_index: dict[str, tuple[float | None, float | None, str | None]] = {}
    for row in stops.select(select_cols).iter_rows(named=True):
        stops_index[row["stop_id"]] = (
            row["stop_lat"],
            row["stop_lon"],
            row.get("parent_station"),  # None when column absent
        )

    notices: list[Notice] = []

    for row in transfers.iter_rows(named=True):
        from_stop_id = row["from_stop_id"]
        to_stop_id = row["to_stop_id"]

        # Skip rows where either stop ID is null
        if from_stop_id is None or to_stop_id is None:
            continue

        from_lat, from_lon = _resolve_stop_latlng(from_stop_id, stops_index)
        to_lat, to_lon = _resolve_stop_latlng(to_stop_id, stops_index)

        distance_m = _haversine_meters(from_lat, from_lon, to_lat, to_lon)

        if distance_m > 10_000:
            notices.append(Notice(
                code="transfer_distance_too_large",
                severity=Severity.WARNING,
                fields={
                    "csv_row_number": row["csv_row_number"],
                    "from_stop_id": from_stop_id,
                    "to_stop_id": to_stop_id,
                    "distance_km": distance_m / 1_000,
                },
            ))
        elif distance_m > 2_000:
            notices.append(Notice(
                code="transfer_distance_above_2_km",
                severity=Severity.INFO,
                fields={
                    "csv_row_number": row["csv_row_number"],
                    "from_stop_id": from_stop_id,
                    "to_stop_id": to_stop_id,
                    "distance_km": distance_m / 1_000,
                },
            ))

    return notices
