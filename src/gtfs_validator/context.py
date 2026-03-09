"""Validation context passed to all validators."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from types import MappingProxyType

import polars as pl

MAX_STOP_PARENT_HOPS = 3


@dataclass(frozen=True)
class StopLocationCache:
    """Precomputed stop metadata reused by validators."""

    resolved_latlng_by_stop_id: Mapping[str, tuple[float, float] | None]
    stop_name_by_stop_id: Mapping[str, str | None]
    existing_stop_ids: frozenset[str]


def build_stop_location_cache(
    stops: pl.DataFrame,
    max_parent_hops: int = MAX_STOP_PARENT_HOPS,
) -> StopLocationCache:
    """Build immutable lookups for stop name/existence and resolved coordinates."""
    cols = ["stop_id", "stop_lat", "stop_lon", "stop_name", "parent_station"]
    available_cols = [c for c in cols if c in stops.columns]
    rows = stops.select(available_cols).iter_rows(named=True)

    raw_by_stop_id: dict[str, dict[str, object]] = {}
    stop_name_by_stop_id: dict[str, str | None] = {}
    for row in rows:
        stop_id = row.get("stop_id")
        if stop_id is None:
            continue
        sid = str(stop_id)
        raw_by_stop_id[sid] = row
        stop_name_value = row.get("stop_name")
        stop_name_by_stop_id[sid] = (
            str(stop_name_value) if stop_name_value is not None else None
        )

    resolved_latlng_by_stop_id: dict[str, tuple[float, float] | None] = {}

    def resolve(stop_id: str) -> tuple[float, float] | None:
        cached = resolved_latlng_by_stop_id.get(stop_id, ...)
        if cached is not ...:
            return cached

        current_id = stop_id
        for _ in range(max_parent_hops):
            row = raw_by_stop_id.get(current_id)
            if row is None:
                resolved_latlng_by_stop_id[stop_id] = None
                return None
            lat = row.get("stop_lat")
            lon = row.get("stop_lon")
            if lat is not None and lon is not None:
                result = (float(str(lat)), float(str(lon)))
                resolved_latlng_by_stop_id[stop_id] = result
                return result
            parent = row.get("parent_station")
            if not parent:
                resolved_latlng_by_stop_id[stop_id] = None
                return None
            current_id = str(parent)

        resolved_latlng_by_stop_id[stop_id] = None
        return None

    for sid in raw_by_stop_id:
        resolve(sid)

    return StopLocationCache(
        resolved_latlng_by_stop_id=MappingProxyType(resolved_latlng_by_stop_id),
        stop_name_by_stop_id=MappingProxyType(stop_name_by_stop_id),
        existing_stop_ids=frozenset(raw_by_stop_id.keys()),
    )


@dataclass(frozen=True)
class ValidationContext:
    """Immutable context created once before validator dispatch."""

    country_code: str
    date_for_validation: date
    stop_location_cache: StopLocationCache | None = None
