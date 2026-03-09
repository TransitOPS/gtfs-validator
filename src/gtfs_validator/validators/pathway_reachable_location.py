"""Validator: pathway_reachable_location."""

from __future__ import annotations

from collections import defaultdict, deque

import polars as pl

from gtfs_validator.context import ValidationContext
from gtfs_validator.notices import Notice, Severity

_STOP = 0
_STATION = 1
_ENTRANCE = 2
_GENERIC_NODE = 3
_BOARDING_AREA = 4


def get_including_station(
    stop_id: str,
    stop_by_id: dict[str, dict],
) -> str | None:
    """Walk up the parent chain to find the enclosing station, capped at 3 iterations."""
    for _ in range(3):
        stop = stop_by_id.get(stop_id)
        if stop is None:
            return None
        if stop["location_type"] == _STATION:
            return stop_id
        parent = stop.get("parent_station")
        if not parent:
            return None
        stop_id = parent
    return None


def bfs(
    seeds: list[str],
    from_map: dict[str, list[tuple[str, int]]],
    to_map: dict[str, list[tuple[str, int]]],
    direction: str,  # "FROM_ENTRANCES" or "TO_EXITS"
) -> set[str]:
    """BFS over the pathway graph in the given direction."""
    visited: set[str] = set(seeds)
    queue: deque[str] = deque(seeds)
    while queue:
        curr = queue.popleft()
        # Follow forward edges (from_map gives edges curr -> neighbor)
        for neighbor, is_bidir in from_map.get(curr, []):
            if direction == "FROM_ENTRANCES" or is_bidir == 1:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        # Follow reverse edges (to_map gives edges neighbor -> curr, reversed)
        for neighbor, is_bidir in to_map.get(curr, []):
            if direction == "TO_EXITS" or is_bidir == 1:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
    return visited


def validate_pathway_reachable_location(
    feed: dict[str, pl.DataFrame],
    ctx: ValidationContext,
) -> list[Notice]:
    """Emit an error for each platform, generic node, or boarding area that is not
    fully reachable (both from an entrance and back to an entrance) within a station
    that has pathways."""
    stops = feed.get("stops")
    pathways = feed.get("pathways")
    if stops is None or stops.is_empty():
        return []
    if pathways is None or pathways.is_empty():
        return []

    # --- Preprocessing ---
    from_map: dict[str, list[tuple[str, int]]] = defaultdict(list)
    to_map: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for row in pathways.select(["from_stop_id", "to_stop_id", "is_bidirectional"]).iter_rows(named=True):
        from_id = row["from_stop_id"]
        to_id = row["to_stop_id"]
        is_bidir = row["is_bidirectional"] or 0
        if from_id and to_id:
            from_map[from_id].append((to_id, int(is_bidir)))
            to_map[to_id].append((from_id, int(is_bidir)))

    stop_by_id: dict[str, dict] = {}
    children_of: dict[str, list[str]] = defaultdict(list)
    for row in stops.iter_rows(named=True):
        stop_by_id[row["stop_id"]] = row
        parent = row.get("parent_station")
        if parent:
            children_of[parent].append(row["stop_id"])

    # --- Step 1: Collect all pathway endpoint stop IDs ---
    pathway_stop_ids = set(from_map.keys()) | set(to_map.keys())

    # --- Step 2: Identify stations that have pathways ---
    stations_with_pathways: set[str] = set()
    for sid in pathway_stop_ids:
        station_id = get_including_station(sid, stop_by_id)
        if station_id:
            stations_with_pathways.add(station_id)

    if not stations_with_pathways:
        return []

    # --- Steps 3 & 4: BFS from all entrances in both directions ---
    entrances = [
        row["stop_id"]
        for row in stops.filter(pl.col("location_type") == _ENTRANCE).iter_rows(named=True)
    ]

    locations_having_entrances = bfs(entrances, from_map, to_map, "FROM_ENTRANCES")
    locations_having_exits = bfs(entrances, from_map, to_map, "TO_EXITS")

    # --- Step 5: Emit notices ---
    notices: list[Notice] = []
    for row in stops.iter_rows(named=True):
        stop_id = row["stop_id"]
        location_type = row.get("location_type")

        # Determine including station
        station_id = get_including_station(stop_id, stop_by_id)
        if station_id is None:
            continue
        if station_id not in stations_with_pathways:
            continue

        # Filter to eligible location types
        if location_type == _STOP:
            # Skip if the platform has any boarding area children
            if children_of.get(stop_id):
                continue
        elif location_type in (_GENERIC_NODE, _BOARDING_AREA):
            pass  # eligible
        else:
            continue  # STATION, ENTRANCE, or unexpected type — skip

        has_entrance = stop_id in locations_having_entrances
        has_exit = stop_id in locations_having_exits

        if not (has_entrance and has_exit):
            notices.append(
                Notice(
                    code="pathway_unreachable_location",
                    severity=Severity.ERROR,
                    fields={
                        "csvRowNumber": row["csvRowNumber"],
                        "stopId": stop_id,
                        "stopName": row.get("stop_name"),
                        "locationType": location_type,
                        "parentStation": row.get("parent_station"),
                        "hasEntrance": has_entrance,
                        "hasExit": has_exit,
                    },
                )
            )

    return notices
