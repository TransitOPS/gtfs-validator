"""Calendar utility helpers for service date resolution.

Shared by any validator that needs to resolve active service dates or
check service ID intersections.
"""

from __future__ import annotations

from datetime import date, timedelta

import polars as pl


def build_service_date_map(
    feed: dict[str, pl.DataFrame],
) -> dict[str, set[date]]:
    """Build a mapping from service_id to its full set of active dates.

    Combines calendar.txt (weekly patterns expanded into date sets) with
    calendar_dates.txt (additions via exception_type=1, removals via
    exception_type=2).

    Returns an empty dict if neither calendar nor calendar_dates is present.
    """
    result: dict[str, set[date]] = {}

    # Phase 1: Expand calendar.txt weekly patterns
    if "calendar" in feed and not feed["calendar"].is_empty():
        calendar = feed["calendar"]
        day_columns = [
            "monday", "tuesday", "wednesday", "thursday",
            "friday", "saturday", "sunday",
        ]

        for row in calendar.iter_rows(named=True):
            service_id = row["service_id"]
            start = _to_date(row["start_date"])
            end = _to_date(row["end_date"])
            if start is None or end is None:
                continue

            active_days: set[int] = set()
            for i, col in enumerate(day_columns):
                if row[col] == 1:
                    active_days.add(i)  # Monday=0 .. Sunday=6

            dates: set[date] = set()
            current = start
            while current <= end:
                if current.weekday() in active_days:
                    dates.add(current)
                current += timedelta(days=1)

            result[service_id] = dates

    # Phase 2: Apply calendar_dates.txt exceptions
    if "calendar_dates" in feed and not feed["calendar_dates"].is_empty():
        for row in feed["calendar_dates"].iter_rows(named=True):
            service_id = row["service_id"]
            d = _to_date(row["date"])
            exc_type = _to_int(row["exception_type"])
            if d is None or exc_type is None:
                continue

            if service_id not in result:
                result[service_id] = set()

            if exc_type == 1:  # added
                result[service_id].add(d)
            elif exc_type == 2:  # removed
                result[service_id].discard(d)

    return result


def _to_date(value: object) -> date | None:
    """Parse GTFS date values from either date objects or YYYYMMDD strings."""
    if isinstance(value, date):
        return value
    if isinstance(value, str) and len(value) == 8 and value.isdigit():
        try:
            return date(int(value[0:4]), int(value[4:6]), int(value[6:8]))
        except ValueError:
            return None
    return None


def _to_int(value: object) -> int | None:
    """Parse integer-like values used in CSV-backed GTFS tables."""
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


class ServiceIdIntersectionCache:
    """Cached pairwise service-date intersection lookups.

    Wraps a ``dict[str, set[date]]`` and caches the first intersecting
    date (or None) for each queried pair of service IDs.
    """

    def __init__(self, service_dates: dict[str, set[date]]) -> None:
        self._service_dates = service_dates
        self._cache: dict[tuple[str, str], date | None] = {}

    def first_intersecting_date(
        self, service_id_a: str, service_id_b: str,
    ) -> date | None:
        """Return the earliest date active in both services, or None."""
        # Normalize key ordering for cache symmetry
        key = (min(service_id_a, service_id_b), max(service_id_a, service_id_b))

        if key in self._cache:
            return self._cache[key]

        dates_a = self._service_dates.get(service_id_a, set())
        dates_b = self._service_dates.get(service_id_b, set())

        common = dates_a & dates_b
        if common:
            result = min(common)
        else:
            result = None

        self._cache[key] = result
        return result
