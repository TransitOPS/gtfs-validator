"""Validator: service_gap — warns when there is a large gap in service coverage."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import polars as pl

from gtfs_validator.context import ValidationContext
from gtfs_validator.notices import Notice, Severity


MAX_GAP_DAYS: int = 13  # Gap must be strictly greater than this to emit a notice


@dataclass
class DateInterval:
    start: date
    end: date  # inclusive

    def length_in_days(self) -> int:
        return (self.end - self.start).days + 1


def _weekly_pattern(mon: int, tue: int, wed: int, thu: int, fri: int, sat: int, sun: int) -> int:
    """Convert the seven day columns to a 7-bit integer (bit 0 = Monday, ..., bit 6 = Sunday)."""
    days = [mon, tue, wed, thu, fri, sat, sun]
    pattern = 0
    for i, d in enumerate(days):
        pattern |= (int(d) & 1) << i
    return pattern


def _is_active_day(d: date, pattern: int) -> bool:
    """Check bit for a date's weekday (Python date.weekday() returns 0 = Monday)."""
    return bool((pattern >> d.weekday()) & 1)


def _add_interval(new_start: date, new_end: date, intervals: list[DateInterval]) -> None:
    """Merge a new [new_start, new_end] into the sorted interval list."""
    new_interval = DateInterval(new_start, new_end)
    # Find all existing intervals that overlap or are adjacent to new_interval
    absorbed: list[int] = []
    for i, iv in enumerate(intervals):
        # Two intervals overlap or are adjacent if:
        # iv.end + 1 day >= new_interval.start AND new_interval.end + 1 day >= iv.start
        if iv.end + timedelta(days=1) >= new_interval.start and new_interval.end + timedelta(days=1) >= iv.start:
            absorbed.append(i)

    if absorbed:
        # Merge all absorbed intervals with the new one
        merged_start = min(new_interval.start, *(intervals[i].start for i in absorbed))
        merged_end = max(new_interval.end, *(intervals[i].end for i in absorbed))
        # Remove absorbed intervals (in reverse order to preserve indices)
        for i in reversed(absorbed):
            intervals.pop(i)
        # Insert the merged interval in sorted position
        merged = DateInterval(merged_start, merged_end)
        _insert_sorted(merged, intervals)
    else:
        _insert_sorted(new_interval, intervals)


def _insert_sorted(interval: DateInterval, intervals: list[DateInterval]) -> None:
    """Insert interval into sorted list (ascending by start date)."""
    for i, iv in enumerate(intervals):
        if interval.start < iv.start:
            intervals.insert(i, interval)
            return
    intervals.append(interval)


def _add_date(d: date, intervals: list[DateInterval]) -> None:
    """Add a single date as a one-day interval."""
    _add_interval(d, d, intervals)


def _remove_date(d: date, intervals: list[DateInterval]) -> None:
    """Remove a single date from the interval list."""
    for i, iv in enumerate(intervals):
        if iv.start <= d <= iv.end:
            if d == iv.start and d == iv.end:
                intervals.pop(i)
            elif d == iv.start:
                iv.start = d + timedelta(days=1)
            elif d == iv.end:
                iv.end = d - timedelta(days=1)
            else:
                # Split I into [I.start, d - 1 day] and [d + 1 day, I.end]
                left = DateInterval(iv.start, d - timedelta(days=1))
                right = DateInterval(d + timedelta(days=1), iv.end)
                intervals.pop(i)
                intervals.insert(i, right)
                intervals.insert(i, left)
            return
    # Date not in any interval — no-op


def _apply_calendar_row(
    start_date: date,
    end_date: date,
    pattern: int,
    intervals: list[DateInterval],
) -> None:
    """Walk a date range using the weekly pattern and merge active-day runs into the interval set."""
    if pattern == 0:
        return  # no active days
    run_start: date | None = None
    current = start_date
    one_day = timedelta(days=1)
    while current <= end_date:
        if _is_active_day(current, pattern):
            if run_start is None:
                run_start = current
        else:
            if run_start is not None:
                _add_interval(run_start, current - one_day, intervals)
                run_start = None
        current += one_day
    if run_start is not None:
        _add_interval(run_start, end_date, intervals)


def _build_service_intervals(
    calendar: pl.DataFrame,
    calendar_dates: pl.DataFrame | None,
) -> dict[str, list[DateInterval]]:
    """Build per-service interval sets from calendar.txt and optionally calendar_dates.txt."""
    intervals: dict[str, list[DateInterval]] = {}

    # Phase 1: calendar.txt weekly patterns
    for row in calendar.iter_rows(named=True):
        sid = row["service_id"]
        pattern = _weekly_pattern(
            row["monday"],
            row["tuesday"],
            row["wednesday"],
            row["thursday"],
            row["friday"],
            row["saturday"],
            row["sunday"],
        )
        if sid not in intervals:
            intervals[sid] = []
        _apply_calendar_row(row["start_date"], row["end_date"], pattern, intervals[sid])

    # Phase 2: calendar_dates.txt exceptions (only for service_ids seen in calendar)
    if calendar_dates is not None and not calendar_dates.is_empty():
        for row in calendar_dates.iter_rows(named=True):
            sid = row["service_id"]
            if sid not in intervals:
                continue  # service not driven by calendar.txt — skip
            if row["exception_type"] == 1:   # SERVICE_ADDED
                _add_date(row["date"], intervals[sid])
            elif row["exception_type"] == 2:  # SERVICE_REMOVED
                _remove_date(row["date"], intervals[sid])

    return intervals


def _get_gaps(active_intervals: list[DateInterval]) -> list[DateInterval]:
    """Return the list of inactive spans between consecutive active intervals."""
    if len(active_intervals) < 2:
        return []
    gaps = []
    for i in range(1, len(active_intervals)):
        prev_end = active_intervals[i - 1].end
        next_start = active_intervals[i].start
        gap_start = prev_end + timedelta(days=1)
        gap_end = next_start - timedelta(days=1)
        if gap_start <= gap_end:
            gaps.append(DateInterval(gap_start, gap_end))
    return gaps


def validate_service_gap(
    feed: dict[str, pl.DataFrame],
    ctx: ValidationContext,
) -> list[Notice]:
    """Emit an info notice for each service with a gap of more than MAX_GAP_DAYS inactive days."""
    if "calendar" not in feed or feed["calendar"].is_empty():
        return []

    calendar = feed["calendar"].select(
        [
            "service_id",
            "start_date",
            "end_date",
            "monday",
            "tuesday",
            "wednesday",
            "thursday",
            "friday",
            "saturday",
            "sunday",
        ]
    )
    calendar_dates: pl.DataFrame | None = None
    if "calendar_dates" in feed and not feed["calendar_dates"].is_empty():
        calendar_dates = feed["calendar_dates"].select(
            ["service_id", "date", "exception_type"]
        )

    service_intervals = _build_service_intervals(calendar, calendar_dates)

    notices: list[Notice] = []

    for service_id, intervals in service_intervals.items():
        if len(intervals) < 2:
            continue
        for gap in _get_gaps(intervals):
            gap_length = gap.length_in_days()
            if gap_length > MAX_GAP_DAYS:
                # gap.start = prev_end + 1 day; gap.end = next_start - 1 day
                prev_end = gap.start - timedelta(days=1)
                next_start = gap.end + timedelta(days=1)
                notices.append(
                    Notice(
                        code="big_gap_in_service",
                        severity=Severity.INFO,
                        fields={
                            "service_id": service_id,
                            "gap_start_date": prev_end.isoformat(),
                            "gap_end_date": next_start.isoformat(),
                            "gap_duration_days": gap_length,
                        },
                    )
                )

    return notices
