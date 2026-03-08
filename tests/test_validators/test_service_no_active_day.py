"""Tests for validate_service_no_active_day."""

from __future__ import annotations

import datetime

import polars as pl
import pytest

from gtfs_validator.context import ValidationContext
from gtfs_validator.notices import Severity
from gtfs_validator.validators.service_no_active_day import validate_service_no_active_day

CTX = ValidationContext(
    country_code="US",
    date_for_validation=datetime.date(2024, 1, 1),
)

_CALENDAR_SCHEMA = {
    "csv_row_number": pl.Int64,
    "service_id": pl.Utf8,
    "monday": pl.Int64,
    "tuesday": pl.Int64,
    "wednesday": pl.Int64,
    "thursday": pl.Int64,
    "friday": pl.Int64,
    "saturday": pl.Int64,
    "sunday": pl.Int64,
}

_ALL_ZERO = {
    "monday": 0, "tuesday": 0, "wednesday": 0, "thursday": 0,
    "friday": 0, "saturday": 0, "sunday": 0,
}

_ALL_ONE = {
    "monday": 1, "tuesday": 1, "wednesday": 1, "thursday": 1,
    "friday": 1, "saturday": 1, "sunday": 1,
}


def make_calendar(rows: list[dict]) -> pl.DataFrame:
    """Build a calendar DataFrame from row dicts. Returns empty frame when rows=[]."""
    if not rows:
        return pl.DataFrame(schema=_CALENDAR_SCHEMA)
    return pl.DataFrame(rows, schema=_CALENDAR_SCHEMA)


def test_service_has_no_active_day_emits_notice() -> None:
    """service_1 with all zeros gets a notice; service_2 with monday=1 does not."""
    rows = [
        {"csv_row_number": 2, "service_id": "service_1", **_ALL_ZERO},
        {"csv_row_number": 3, "service_id": "service_2", **_ALL_ZERO, "monday": 1},
    ]
    feed = {"calendar": make_calendar(rows)}
    notices = validate_service_no_active_day(feed, CTX)

    assert len(notices) == 1
    notice = notices[0]
    assert notice.code == "service_has_no_active_day_of_the_week"
    assert notice.severity == Severity.WARNING
    assert notice.fields["serviceId"] == "service_1"
    assert notice.fields["csvRowNumber"] == 2
    service_ids = [n.fields["serviceId"] for n in notices]
    assert "service_2" not in service_ids


def test_service_has_active_day_no_notice() -> None:
    """Two services both with monday=1 produce no notices."""
    rows = [
        {"csv_row_number": 2, "service_id": "service_1", **_ALL_ZERO, "monday": 1},
        {"csv_row_number": 3, "service_id": "service_2", **_ALL_ZERO, "monday": 1},
    ]
    feed = {"calendar": make_calendar(rows)}
    assert validate_service_no_active_day(feed, CTX) == []


def test_all_days_active_no_notice() -> None:
    """Single row with all day columns = 1 produces no notice."""
    rows = [{"csv_row_number": 2, "service_id": "svc", **_ALL_ONE}]
    feed = {"calendar": make_calendar(rows)}
    assert validate_service_no_active_day(feed, CTX) == []


def test_multiple_inactive_services_multiple_notices() -> None:
    """Three all-zero rows each produce a notice."""
    rows = [
        {"csv_row_number": 2, "service_id": "service_a", **_ALL_ZERO},
        {"csv_row_number": 3, "service_id": "service_b", **_ALL_ZERO},
        {"csv_row_number": 4, "service_id": "service_c", **_ALL_ZERO},
    ]
    feed = {"calendar": make_calendar(rows)}
    notices = validate_service_no_active_day(feed, CTX)

    assert len(notices) == 3
    service_ids = {n.fields["serviceId"] for n in notices}
    assert service_ids == {"service_a", "service_b", "service_c"}


def test_mixed_active_and_inactive_only_inactive_flagged() -> None:
    """Only the all-zero service is flagged; the one with saturday=1 is not."""
    rows = [
        {"csv_row_number": 2, "service_id": "service_inactive", **_ALL_ZERO},
        {"csv_row_number": 3, "service_id": "service_active", **_ALL_ZERO, "saturday": 1},
    ]
    feed = {"calendar": make_calendar(rows)}
    notices = validate_service_no_active_day(feed, CTX)

    assert len(notices) == 1
    assert notices[0].fields["serviceId"] == "service_inactive"
    service_ids = [n.fields["serviceId"] for n in notices]
    assert "service_active" not in service_ids


def test_calendar_absent_no_notice() -> None:
    """Empty feed (no calendar key) produces no notices."""
    assert validate_service_no_active_day({}, CTX) == []


def test_empty_calendar_no_notice() -> None:
    """Empty calendar DataFrame produces no notices."""
    feed = {"calendar": make_calendar([])}
    assert validate_service_no_active_day(feed, CTX) == []


def test_null_service_id_row_skipped() -> None:
    """Row with null service_id and all-zero days is not flagged."""
    rows = [{"csv_row_number": 2, "service_id": None, **_ALL_ZERO}]
    feed = {"calendar": make_calendar(rows)}
    assert validate_service_no_active_day(feed, CTX) == []


def test_null_day_column_treated_as_zero() -> None:
    """Null in a day column is treated as 0 (NOT_AVAILABLE)."""
    row = {"csv_row_number": 2, "service_id": "svc", **_ALL_ZERO, "monday": None}
    feed = {"calendar": make_calendar([row])}
    notices = validate_service_no_active_day(feed, CTX)

    assert len(notices) == 1
    assert notices[0].fields["serviceId"] == "svc"


@pytest.mark.parametrize("active_day", [
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
])
def test_exactly_one_active_day_each_day_position(active_day: str) -> None:
    """A single active day column suppresses the notice regardless of position."""
    row = {"csv_row_number": 2, "service_id": "svc", **_ALL_ZERO, active_day: 1}
    feed = {"calendar": make_calendar([row])}
    assert validate_service_no_active_day(feed, CTX) == []


def test_csv_row_number_preserved_in_notice() -> None:
    """csv_row_number is read from the DataFrame, not recomputed."""
    rows = [
        {"csv_row_number": 5, "service_id": "svc_inactive", **_ALL_ZERO},
        {"csv_row_number": 2, "service_id": "svc_active", **_ALL_ZERO, "monday": 1},
    ]
    feed = {"calendar": make_calendar(rows)}
    notices = validate_service_no_active_day(feed, CTX)

    assert len(notices) == 1
    assert notices[0].fields["csvRowNumber"] == 5


def test_notice_code_and_severity() -> None:
    """Emitted notice has the correct code and WARNING severity."""
    rows = [{"csv_row_number": 2, "service_id": "svc", **_ALL_ZERO}]
    feed = {"calendar": make_calendar(rows)}
    notices = validate_service_no_active_day(feed, CTX)

    assert len(notices) == 1
    assert notices[0].code == "service_has_no_active_day_of_the_week"
    assert notices[0].severity == Severity.WARNING


def test_calendar_dates_ignored() -> None:
    """calendar_dates entries for a service do not suppress the notice."""
    calendar_rows = [{"csv_row_number": 2, "service_id": "service_1", **_ALL_ZERO}]
    calendar_dates_rows = [
        {"csv_row_number": 2, "service_id": "service_1", "date": "20240101", "exception_type": 1}
    ]
    feed = {
        "calendar": make_calendar(calendar_rows),
        "calendar_dates": pl.DataFrame(calendar_dates_rows),
    }
    notices = validate_service_no_active_day(feed, CTX)

    assert len(notices) == 1
    assert notices[0].fields["serviceId"] == "service_1"


def test_single_row_all_active_no_notice() -> None:
    """Single calendar row with all seven day columns = 1 produces no notice."""
    rows = [{"csv_row_number": 2, "service_id": "svc", **_ALL_ONE}]
    feed = {"calendar": make_calendar(rows)}
    assert validate_service_no_active_day(feed, CTX) == []


def test_single_row_all_inactive_one_notice() -> None:
    """Single calendar row with all seven day columns = 0 produces exactly one notice."""
    rows = [{"csv_row_number": 2, "service_id": "svc", **_ALL_ZERO}]
    feed = {"calendar": make_calendar(rows)}
    notices = validate_service_no_active_day(feed, CTX)

    assert len(notices) == 1
    assert notices[0].fields["serviceId"] == "svc"
