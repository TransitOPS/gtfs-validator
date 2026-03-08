"""Tests for validate_timepoint_time."""

from __future__ import annotations

import polars as pl
import pytest

from gtfs_validator.notices import Severity
from gtfs_validator.validators.timepoint_time import validate_timepoint_time


def _make_ctx():
    from datetime import date

    from gtfs_validator.context import ValidationContext

    return ValidationContext(country_code="US", date_for_validation=date(2024, 1, 1))


def _make_st(
    *,
    trip_id: str = "T1",
    stop_sequence: int = 1,
    arrival_time: str | None = None,
    departure_time: str | None = None,
    timepoint: int | None = None,
    csv_row_number: int = 1,
    include_timepoint_col: bool = True,
) -> pl.DataFrame:
    data: dict[str, list] = {
        "trip_id": [trip_id],
        "stop_sequence": [stop_sequence],
        "arrival_time": [arrival_time],
        "departure_time": [departure_time],
        "csv_row_number": [csv_row_number],
    }
    if include_timepoint_col:
        data["timepoint"] = [timepoint]
    return pl.DataFrame(data)


def _feed(st: pl.DataFrame) -> dict:
    return {"stop_times": st}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_exact_timepoint_both_times_missing_emits_two_errors():
    """timepoint=1, both times absent → two stop_time_timepoint_without_times ERRORs."""
    ctx = _make_ctx()
    st = _make_st(timepoint=1, arrival_time=None, departure_time=None)
    notices = validate_timepoint_time(_feed(st), ctx)

    assert len(notices) == 2
    for n in notices:
        assert n.code == "stop_time_timepoint_without_times"
        assert n.severity == Severity.ERROR

    fields_list = [n.fields["specified_field"] for n in notices]
    assert "arrival_time" in fields_list
    assert "departure_time" in fields_list


def test_exact_timepoint_both_times_present_no_notices():
    """timepoint=1, both times present → no notices."""
    ctx = _make_ctx()
    st = _make_st(timepoint=1, arrival_time="00:07:30", departure_time="00:09:40")
    assert validate_timepoint_time(_feed(st), ctx) == []


def test_exact_timepoint_missing_departure_time_emits_one_error():
    """timepoint=1, departure_time absent → one ERROR for departure_time."""
    ctx = _make_ctx()
    st = _make_st(timepoint=1, arrival_time="00:07:30", departure_time=None)
    notices = validate_timepoint_time(_feed(st), ctx)

    assert len(notices) == 1
    assert notices[0].code == "stop_time_timepoint_without_times"
    assert notices[0].severity == Severity.ERROR
    assert notices[0].fields["specified_field"] == "departure_time"


def test_exact_timepoint_missing_arrival_time_emits_one_error():
    """timepoint=1, arrival_time absent → one ERROR for arrival_time."""
    ctx = _make_ctx()
    st = _make_st(timepoint=1, arrival_time=None, departure_time="00:07:30")
    notices = validate_timepoint_time(_feed(st), ctx)

    assert len(notices) == 1
    assert notices[0].code == "stop_time_timepoint_without_times"
    assert notices[0].severity == Severity.ERROR
    assert notices[0].fields["specified_field"] == "arrival_time"


def test_approximate_timepoint_no_times_no_notices():
    """timepoint=0, no times → no notices."""
    ctx = _make_ctx()
    st = _make_st(timepoint=0, arrival_time=None, departure_time=None)
    assert validate_timepoint_time(_feed(st), ctx) == []


def test_approximate_timepoint_both_times_present_no_notices():
    """timepoint=0, both times present → no notices."""
    ctx = _make_ctx()
    st = _make_st(timepoint=0, arrival_time="00:07:30", departure_time="00:09:40")
    assert validate_timepoint_time(_feed(st), ctx) == []


def test_null_timepoint_no_times_no_notices():
    """timepoint=None, no times → no notices (Check 1 requires at least one time)."""
    ctx = _make_ctx()
    st = _make_st(timepoint=None, arrival_time=None, departure_time=None)
    assert validate_timepoint_time(_feed(st), ctx) == []


def test_null_timepoint_both_times_present_emits_warning():
    """timepoint=None (column present), both times present → one missing_timepoint_value WARNING."""
    ctx = _make_ctx()
    st = _make_st(timepoint=None, arrival_time="00:07:30", departure_time="00:09:40")
    notices = validate_timepoint_time(_feed(st), ctx)

    assert len(notices) == 1
    assert notices[0].code == "missing_timepoint_value"
    assert notices[0].severity == Severity.WARNING
    assert "csv_row_number" in notices[0].fields
    assert "trip_id" in notices[0].fields
    assert "stop_sequence" in notices[0].fields


def test_null_timepoint_one_time_present_emits_warning():
    """timepoint=None, only arrival_time present → one missing_timepoint_value WARNING."""
    ctx = _make_ctx()
    st = _make_st(timepoint=None, arrival_time="00:07:30", departure_time=None)
    notices = validate_timepoint_time(_feed(st), ctx)

    assert len(notices) == 1
    assert notices[0].code == "missing_timepoint_value"
    assert notices[0].severity == Severity.WARNING


def test_timepoint_column_absent_no_notices():
    """No timepoint column in DataFrame → no notices (header-level guard)."""
    ctx = _make_ctx()
    st = _make_st(arrival_time="00:07:30", departure_time="00:09:40", include_timepoint_col=False)
    assert validate_timepoint_time(_feed(st), ctx) == []


def test_stop_times_absent_no_notices():
    """No stop_times key in feed → no notices."""
    ctx = _make_ctx()
    assert validate_timepoint_time({}, ctx) == []


def test_stop_times_empty_no_notices():
    """Empty stop_times DataFrame → no notices."""
    ctx = _make_ctx()
    st = pl.DataFrame(
        {
            "trip_id": [],
            "stop_sequence": [],
            "arrival_time": [],
            "departure_time": [],
            "timepoint": [],
            "csv_row_number": [],
        }
    )
    assert validate_timepoint_time({"stop_times": st}, ctx) == []


def test_multiple_rows_mixed_violations():
    """Three rows with mixed violation patterns produce correct notices in order."""
    ctx = _make_ctx()
    st = pl.DataFrame(
        {
            "trip_id": ["T1", "T2", "T3"],
            "stop_sequence": [1, 2, 3],
            "arrival_time": ["00:05:00", None, None],
            "departure_time": ["00:05:30", None, None],
            "timepoint": [None, 1, 0],
            "csv_row_number": [1, 2, 3],
        }
    )
    notices = validate_timepoint_time({"stop_times": st}, ctx)

    assert len(notices) == 3

    # First: Check 1 WARNING for row 1
    assert notices[0].code == "missing_timepoint_value"
    assert notices[0].severity == Severity.WARNING
    assert notices[0].fields["csv_row_number"] == 1

    # Second: Check 2a ERROR for row 2, arrival_time
    assert notices[1].code == "stop_time_timepoint_without_times"
    assert notices[1].severity == Severity.ERROR
    assert notices[1].fields["csv_row_number"] == 2
    assert notices[1].fields["specified_field"] == "arrival_time"

    # Third: Check 2b ERROR for row 2, departure_time
    assert notices[2].code == "stop_time_timepoint_without_times"
    assert notices[2].severity == Severity.ERROR
    assert notices[2].fields["csv_row_number"] == 2
    assert notices[2].fields["specified_field"] == "departure_time"


def test_notice_fields_complete():
    """All four fields on stop_time_timepoint_without_times are present and correct."""
    ctx = _make_ctx()
    st = _make_st(
        trip_id="TRIP_A",
        stop_sequence=5,
        csv_row_number=42,
        timepoint=1,
        arrival_time=None,
        departure_time="00:10:00",
    )
    notices = validate_timepoint_time(_feed(st), ctx)

    assert len(notices) == 1
    f = notices[0].fields
    assert f["csv_row_number"] == 42
    assert f["trip_id"] == "TRIP_A"
    assert f["stop_sequence"] == 5
    assert f["specified_field"] == "arrival_time"
