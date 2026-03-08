"""Tests for StopTimeArrivalAndDepartureTimeValidator."""

from datetime import date

import polars as pl
import pytest

from gtfs_validator.context import ValidationContext
from gtfs_validator.notices import Severity
from gtfs_validator.validators.stop_time_arrival_departure import (
    validate_stop_time_arrival_departure,
)

CTX = ValidationContext(country_code="US", date_for_validation=date(2024, 1, 1))


def make_feed(rows: list[dict]) -> dict[str, pl.DataFrame]:
    """Build a minimal feed dict from a list of stop_times row dicts."""
    return {"stop_times": pl.DataFrame(rows)}


# ---------------------------------------------------------------------------
# Test 1
# ---------------------------------------------------------------------------


def test_both_times_absent_emits_no_notice():
    """Both absent is valid; neither check fires."""
    feed = make_feed(
        [
            {
                "csv_row_number": 1,
                "trip_id": "first trip id",
                "stop_sequence": 2,
                "arrival_time": None,
                "departure_time": None,
            }
        ]
    )
    notices = validate_stop_time_arrival_departure(feed, CTX)
    assert notices == []


# ---------------------------------------------------------------------------
# Test 2
# ---------------------------------------------------------------------------


def test_arrival_before_previous_departure_emits_notice():
    """Second stop's arrival (420) is before first stop's departure (518) → error."""
    feed = make_feed(
        [
            {
                "csv_row_number": 1,
                "trip_id": "first trip id",
                "stop_sequence": 2,
                "arrival_time": 340,
                "departure_time": 518,
            },
            {
                "csv_row_number": 2,
                "trip_id": "first trip id",
                "stop_sequence": 3,
                "arrival_time": 420,
                "departure_time": 747,
            },
        ]
    )
    notices = validate_stop_time_arrival_departure(feed, CTX)
    assert len(notices) == 1
    n = notices[0]
    assert n.code == "stop_time_with_arrival_before_previous_departure_time"
    assert n.severity == Severity.ERROR
    assert n.fields["csv_row_number"] == 2
    assert n.fields["prev_csv_row_number"] == 1
    assert n.fields["trip_id"] == "first trip id"
    assert n.fields["arrival_time"] == 420
    assert n.fields["departure_time"] == 518


# ---------------------------------------------------------------------------
# Test 3
# ---------------------------------------------------------------------------


def test_arrival_after_previous_departure_emits_no_notice():
    """Second stop's arrival (518) > first stop's departure (420) → no error."""
    feed = make_feed(
        [
            {
                "csv_row_number": 1,
                "trip_id": "first trip id",
                "stop_sequence": 2,
                "arrival_time": 340,
                "departure_time": 420,
            },
            {
                "csv_row_number": 2,
                "trip_id": "first trip id",
                "stop_sequence": 3,
                "arrival_time": 518,
                "departure_time": 747,
            },
        ]
    )
    notices = validate_stop_time_arrival_departure(feed, CTX)
    assert notices == []


# ---------------------------------------------------------------------------
# Test 4
# ---------------------------------------------------------------------------


def test_missing_arrival_time_with_departure_present_emits_notice():
    """arrival_time absent, departure_time present → Check 1 fires with specified_field='departure_time'."""
    feed = make_feed(
        [
            {
                "csv_row_number": 1,
                "trip_id": "first trip id",
                "stop_sequence": 2,
                "arrival_time": None,
                "departure_time": 518,
            }
        ]
    )
    notices = validate_stop_time_arrival_departure(feed, CTX)
    assert len(notices) == 1
    n = notices[0]
    assert n.code == "stop_time_with_only_arrival_or_departure_time"
    assert n.severity == Severity.ERROR
    assert n.fields["csv_row_number"] == 1
    assert n.fields["trip_id"] == "first trip id"
    assert n.fields["stop_sequence"] == 2
    assert n.fields["specified_field"] == "departure_time"


# ---------------------------------------------------------------------------
# Test 5
# ---------------------------------------------------------------------------


def test_missing_departure_time_with_arrival_present_emits_notice():
    """departure_time absent, arrival_time present → Check 1 fires with specified_field='arrival_time'."""
    feed = make_feed(
        [
            {
                "csv_row_number": 1,
                "trip_id": "first trip id",
                "stop_sequence": 2,
                "arrival_time": 518,
                "departure_time": None,
            }
        ]
    )
    notices = validate_stop_time_arrival_departure(feed, CTX)
    assert len(notices) == 1
    n = notices[0]
    assert n.code == "stop_time_with_only_arrival_or_departure_time"
    assert n.severity == Severity.ERROR
    assert n.fields["csv_row_number"] == 1
    assert n.fields["trip_id"] == "first trip id"
    assert n.fields["stop_sequence"] == 2
    assert n.fields["specified_field"] == "arrival_time"


# ---------------------------------------------------------------------------
# Test 6
# ---------------------------------------------------------------------------


def test_stop_times_table_absent_emits_no_notice():
    """Missing stop_times key → guard returns []."""
    notices = validate_stop_time_arrival_departure({}, CTX)
    assert notices == []


# ---------------------------------------------------------------------------
# Test 7
# ---------------------------------------------------------------------------


def test_stop_times_table_empty_emits_no_notice():
    """Empty stop_times DataFrame → guard returns []."""
    feed = {
        "stop_times": pl.DataFrame(
            {
                "csv_row_number": pl.Series([], dtype=pl.Int64),
                "trip_id": pl.Series([], dtype=pl.Utf8),
                "stop_sequence": pl.Series([], dtype=pl.Int64),
                "arrival_time": pl.Series([], dtype=pl.Int64),
                "departure_time": pl.Series([], dtype=pl.Int64),
            }
        )
    }
    notices = validate_stop_time_arrival_departure(feed, CTX)
    assert notices == []


# ---------------------------------------------------------------------------
# Test 8
# ---------------------------------------------------------------------------


def test_both_times_present_emits_no_notice():
    """Both present, single stop → valid in both checks."""
    feed = make_feed(
        [
            {
                "csv_row_number": 1,
                "trip_id": "T1",
                "stop_sequence": 1,
                "arrival_time": 340,
                "departure_time": 518,
            }
        ]
    )
    notices = validate_stop_time_arrival_departure(feed, CTX)
    assert notices == []


# ---------------------------------------------------------------------------
# Test 9
# ---------------------------------------------------------------------------


def test_single_stop_trip_no_check2_notice():
    """Only one stop → no previous departure for Check 2."""
    feed = make_feed(
        [
            {
                "csv_row_number": 1,
                "trip_id": "T1",
                "stop_sequence": 1,
                "arrival_time": 100,
                "departure_time": 200,
            }
        ]
    )
    notices = validate_stop_time_arrival_departure(feed, CTX)
    assert notices == []


# ---------------------------------------------------------------------------
# Test 10
# ---------------------------------------------------------------------------


def test_arrival_equals_previous_departure_emits_no_notice():
    """Equality (200 == 200) is not strictly less-than → boundary must not fire."""
    feed = make_feed(
        [
            {
                "csv_row_number": 1,
                "trip_id": "T1",
                "stop_sequence": 1,
                "arrival_time": 100,
                "departure_time": 200,
            },
            {
                "csv_row_number": 2,
                "trip_id": "T1",
                "stop_sequence": 2,
                "arrival_time": 200,
                "departure_time": 300,
            },
        ]
    )
    notices = validate_stop_time_arrival_departure(feed, CTX)
    assert notices == []


# ---------------------------------------------------------------------------
# Test 11
# ---------------------------------------------------------------------------


def test_multiple_trips_checked_independently():
    """Two trips each with an ordering violation → 2 notices (one per trip)."""
    feed = make_feed(
        [
            # Trip T1
            {
                "csv_row_number": 1,
                "trip_id": "T1",
                "stop_sequence": 1,
                "arrival_time": 100,
                "departure_time": 500,
            },
            {
                "csv_row_number": 2,
                "trip_id": "T1",
                "stop_sequence": 2,
                "arrival_time": 200,
                "departure_time": 600,
            },
            # Trip T2
            {
                "csv_row_number": 3,
                "trip_id": "T2",
                "stop_sequence": 1,
                "arrival_time": 100,
                "departure_time": 500,
            },
            {
                "csv_row_number": 4,
                "trip_id": "T2",
                "stop_sequence": 2,
                "arrival_time": 200,
                "departure_time": 600,
            },
        ]
    )
    notices = validate_stop_time_arrival_departure(feed, CTX)
    check2_notices = [
        n
        for n in notices
        if n.code == "stop_time_with_arrival_before_previous_departure_time"
    ]
    assert len(check2_notices) == 2
    trip_ids = {n.fields["trip_id"] for n in check2_notices}
    assert trip_ids == {"T1", "T2"}


# ---------------------------------------------------------------------------
# Test 12
# ---------------------------------------------------------------------------


def test_stop_times_out_of_sequence_order_in_csv():
    """Rows in reverse stop_sequence order in the CSV; correct sort prevents spurious violation."""
    # If processed raw (seq 2 before seq 1), arrival[seq=2]=300 > dep[seq=1]=100
    # would look like: prev_dep from row with seq=2 (dep=600), then row with seq=1
    # arrival=100 < 600 → spurious notice. After correct sort by stop_sequence,
    # row seq=1 (dep=100) comes first, then seq=2 arrival=300 > 100 → no violation.
    feed = make_feed(
        [
            {
                "csv_row_number": 1,
                "trip_id": "T1",
                "stop_sequence": 2,
                "arrival_time": 300,
                "departure_time": 600,
            },
            {
                "csv_row_number": 2,
                "trip_id": "T1",
                "stop_sequence": 1,
                "arrival_time": 50,
                "departure_time": 100,
            },
        ]
    )
    notices = validate_stop_time_arrival_departure(feed, CTX)
    assert notices == []


# ---------------------------------------------------------------------------
# Test 13
# ---------------------------------------------------------------------------


def test_departure_absent_does_not_update_prev_dep_row():
    """Row 2 has no departure; prev_dep_row remains at Row 1 for Row 3's comparison."""
    feed = make_feed(
        [
            {
                "csv_row_number": 1,
                "trip_id": "T1",
                "stop_sequence": 1,
                "arrival_time": 100,
                "departure_time": 200,
            },
            {
                "csv_row_number": 2,
                "trip_id": "T1",
                "stop_sequence": 2,
                "arrival_time": 300,
                "departure_time": None,
            },
            {
                "csv_row_number": 3,
                "trip_id": "T1",
                "stop_sequence": 3,
                "arrival_time": 150,
                "departure_time": 400,
            },
        ]
    )
    notices = validate_stop_time_arrival_departure(feed, CTX)

    check1 = [
        n
        for n in notices
        if n.code == "stop_time_with_only_arrival_or_departure_time"
    ]
    check2 = [
        n
        for n in notices
        if n.code == "stop_time_with_arrival_before_previous_departure_time"
    ]

    assert len(check1) == 1
    assert check1[0].fields["csv_row_number"] == 2
    assert check1[0].fields["specified_field"] == "arrival_time"

    assert len(check2) == 1
    assert check2[0].fields["csv_row_number"] == 3
    assert check2[0].fields["prev_csv_row_number"] == 1
    assert check2[0].fields["departure_time"] == 200


# ---------------------------------------------------------------------------
# Test 14
# ---------------------------------------------------------------------------


def test_row_triggers_both_checks_simultaneously():
    """Row 2 has only arrival_time (Check 1) and arrival < prev departure (Check 2)."""
    feed = make_feed(
        [
            {
                "csv_row_number": 1,
                "trip_id": "T1",
                "stop_sequence": 1,
                "arrival_time": 100,
                "departure_time": 300,
            },
            {
                "csv_row_number": 2,
                "trip_id": "T1",
                "stop_sequence": 2,
                "arrival_time": 50,
                "departure_time": None,
            },
        ]
    )
    notices = validate_stop_time_arrival_departure(feed, CTX)
    assert len(notices) == 2

    codes = {n.code for n in notices}
    assert "stop_time_with_only_arrival_or_departure_time" in codes
    assert "stop_time_with_arrival_before_previous_departure_time" in codes

    for n in notices:
        assert n.fields["csv_row_number"] == 2


# ---------------------------------------------------------------------------
# Test 15
# ---------------------------------------------------------------------------


def test_multiple_offending_rows_each_emit_one_notice():
    """Rows 2 and 3 each have arrival < most recent departure → 2 Check 2 notices."""
    feed = make_feed(
        [
            {
                "csv_row_number": 1,
                "trip_id": "T1",
                "stop_sequence": 1,
                "arrival_time": 100,
                "departure_time": 500,
            },
            {
                "csv_row_number": 2,
                "trip_id": "T1",
                "stop_sequence": 2,
                "arrival_time": 200,
                "departure_time": 600,
            },
            {
                "csv_row_number": 3,
                "trip_id": "T1",
                "stop_sequence": 3,
                "arrival_time": 300,
                "departure_time": 700,
            },
        ]
    )
    notices = validate_stop_time_arrival_departure(feed, CTX)
    check2 = [
        n
        for n in notices
        if n.code == "stop_time_with_arrival_before_previous_departure_time"
    ]
    assert len(check2) == 2


# ---------------------------------------------------------------------------
# Test 16
# ---------------------------------------------------------------------------


def test_notice_fields_are_complete_check1():
    """Check 1 notice contains exactly the expected field keys."""
    feed = make_feed(
        [
            {
                "csv_row_number": 1,
                "trip_id": "first trip id",
                "stop_sequence": 2,
                "arrival_time": None,
                "departure_time": 518,
            }
        ]
    )
    notices = validate_stop_time_arrival_departure(feed, CTX)
    assert len(notices) == 1
    assert set(notices[0].fields.keys()) == {
        "csv_row_number",
        "trip_id",
        "stop_sequence",
        "specified_field",
    }


# ---------------------------------------------------------------------------
# Test 17
# ---------------------------------------------------------------------------


def test_notice_fields_are_complete_check2():
    """Check 2 notice contains exactly the expected field keys."""
    feed = make_feed(
        [
            {
                "csv_row_number": 1,
                "trip_id": "T1",
                "stop_sequence": 1,
                "arrival_time": 100,
                "departure_time": 500,
            },
            {
                "csv_row_number": 2,
                "trip_id": "T1",
                "stop_sequence": 2,
                "arrival_time": 200,
                "departure_time": 600,
            },
        ]
    )
    notices = validate_stop_time_arrival_departure(feed, CTX)
    check2 = [
        n
        for n in notices
        if n.code == "stop_time_with_arrival_before_previous_departure_time"
    ]
    assert len(check2) == 1
    assert set(check2[0].fields.keys()) == {
        "csv_row_number",
        "prev_csv_row_number",
        "trip_id",
        "arrival_time",
        "departure_time",
    }
