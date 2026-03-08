"""Tests for LocationHasStopTimesValidator."""

from datetime import date

import polars as pl
import pytest

from gtfs_validator.context import ValidationContext
from gtfs_validator.notices import Severity
from gtfs_validator.validators.location_has_stop_times import (
    validate_location_has_stop_times,
)

CTX = ValidationContext(country_code="US", date_for_validation=date(2026, 3, 8))


def make_stops(
    rows: list[tuple[str, str | None, int | None]], row_start: int = 2
) -> pl.DataFrame:
    """Each row is (stop_id, stop_name, location_type)."""
    return pl.DataFrame(
        {
            "csvRowNumber": list(range(row_start, row_start + len(rows))),
            "stop_id": [r[0] for r in rows],
            "stop_name": [r[1] for r in rows],
            "location_type": [r[2] for r in rows],
        },
        schema={
            "csvRowNumber": pl.Int64,
            "stop_id": pl.Utf8,
            "stop_name": pl.Utf8,
            "location_type": pl.Int64,
        },
    )


def make_stop_times(
    rows: list[tuple[str, str | None]],
    row_start: int = 2,
    include_location_group_id: bool = False,
) -> pl.DataFrame:
    """Each row is (stop_id, location_group_id).

    If include_location_group_id is False, the location_group_id column is omitted.
    """
    data: dict = {
        "csvRowNumber": list(range(row_start, row_start + len(rows))),
        "stop_id": [r[0] for r in rows],
    }
    if include_location_group_id:
        data["location_group_id"] = [r[1] for r in rows]
    return pl.DataFrame(data)


def make_location_group_stops(rows: list[tuple[str, str]]) -> pl.DataFrame:
    """Each row is (location_group_id, stop_id)."""
    return pl.DataFrame(
        {
            "location_group_id": [r[0] for r in rows],
            "stop_id": [r[1] for r in rows],
        }
    )


# --- Test 1: stop with stop_time, no notice ---
def test_stop_with_stop_time_no_notice():
    feed = {
        "stops": make_stops([("location1", "Stop 1", 0)]),
        "stop_times": make_stop_times([("location1", None)]),
        "location_group_stops": make_location_group_stops([("lg1", "other")]),
    }
    notices = validate_location_has_stop_times(feed, CTX)
    assert len(notices) == 0


# --- Test 2: stop without stop_time yields warning ---
def test_stop_without_stop_time_yields_notice():
    feed = {
        "stops": make_stops([("stopId", "My Stop", 0)]),
        "stop_times": make_stop_times([]),
    }
    notices = validate_location_has_stop_times(feed, CTX)
    assert len(notices) == 1
    n = notices[0]
    assert n.code == "stop_without_stop_time"
    assert n.severity == Severity.WARNING
    assert n.fields["stop_id"] == "stopId"
    assert n.fields["stop_name"] == "My Stop"


# --- Test 3: station with stop_time yields error ---
def test_station_with_stop_time_yields_notice():
    feed = {
        "stops": make_stops([("location1", "Station 1", 1)]),
        "stop_times": make_stop_times([("location1", None)]),
    }
    notices = validate_location_has_stop_times(feed, CTX)
    assert len(notices) == 1
    n = notices[0]
    assert n.code == "location_with_unexpected_stop_time"
    assert n.severity == Severity.ERROR
    assert n.fields["stop_id"] == "location1"
    assert isinstance(n.fields["stop_time_csv_row_number"], int)


# --- Test 4: station without stop_time, no notice ---
def test_station_without_stop_time_no_notice():
    feed = {
        "stops": make_stops([("location1", "Station 1", 1)]),
        "stop_times": make_stop_times([]),
    }
    notices = validate_location_has_stop_times(feed, CTX)
    assert len(notices) == 0


# --- Test 5: entrance with stop_time yields error ---
def test_entrance_with_stop_time_yields_notice():
    feed = {
        "stops": make_stops([("location1", "Entrance 1", 2)]),
        "stop_times": make_stop_times([("location1", None)]),
    }
    notices = validate_location_has_stop_times(feed, CTX)
    assert len(notices) == 1
    assert notices[0].code == "location_with_unexpected_stop_time"


# --- Test 6: entrance without stop_time, no notice ---
def test_entrance_without_stop_time_no_notice():
    feed = {
        "stops": make_stops([("location1", "Entrance 1", 2)]),
        "stop_times": make_stop_times([]),
    }
    notices = validate_location_has_stop_times(feed, CTX)
    assert len(notices) == 0


# --- Test 7: stops missing, no notices ---
def test_stops_missing_no_notices():
    feed = {
        "stop_times": make_stop_times([("s1", None)]),
    }
    notices = validate_location_has_stop_times(feed, CTX)
    assert len(notices) == 0


# --- Test 8: stop_times missing, no notices ---
def test_stop_times_missing_no_notices():
    feed = {
        "stops": make_stops([("s1", "Stop 1", 0)]),
    }
    notices = validate_location_has_stop_times(feed, CTX)
    assert len(notices) == 0


# --- Test 9: location_group_stops missing, warning emitted ---
def test_location_group_stops_missing_no_notices():
    feed = {
        "stops": make_stops([("s1", "Stop 1", 0)]),
        "stop_times": make_stop_times([]),
    }
    notices = validate_location_has_stop_times(feed, CTX)
    assert len(notices) == 1
    assert notices[0].code == "stop_without_stop_time"


# --- Test 10: stop referenced only via location group, no warning ---
def test_stop_referenced_only_via_location_group_no_warning():
    feed = {
        "stops": make_stops([("s1", "Stop 1", 0)]),
        "stop_times": make_stop_times(
            [("other", "lg1")], include_location_group_id=True
        ),
        "location_group_stops": make_location_group_stops([("lg1", "s1")]),
    }
    notices = validate_location_has_stop_times(feed, CTX)
    assert len(notices) == 0


# --- Test 11: stop with both direct and group references, no warning ---
def test_stop_with_both_direct_and_group_references_no_warning():
    feed = {
        "stops": make_stops([("s1", "Stop 1", 0)]),
        "stop_times": make_stop_times(
            [("s1", None), ("other", "lg1")], include_location_group_id=True
        ),
        "location_group_stops": make_location_group_stops([("lg1", "s1")]),
    }
    notices = validate_location_has_stop_times(feed, CTX)
    assert len(notices) == 0


# --- Test 12: generic node with stop_time yields error ---
def test_generic_node_with_stop_time_yields_error():
    feed = {
        "stops": make_stops([("n1", "Node 1", 3)]),
        "stop_times": make_stop_times([("n1", None)]),
    }
    notices = validate_location_has_stop_times(feed, CTX)
    assert len(notices) == 1
    assert notices[0].code == "location_with_unexpected_stop_time"


# --- Test 13: boarding area with stop_time yields error ---
def test_boarding_area_with_stop_time_yields_error():
    feed = {
        "stops": make_stops([("b1", "Board 1", 4)]),
        "stop_times": make_stop_times([("b1", None)]),
    }
    notices = validate_location_has_stop_times(feed, CTX)
    assert len(notices) == 1
    assert notices[0].code == "location_with_unexpected_stop_time"


# --- Test 14: non-stop location referenced only via location group, no error ---
def test_non_stop_location_referenced_only_via_location_group_no_error():
    feed = {
        "stops": make_stops([("s1", "Station 1", 1)]),
        "stop_times": make_stop_times(
            [("other", "lg1")], include_location_group_id=True
        ),
        "location_group_stops": make_location_group_stops([("lg1", "s1")]),
    }
    notices = validate_location_has_stop_times(feed, CTX)
    assert len(notices) == 0


# --- Test 15: multiple stop_times for non-stop location, one error ---
def test_multiple_stop_times_for_non_stop_location_one_error():
    feed = {
        "stops": make_stops([("s1", "Station 1", 1)]),
        "stop_times": make_stop_times(
            [("s1", None), ("s1", None), ("s1", None)], row_start=2
        ),
    }
    notices = validate_location_has_stop_times(feed, CTX)
    assert len(notices) == 1
    assert notices[0].fields["stop_time_csv_row_number"] == 2


# --- Test 16: mixed stops, correct notices ---
def test_mixed_stops_correct_notices():
    feed = {
        "stops": make_stops([
            ("s1", "Stop 1", 0),
            ("s2", "Stop 2", 0),
            ("s3", "Station", 1),
            ("s4", "Entrance", 2),
        ]),
        "stop_times": make_stop_times([("s1", None), ("s3", None)]),
    }
    notices = validate_location_has_stop_times(feed, CTX)
    warnings = [n for n in notices if n.severity == Severity.WARNING]
    errors = [n for n in notices if n.severity == Severity.ERROR]
    assert len(warnings) == 1
    assert warnings[0].fields["stop_id"] == "s2"
    assert len(errors) == 1
    assert errors[0].fields["stop_id"] == "s3"


# --- Test 17: empty stops, no notices ---
def test_empty_stops_no_notices():
    feed = {
        "stops": make_stops([]),
        "stop_times": make_stop_times([("s1", None)]),
    }
    notices = validate_location_has_stop_times(feed, CTX)
    assert len(notices) == 0


# --- Test 18: empty stop_times, all stops warned ---
def test_empty_stop_times_all_stops_warned():
    feed = {
        "stops": make_stops([("s1", "Stop 1", 0), ("s2", "Stop 2", 0)]),
        "stop_times": make_stop_times([]),
    }
    notices = validate_location_has_stop_times(feed, CTX)
    assert len(notices) == 2
    assert all(n.code == "stop_without_stop_time" for n in notices)


# --- Test 19: null location_type treated as stop ---
def test_null_location_type_treated_as_stop():
    feed = {
        "stops": make_stops([("s1", "Stop 1", None)]),
        "stop_times": make_stop_times([]),
    }
    notices = validate_location_has_stop_times(feed, CTX)
    assert len(notices) == 1
    assert notices[0].code == "stop_without_stop_time"


# --- Test 20: location_group_id column missing in stop_times ---
def test_location_group_id_column_missing_in_stop_times():
    feed = {
        "stops": make_stops([("s1", "Stop 1", 0)]),
        "stop_times": make_stop_times(
            [("other", None)], include_location_group_id=False
        ),
        "location_group_stops": make_location_group_stops([("lg1", "s1")]),
    }
    notices = validate_location_has_stop_times(feed, CTX)
    assert len(notices) == 1
    assert notices[0].code == "stop_without_stop_time"
