"""Tests for StopTimesShapeDistTraveledPresenceValidator."""

from datetime import date

import polars as pl
import pytest

from gtfs_validator.context import ValidationContext
from gtfs_validator.notices import Severity
from gtfs_validator.validators.stop_times_shape_dist_traveled_presence import (
    validate_stop_times_shape_dist_traveled_presence,
)

CTX = ValidationContext(country_code="US", date_for_validation=date(2024, 1, 1))


def _make_st(**kwargs) -> pl.DataFrame:
    """Helper to build a single-row stop_times DataFrame from keyword args."""
    return pl.DataFrame({k: [v] for k, v in kwargs.items()})


def test_location_with_shape_distance_should_generate_notice() -> None:
    """Mirrors Java test locationWithShapeDistanceShouldGenerateNotice."""
    st = pl.DataFrame({
        "csv_row_number": [1, 2, 3],
        "trip_id": ["first trip", "first trip", "first trip"],
        "location_group_id": ["loc1", None, None],
        "location_id": [None, "loc2", None],
        "stop_id": [None, None, "stop1"],
        "stop_sequence": [2, 42, 46],
        "shape_dist_traveled": [10.0, 45.0, 64.0],
    })
    feed = {"stop_times": st}
    notices = validate_stop_times_shape_dist_traveled_presence(feed, CTX)

    assert len(notices) == 2

    n0 = notices[0]
    assert n0.code == "forbidden_shape_dist_traveled"
    assert n0.severity == Severity.ERROR
    assert n0.fields["csv_row_number"] == 1
    assert n0.fields["trip_id"] == "first trip"
    assert n0.fields["location_group_id"] == "loc1"
    assert n0.fields["location_id"] is None
    assert n0.fields["shape_dist_traveled"] == 10.0

    n1 = notices[1]
    assert n1.code == "forbidden_shape_dist_traveled"
    assert n1.severity == Severity.ERROR
    assert n1.fields["csv_row_number"] == 2
    assert n1.fields["trip_id"] == "first trip"
    assert n1.fields["location_group_id"] is None
    assert n1.fields["location_id"] == "loc2"
    assert n1.fields["shape_dist_traveled"] == 45.0


def test_stop_times_absent_returns_empty() -> None:
    """Guard 1: stop_times key missing from feed."""
    assert validate_stop_times_shape_dist_traveled_presence({}, CTX) == []


def test_shape_dist_traveled_column_absent_returns_empty() -> None:
    """Guard 2: shape_dist_traveled column absent."""
    st = pl.DataFrame({
        "csv_row_number": [1],
        "trip_id": ["t1"],
        "location_id": ["loc1"],
        "location_group_id": [None],
        "stop_id": [None],
    })
    assert validate_stop_times_shape_dist_traveled_presence({"stop_times": st}, CTX) == []


def test_no_flex_location_columns_present_returns_empty() -> None:
    """Guard 3: neither location_id nor location_group_id present."""
    st = pl.DataFrame({
        "csv_row_number": [1],
        "trip_id": ["t1"],
        "stop_id": [None],
        "shape_dist_traveled": [10.0],
    })
    assert validate_stop_times_shape_dist_traveled_presence({"stop_times": st}, CTX) == []


def test_empty_stop_times_returns_empty() -> None:
    """Guard 4: empty DataFrame."""
    st = pl.DataFrame({
        "csv_row_number": pl.Series([], dtype=pl.Int64),
        "trip_id": pl.Series([], dtype=pl.Utf8),
        "stop_id": pl.Series([], dtype=pl.Utf8),
        "location_id": pl.Series([], dtype=pl.Utf8),
        "location_group_id": pl.Series([], dtype=pl.Utf8),
        "shape_dist_traveled": pl.Series([], dtype=pl.Float64),
    })
    assert validate_stop_times_shape_dist_traveled_presence({"stop_times": st}, CTX) == []


def test_only_location_id_column_present_no_location_group_col() -> None:
    """location_group_id column absent entirely; location_id present."""
    st = pl.DataFrame({
        "csv_row_number": [1],
        "trip_id": ["t1"],
        "stop_id": [None],
        "location_id": ["loc1"],
        "shape_dist_traveled": [5.0],
    })
    notices = validate_stop_times_shape_dist_traveled_presence({"stop_times": st}, CTX)
    assert len(notices) == 1
    assert notices[0].fields["location_group_id"] is None
    assert notices[0].fields["location_id"] == "loc1"


def test_only_location_group_col_present_no_location_id_col() -> None:
    """location_id column absent entirely; location_group_id present."""
    st = pl.DataFrame({
        "csv_row_number": [1],
        "trip_id": ["t1"],
        "stop_id": [None],
        "location_group_id": ["grp1"],
        "shape_dist_traveled": [5.0],
    })
    notices = validate_stop_times_shape_dist_traveled_presence({"stop_times": st}, CTX)
    assert len(notices) == 1
    assert notices[0].fields["location_id"] is None
    assert notices[0].fields["location_group_id"] == "grp1"


def test_stop_id_set_with_shape_dist_traveled_no_notice() -> None:
    """Row with non-null stop_id must not generate a notice."""
    st = pl.DataFrame({
        "csv_row_number": [1],
        "trip_id": ["t1"],
        "stop_id": ["stop1"],
        "location_group_id": [None],
        "location_id": [None],
        "shape_dist_traveled": [10.0],
    })
    assert validate_stop_times_shape_dist_traveled_presence({"stop_times": st}, CTX) == []


def test_no_stop_id_no_flex_location_with_shape_dist_no_notice() -> None:
    """No flex location field set — conditions not met, no notice."""
    st = pl.DataFrame({
        "csv_row_number": [1],
        "trip_id": ["t1"],
        "stop_id": [None],
        "location_group_id": [None],
        "location_id": [None],
        "shape_dist_traveled": [20.0],
    })
    assert validate_stop_times_shape_dist_traveled_presence({"stop_times": st}, CTX) == []


def test_flex_location_with_null_shape_dist_no_notice() -> None:
    """Flex location set but shape_dist_traveled is null — no notice."""
    st = pl.DataFrame({
        "csv_row_number": [1],
        "trip_id": ["t1"],
        "stop_id": [None],
        "location_id": ["loc1"],
        "location_group_id": [None],
        "shape_dist_traveled": [None],
    })
    assert validate_stop_times_shape_dist_traveled_presence({"stop_times": st}, CTX) == []


def test_both_flex_fields_set_with_shape_dist_emits_one_notice() -> None:
    """Both flex fields set: exactly 1 notice with both fields populated."""
    st = pl.DataFrame({
        "csv_row_number": [1],
        "trip_id": ["t1"],
        "stop_id": [None],
        "location_group_id": ["grp1"],
        "location_id": ["loc1"],
        "shape_dist_traveled": [7.5],
    })
    notices = validate_stop_times_shape_dist_traveled_presence({"stop_times": st}, CTX)
    assert len(notices) == 1
    assert notices[0].fields["location_group_id"] == "grp1"
    assert notices[0].fields["location_id"] == "loc1"


def test_multiple_offending_rows_across_trips() -> None:
    """Three violating rows across two different trips produce 3 notices."""
    st = pl.DataFrame({
        "csv_row_number": [1, 2, 3],
        "trip_id": ["trip_a", "trip_a", "trip_b"],
        "stop_id": [None, None, None],
        "location_group_id": ["grp1", None, "grp2"],
        "location_id": [None, "loc1", None],
        "shape_dist_traveled": [1.0, 2.0, 3.0],
    })
    notices = validate_stop_times_shape_dist_traveled_presence({"stop_times": st}, CTX)
    assert len(notices) == 3


def test_mixed_rows_only_violating_rows_produce_notices() -> None:
    """Four rows; only two are violations."""
    st = pl.DataFrame({
        "csv_row_number": [1, 2, 3, 4],
        "trip_id": ["t1", "t1", "t1", "t1"],
        "stop_id": [None, None, "stop1", None],
        "location_group_id": ["grp1", None, None, "grp2"],
        "location_id": [None, "loc1", None, None],
        # row 1: violation (flex + shape_dist_traveled)
        # row 2: violation (flex + shape_dist_traveled)
        # row 3: no violation (stop_id set)
        # row 4: no violation (shape_dist_traveled null)
        "shape_dist_traveled": [5.0, 6.0, 7.0, None],
    })
    notices = validate_stop_times_shape_dist_traveled_presence({"stop_times": st}, CTX)
    assert len(notices) == 2
    row_numbers = {n.fields["csv_row_number"] for n in notices}
    assert row_numbers == {1, 2}


def test_notice_fields_are_complete() -> None:
    """Notice fields dict contains exactly the expected keys."""
    st = pl.DataFrame({
        "csv_row_number": [1],
        "trip_id": ["t1"],
        "stop_id": [None],
        "location_group_id": ["grp1"],
        "location_id": [None],
        "shape_dist_traveled": [3.14],
    })
    notices = validate_stop_times_shape_dist_traveled_presence({"stop_times": st}, CTX)
    assert len(notices) == 1
    expected_keys = {"csv_row_number", "trip_id", "location_group_id", "location_id", "shape_dist_traveled"}
    assert set(notices[0].fields.keys()) == expected_keys
