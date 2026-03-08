"""Tests for validate_missing_stops_file."""

from datetime import date

import polars as pl
import pytest

from gtfs_validator.context import ValidationContext
from gtfs_validator.notices import Severity
from gtfs_validator.validators.missing_stops_file import validate_missing_stops_file

CTX = ValidationContext(country_code="US", date_for_validation=date(2026, 3, 8))


def make_stops() -> pl.DataFrame:
    """Minimal stops DataFrame with one row."""
    return pl.DataFrame(
        {"stop_id": ["s1"]},
        schema={"stop_id": pl.Utf8},
    )


def make_empty_stops() -> pl.DataFrame:
    """stops.txt present but zero data rows (headers only)."""
    return pl.DataFrame(schema={"stop_id": pl.Utf8})


def make_geojson() -> list[dict]:
    """Minimal locations_geojson value — non-empty list."""
    return [{"type": "Feature", "geometry": None, "properties": {"stop_id": "loc1"}}]


def make_empty_geojson() -> list[dict]:
    """locations_geojson present but zero features."""
    return []


def test_both_files_absent_yields_notice() -> None:
    feed: dict = {}
    notices = validate_missing_stops_file(feed, CTX)
    assert len(notices) == 1
    assert notices[0].code == "missing_required_file"
    assert notices[0].severity == Severity.ERROR
    assert notices[0].fields == {"filename": "stops.txt"}


def test_geojson_present_stops_absent_no_notice() -> None:
    feed = {"locations_geojson": make_empty_geojson()}
    notices = validate_missing_stops_file(feed, CTX)
    assert notices == []


def test_stops_present_geojson_absent_no_notice() -> None:
    feed = {"stops": make_stops()}
    notices = validate_missing_stops_file(feed, CTX)
    assert notices == []


def test_both_files_present_no_notice() -> None:
    feed = {"stops": make_stops(), "locations_geojson": make_geojson()}
    notices = validate_missing_stops_file(feed, CTX)
    assert notices == []


def test_empty_stops_df_counts_as_present() -> None:
    feed = {"stops": make_empty_stops()}
    notices = validate_missing_stops_file(feed, CTX)
    assert notices == []


def test_empty_geojson_list_counts_as_present() -> None:
    feed = {"locations_geojson": make_empty_geojson()}
    notices = validate_missing_stops_file(feed, CTX)
    assert notices == []


def test_notice_filename_is_always_stops_txt() -> None:
    feed: dict = {}
    notices = validate_missing_stops_file(feed, CTX)
    assert len(notices) == 1
    assert notices[0].fields["filename"] == "stops.txt"


def test_ctx_unused_does_not_affect_result() -> None:
    feed: dict = {}
    ctx_a = ValidationContext(country_code="US", date_for_validation=date(2026, 1, 1))
    ctx_b = ValidationContext(country_code="DE", date_for_validation=date(2099, 12, 31))
    notices_a = validate_missing_stops_file(feed, ctx_a)
    notices_b = validate_missing_stops_file(feed, ctx_b)
    assert len(notices_a) == 1
    assert len(notices_b) == 1
    assert notices_a[0].code == notices_b[0].code
    assert notices_a[0].fields == notices_b[0].fields


def test_other_tables_present_do_not_affect_result() -> None:
    feed = {"agency": pl.DataFrame({"agency_id": ["A1"]})}
    notices = validate_missing_stops_file(feed, CTX)
    assert len(notices) == 1
    assert notices[0].code == "missing_required_file"
