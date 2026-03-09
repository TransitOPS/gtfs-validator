"""Tests for validate_url_consistency."""

from __future__ import annotations

from datetime import date

import polars as pl

from gtfs_validator.context import ValidationContext
from gtfs_validator.notices import Severity
from gtfs_validator.validators.url_consistency import validate_url_consistency

CTX = ValidationContext(country_code="US", date_for_validation=date(2024, 1, 1))


def test_different_route_and_agency_url_no_notice() -> None:
    """Different URLs for route and agency -> no notice."""
    feed = {
        "agency": pl.DataFrame({
            "agency_name": ["Agency A", "Agency B"],
            "agency_url": ["www.mobilitydata.org", "www.someotherurl.com"],
            "csv_row_number": [1, 2],
        }),
        "routes": pl.DataFrame({
            "route_id": ["route1"],
            "route_url": ["www.atotallydifferenturl.com"],
            "csv_row_number": [4],
        }),
    }
    notices = validate_url_consistency(feed, CTX)
    assert notices == []


def test_same_route_and_agency_url_generates_notice() -> None:
    """Route URL matches agency URL -> notice."""
    feed = {
        "agency": pl.DataFrame({
            "agency_name": ["Agency A", "Agency B", "Agency C"],
            "agency_url": ["www.mobilitydata.org", "www.anotherurl.com", "www.MobilityData.org"],
            "csv_row_number": [1, 2, 3],
        }),
        "routes": pl.DataFrame({
            "route_id": ["route1", "route2"],
            "route_url": ["www.mobilitydata.org", None],
            "csv_row_number": [4, 5],
        }),
    }
    notices = validate_url_consistency(feed, CTX)
    # Should match both Agency A and Agency C (case-insensitive)
    route_notices = [n for n in notices if n.code == "same_route_and_agency_url"]
    assert len(route_notices) == 2
    for n in route_notices:
        assert n.severity == Severity.WARNING
        assert n.fields["route_id"] == "route1"
        assert n.fields["route_url"] == "www.mobilitydata.org"


def test_different_stop_and_agency_url_no_notice() -> None:
    """Different URLs for stop and agency -> no notice."""
    feed = {
        "agency": pl.DataFrame({
            "agency_name": ["Agency A"],
            "agency_url": ["www.mobilitydata.org"],
            "csv_row_number": [1],
        }),
        "stops": pl.DataFrame({
            "stop_id": ["stop1"],
            "stop_url": ["www.stopurl.com"],
            "csv_row_number": [44],
        }),
    }
    notices = validate_url_consistency(feed, CTX)
    assert notices == []


def test_same_stop_and_agency_url_generates_notice() -> None:
    """Stop URL matches agency URL -> notice."""
    feed = {
        "agency": pl.DataFrame({
            "agency_name": ["Agency A", "Agency B", "Agency C", "Agency D"],
            "agency_url": ["www.mobilitydata.org", "www.anotherurl.com", "www.mobilitydata.org", None],
            "csv_row_number": [1, 2, 3, 4],
        }),
        "stops": pl.DataFrame({
            "stop_id": ["stop1", "stop2", "stop3"],
            "stop_url": ["www.mobilitydata.org", None, "www.anotherurl.com"],
            "csv_row_number": [456, 55, 77],
        }),
    }
    notices = validate_url_consistency(feed, CTX)
    stop_agency_notices = [n for n in notices if n.code == "same_stop_and_agency_url"]
    # stop1 matches Agency A and Agency C (same URL, case-insensitive)
    # stop3 matches Agency B
    assert len(stop_agency_notices) == 3
    for n in stop_agency_notices:
        assert n.severity == Severity.WARNING


def test_different_stop_and_route_url_no_notice() -> None:
    """Different URLs for stop and route -> no notice."""
    feed = {
        "routes": pl.DataFrame({
            "route_id": ["route1"],
            "route_url": ["www.mobilitydata.org"],
            "csv_row_number": [8],
        }),
        "stops": pl.DataFrame({
            "stop_id": ["stop1"],
            "stop_url": ["www.stopurl.com"],
            "csv_row_number": [44],
        }),
    }
    notices = validate_url_consistency(feed, CTX)
    assert notices == []


def test_same_stop_and_route_url_generates_notice() -> None:
    """Stop URL matches route URL -> notice."""
    feed = {
        "routes": pl.DataFrame({
            "route_id": ["route1"],
            "route_url": ["www.mobilitydata.org"],
            "csv_row_number": [5],
        }),
        "stops": pl.DataFrame({
            "stop_id": ["stop1", "stop2"],
            "stop_url": ["www.mobilitydata.org", "www.mobilitYData.org"],
            "csv_row_number": [456, 88],
        }),
    }
    notices = validate_url_consistency(feed, CTX)
    stop_route_notices = [n for n in notices if n.code == "same_stop_and_route_url"]
    # Both stops match the route (case-insensitive)
    assert len(stop_route_notices) == 2
    for n in stop_route_notices:
        assert n.severity == Severity.WARNING
        assert n.fields["route_id"] == "route1"


def test_no_tables_no_notice() -> None:
    """No tables present -> no notices."""
    feed: dict[str, pl.DataFrame] = {}
    notices = validate_url_consistency(feed, CTX)
    assert notices == []


def test_empty_tables_no_notice() -> None:
    """Empty tables -> no notices."""
    feed = {
        "agency": pl.DataFrame({
            "agency_name": pl.Series([], dtype=pl.String),
            "agency_url": pl.Series([], dtype=pl.String),
            "csv_row_number": pl.Series([], dtype=pl.Int64),
        }),
        "routes": pl.DataFrame({
            "route_id": pl.Series([], dtype=pl.String),
            "route_url": pl.Series([], dtype=pl.String),
            "csv_row_number": pl.Series([], dtype=pl.Int64),
        }),
    }
    notices = validate_url_consistency(feed, CTX)
    assert notices == []


def test_route_without_url_skipped() -> None:
    """Route without URL is skipped."""
    feed = {
        "agency": pl.DataFrame({
            "agency_name": ["Agency A"],
            "agency_url": ["www.mobilitydata.org"],
            "csv_row_number": [1],
        }),
        "routes": pl.DataFrame({
            "route_id": ["route1"],
            "route_url": [None],
            "csv_row_number": [2],
        }),
    }
    notices = validate_url_consistency(feed, CTX)
    assert notices == []


def test_stop_without_url_skipped() -> None:
    """Stop without URL is skipped."""
    feed = {
        "agency": pl.DataFrame({
            "agency_name": ["Agency A"],
            "agency_url": ["www.mobilitydata.org"],
            "csv_row_number": [1],
        }),
        "stops": pl.DataFrame({
            "stop_id": ["stop1"],
            "stop_url": [None],
            "csv_row_number": [2],
        }),
    }
    notices = validate_url_consistency(feed, CTX)
    assert notices == []


def test_ctx_unused() -> None:
    """ctx value has no effect on the output."""
    wrong_ctx = ValidationContext(country_code="XX", date_for_validation=date(1900, 1, 1))
    feed = {
        "agency": pl.DataFrame({
            "agency_name": ["Agency A"],
            "agency_url": ["www.example.com"],
            "csv_row_number": [1],
        }),
        "routes": pl.DataFrame({
            "route_id": ["route1"],
            "route_url": ["www.example.com"],
            "csv_row_number": [2],
        }),
    }
    notices = validate_url_consistency(feed, wrong_ctx)
    assert len(notices) == 1