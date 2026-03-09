"""Tests for validate_route_agency_id."""

from __future__ import annotations

from datetime import date

import polars as pl

from gtfs_validator.context import ValidationContext
from gtfs_validator.notices import Severity
from gtfs_validator.validators.route_agency_id import validate_route_agency_id

CTX = ValidationContext(country_code="US", date_for_validation=date(2026, 3, 8))


def _ctx() -> ValidationContext:
    return CTX


def _agency(n: int) -> pl.DataFrame:
    """Return an agency DataFrame with n rows (content irrelevant)."""
    return pl.DataFrame({"agency_id": [f"agency{i}" for i in range(n)]})


def test_multi_agency_missing_agency_id_emits_error() -> None:
    """Multi-agency feed: route with null agency_id emits ERROR."""
    feed = {
        "agency": _agency(2),
        "routes": pl.DataFrame(
            {
                "csv_row_number": [0, 1],
                "agency_id": ["agency0", None],
            }
        ),
    }
    notices = validate_route_agency_id(feed, _ctx())
    assert len(notices) == 1
    n = notices[0]
    assert n.code == "missing_required_agency_id"
    assert n.severity == Severity.ERROR
    assert n.fields["filename"] == "routes.txt"
    assert n.fields["csv_row_number"] == 1
    assert n.fields["agency_name"] is None


def test_single_agency_missing_agency_id_emits_warning() -> None:
    """Single-agency feed: route with null agency_id emits WARNING."""
    feed = {
        "agency": _agency(1),
        "routes": pl.DataFrame(
            {
                "csv_row_number": [0],
                "agency_id": [None],
            }
        ),
    }
    notices = validate_route_agency_id(feed, _ctx())
    assert len(notices) == 1
    n = notices[0]
    assert n.code == "missing_recommended_field"
    assert n.severity == Severity.WARNING
    assert n.fields["filename"] == "routes.txt"
    assert n.fields["csv_row_number"] == 0
    assert n.fields["field_name"] == "agency_id"


def test_single_agency_all_routes_have_agency_id_no_notice() -> None:
    """Single-agency feed: all routes have agency_id → no notices."""
    feed = {
        "agency": _agency(1),
        "routes": pl.DataFrame(
            {
                "csv_row_number": [0, 1],
                "agency_id": ["agency1", "agency1"],
            }
        ),
    }
    assert validate_route_agency_id(feed, _ctx()) == []


def test_multi_agency_all_routes_have_agency_id_no_notice() -> None:
    """Multi-agency feed: all routes have agency_id → no notices."""
    feed = {
        "agency": _agency(2),
        "routes": pl.DataFrame(
            {
                "csv_row_number": [0, 1],
                "agency_id": ["agency0", "agency1"],
            }
        ),
    }
    assert validate_route_agency_id(feed, _ctx()) == []


def test_agency_absent_skips_validation() -> None:
    """Feed without agency table → no notices."""
    feed = {
        "routes": pl.DataFrame(
            {
                "csv_row_number": [0],
                "agency_id": [None],
            }
        ),
    }
    assert validate_route_agency_id(feed, _ctx()) == []


def test_agency_empty_skips_validation() -> None:
    """Empty agency table → no notices."""
    feed = {
        "agency": pl.DataFrame({"agency_id": []}).cast({"agency_id": pl.Utf8}),
        "routes": pl.DataFrame(
            {
                "csv_row_number": [0],
                "agency_id": [None],
            }
        ),
    }
    assert validate_route_agency_id(feed, _ctx()) == []


def test_routes_absent_skips_validation() -> None:
    """Feed without routes table → no notices."""
    feed = {
        "agency": _agency(1),
    }
    assert validate_route_agency_id(feed, _ctx()) == []


def test_routes_empty_skips_validation() -> None:
    """Empty routes table → no notices."""
    feed = {
        "agency": _agency(1),
        "routes": pl.DataFrame(
            {"csv_row_number": [], "agency_id": []}
        ).cast({"csv_row_number": pl.Int64, "agency_id": pl.Utf8}),
    }
    assert validate_route_agency_id(feed, _ctx()) == []


def test_agency_id_column_absent_treated_as_null() -> None:
    """Routes without agency_id column (all-null) in multi-agency feed → 2 ERRORs."""
    feed = {
        "agency": _agency(2),
        "routes": pl.DataFrame(
            {
                "csv_row_number": [0, 1],
                "agency_id": [None, None],
            }
        ),
    }
    notices = validate_route_agency_id(feed, _ctx())
    assert len(notices) == 2
    for n in notices:
        assert n.code == "missing_required_agency_id"
        assert n.severity == Severity.ERROR


def test_multi_agency_only_some_routes_missing_agency_id() -> None:
    """Multi-agency feed: only route 1 is missing agency_id → 1 ERROR for row 1."""
    feed = {
        "agency": _agency(3),
        "routes": pl.DataFrame(
            {
                "csv_row_number": [0, 1, 2],
                "agency_id": ["agency0", None, "agency2"],
            }
        ),
    }
    notices = validate_route_agency_id(feed, _ctx())
    assert len(notices) == 1
    n = notices[0]
    assert n.code == "missing_required_agency_id"
    assert n.fields["csv_row_number"] == 1
