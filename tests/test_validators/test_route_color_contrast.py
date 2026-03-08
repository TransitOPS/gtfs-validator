"""Tests for RouteColorContrastValidator."""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from gtfs_validator.context import ValidationContext
from gtfs_validator.notices import Severity
from gtfs_validator.validators.route_color_contrast import validate_route_color_contrast

CTX = ValidationContext(country_code="US", date_for_validation=date(2024, 1, 1))


def make_routes(rows: list[dict]) -> pl.DataFrame:
    """Create a routes DataFrame with columns used by this validator."""
    if not rows:
        return pl.DataFrame(
            schema={
                "route_id": pl.Utf8,
                "csv_row_number": pl.Int64,
                "route_color": pl.Utf8,
                "route_text_color": pl.Utf8,
            }
        )
    return pl.DataFrame(rows).cast(
        {
            "route_id": pl.Utf8,
            "csv_row_number": pl.Int64,
            "route_color": pl.Utf8,
            "route_text_color": pl.Utf8,
        }
    )


def test_no_route_color_should_not_generate_notice():
    """Route with null route_color should produce no notices."""
    routes = make_routes(
        [{"route_id": "r1", "csv_row_number": 2, "route_color": None, "route_text_color": "0000DE"}]
    )
    notices = validate_route_color_contrast({"routes": routes}, CTX)
    assert notices == []


def test_no_route_text_color_should_not_generate_notice():
    """Route with null route_text_color should produce no notices."""
    routes = make_routes(
        [{"route_id": "r1", "csv_row_number": 2, "route_color": "0000DE", "route_text_color": None}]
    )
    notices = validate_route_color_contrast({"routes": routes}, CTX)
    assert notices == []


def test_both_colors_absent_should_not_generate_notice():
    """Route with both colors null should produce no notices."""
    routes = make_routes(
        [{"route_id": "r1", "csv_row_number": 2, "route_color": None, "route_text_color": None}]
    )
    notices = validate_route_color_contrast({"routes": routes}, CTX)
    assert notices == []


def test_contrasting_colors_should_not_generate_notice():
    """White and black have sufficient contrast (luma diff = 255)."""
    routes = make_routes(
        [{"route_id": "r1", "csv_row_number": 2, "route_color": "FFFFFF", "route_text_color": "000000"}]
    )
    notices = validate_route_color_contrast({"routes": routes}, CTX)
    assert notices == []


def test_non_contrasting_colors_should_generate_notice():
    """route_color='4a4444' and route_text_color='3d3838' have luma diff=12, below threshold."""
    routes = make_routes(
        [
            {
                "route_id": "route id value",
                "csv_row_number": 2,
                "route_color": "4a4444",
                "route_text_color": "3d3838",
            }
        ]
    )
    notices = validate_route_color_contrast({"routes": routes}, CTX)
    assert len(notices) == 1
    n = notices[0]
    assert n.code == "route_color_contrast"
    assert n.severity == Severity.WARNING
    assert n.fields["route_id"] == "route id value"
    assert n.fields["csv_row_number"] == 2
    assert n.fields["route_color"] == "4a4444"
    assert n.fields["route_text_color"] == "3d3838"


def test_identical_colors_should_generate_notice():
    """Same color for route_color and route_text_color yields luma diff=0."""
    routes = make_routes(
        [{"route_id": "r1", "csv_row_number": 2, "route_color": "FF0000", "route_text_color": "FF0000"}]
    )
    notices = validate_route_color_contrast({"routes": routes}, CTX)
    assert len(notices) == 1


def test_luma_diff_exactly_72_should_not_generate_notice():
    """Luma diff of exactly 72 is not strictly less than threshold; no notice."""
    # "000000" luma=0, "484848" (v=72 greyscale) luma=72; diff=72
    routes = make_routes(
        [{"route_id": "r1", "csv_row_number": 2, "route_color": "000000", "route_text_color": "484848"}]
    )
    notices = validate_route_color_contrast({"routes": routes}, CTX)
    assert notices == []


def test_luma_diff_exactly_71_should_generate_notice():
    """Luma diff of 71 is strictly less than 72; notice is emitted."""
    # "000000" luma=0, "474747" (v=71 greyscale) luma=71; diff=71
    routes = make_routes(
        [{"route_id": "r1", "csv_row_number": 2, "route_color": "000000", "route_text_color": "474747"}]
    )
    notices = validate_route_color_contrast({"routes": routes}, CTX)
    assert len(notices) == 1


def test_missing_table_no_notice():
    """No 'routes' key in feed produces no notices."""
    notices = validate_route_color_contrast({}, CTX)
    assert notices == []


def test_empty_table_no_notice():
    """Empty routes DataFrame produces no notices."""
    routes = make_routes([])
    notices = validate_route_color_contrast({"routes": routes}, CTX)
    assert notices == []


def test_multiple_routes_mixed_results():
    """Only the non-contrasting row generates a notice."""
    routes = make_routes(
        [
            {"route_id": "r1", "csv_row_number": 2, "route_color": "FFFFFF", "route_text_color": "000000"},
            {"route_id": "r2", "csv_row_number": 3, "route_color": "4a4444", "route_text_color": "3d3838"},
        ]
    )
    notices = validate_route_color_contrast({"routes": routes}, CTX)
    assert len(notices) == 1
    assert notices[0].fields["route_id"] == "r2"


def test_multiple_failing_routes_emit_one_notice_each():
    """Two non-contrasting rows each generate one notice."""
    routes = make_routes(
        [
            {"route_id": "r1", "csv_row_number": 2, "route_color": "4a4444", "route_text_color": "3d3838"},
            {"route_id": "r2", "csv_row_number": 3, "route_color": "111111", "route_text_color": "222222"},
        ]
    )
    notices = validate_route_color_contrast({"routes": routes}, CTX)
    assert len(notices) == 2
    route_ids = {n.fields["route_id"] for n in notices}
    assert route_ids == {"r1", "r2"}


def test_notice_code_and_severity():
    """Notice has correct code and severity."""
    routes = make_routes(
        [{"route_id": "r1", "csv_row_number": 2, "route_color": "4a4444", "route_text_color": "3d3838"}]
    )
    notices = validate_route_color_contrast({"routes": routes}, CTX)
    assert len(notices) == 1
    assert notices[0].code == "route_color_contrast"
    assert notices[0].severity == Severity.WARNING


def test_all_null_colors_no_notices():
    """All rows with null colors produce no notices."""
    routes = make_routes(
        [
            {"route_id": "r1", "csv_row_number": 2, "route_color": None, "route_text_color": None},
            {"route_id": "r2", "csv_row_number": 3, "route_color": None, "route_text_color": None},
            {"route_id": "r3", "csv_row_number": 4, "route_color": None, "route_text_color": None},
        ]
    )
    notices = validate_route_color_contrast({"routes": routes}, CTX)
    assert notices == []
