"""Tests for attribution validators."""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from gtfs_validator.context import ValidationContext
from gtfs_validator.notices import Severity
from gtfs_validator.validators.attribution import validate_attribution_without_role

CTX = ValidationContext(country_code="US", date_for_validation=date(2024, 6, 1))


# ------------------------------------------------------------------
# Test 1: notice emitted when no role assigned
# ------------------------------------------------------------------


def test_attribution_without_role_generates_notice() -> None:
    feed = {
        "attributions": pl.DataFrame({
            "csv_row_number": [2],
            "attribution_id": ["attr-1"],
            "is_producer": [0],
            "is_operator": [0],
            "is_authority": [0],
        })
    }
    notices = validate_attribution_without_role(feed, CTX)
    assert len(notices) == 1
    n = notices[0]
    assert n.code == "attribution_without_role"
    assert n.severity == Severity.WARNING
    assert n.fields == {"csv_row_number": 2, "attribution_id": "attr-1"}


# ------------------------------------------------------------------
# Test 2: empty attribution_id still produces notice
# ------------------------------------------------------------------


def test_attribution_without_role_no_id_generates_notice() -> None:
    feed = {
        "attributions": pl.DataFrame({
            "csv_row_number": [2],
            "attribution_id": [""],
            "is_producer": [0],
            "is_operator": [0],
            "is_authority": [0],
        })
    }
    notices = validate_attribution_without_role(feed, CTX)
    assert len(notices) == 1
    assert notices[0].fields["attribution_id"] == ""


# ------------------------------------------------------------------
# Test 3: single role assigned -> no notice
# ------------------------------------------------------------------


@pytest.mark.parametrize(
    "role_col",
    ["is_authority", "is_operator", "is_producer"],
)
def test_attribution_with_single_role_no_notice(role_col: str) -> None:
    data: dict[str, list] = {  # type: ignore[type-arg]
        "csv_row_number": [2],
        "attribution_id": ["attr-1"],
        "is_producer": [0],
        "is_operator": [0],
        "is_authority": [0],
    }
    data[role_col] = [1]
    feed = {"attributions": pl.DataFrame(data)}
    notices = validate_attribution_without_role(feed, CTX)
    assert len(notices) == 0


# ------------------------------------------------------------------
# Test 4: all roles assigned -> no notice
# ------------------------------------------------------------------


def test_attribution_with_all_roles_no_notice() -> None:
    feed = {
        "attributions": pl.DataFrame({
            "csv_row_number": [2],
            "attribution_id": ["attr-1"],
            "is_producer": [1],
            "is_operator": [1],
            "is_authority": [1],
        })
    }
    notices = validate_attribution_without_role(feed, CTX)
    assert len(notices) == 0


# ------------------------------------------------------------------
# Test 5: missing attributions table
# ------------------------------------------------------------------


def test_missing_attributions_table() -> None:
    feed: dict[str, pl.DataFrame] = {}
    notices = validate_attribution_without_role(feed, CTX)
    assert len(notices) == 0


# ------------------------------------------------------------------
# Test 6: empty DataFrame
# ------------------------------------------------------------------


def test_empty_dataframe() -> None:
    feed = {
        "attributions": pl.DataFrame(
            schema={
                "csv_row_number": pl.Int64,
                "attribution_id": pl.Utf8,
                "is_producer": pl.Int64,
                "is_operator": pl.Int64,
                "is_authority": pl.Int64,
            }
        )
    }
    notices = validate_attribution_without_role(feed, CTX)
    assert len(notices) == 0


# ------------------------------------------------------------------
# Test 7: no role columns present at all
# ------------------------------------------------------------------


def test_no_role_columns_present() -> None:
    feed = {
        "attributions": pl.DataFrame({
            "csv_row_number": [2],
            "attribution_id": ["attr-1"],
            "organization_name": ["ACME"],
        })
    }
    notices = validate_attribution_without_role(feed, CTX)
    assert len(notices) == 0


# ------------------------------------------------------------------
# Test 8: partial role columns, no role assigned
# ------------------------------------------------------------------


def test_partial_role_columns_no_role_assigned() -> None:
    feed = {
        "attributions": pl.DataFrame({
            "csv_row_number": [2],
            "attribution_id": ["attr-1"],
            "is_producer": [0],
        })
    }
    notices = validate_attribution_without_role(feed, CTX)
    assert len(notices) == 1


# ------------------------------------------------------------------
# Test 9: partial role columns, role assigned
# ------------------------------------------------------------------


def test_partial_role_columns_role_assigned() -> None:
    feed = {
        "attributions": pl.DataFrame({
            "csv_row_number": [2],
            "attribution_id": ["attr-1"],
            "is_producer": [1],
        })
    }
    notices = validate_attribution_without_role(feed, CTX)
    assert len(notices) == 0


# ------------------------------------------------------------------
# Test 10: null role values treated as not assigned
# ------------------------------------------------------------------


def test_null_role_values_treated_as_not_assigned() -> None:
    feed = {
        "attributions": pl.DataFrame({
            "csv_row_number": [2],
            "attribution_id": ["attr-1"],
            "is_producer": [None],
            "is_operator": [None],
            "is_authority": [None],
        }).cast({"is_producer": pl.Int64, "is_operator": pl.Int64, "is_authority": pl.Int64})
    }
    notices = validate_attribution_without_role(feed, CTX)
    assert len(notices) == 1
    assert notices[0].fields["attribution_id"] == "attr-1"


# ------------------------------------------------------------------
# Test 11: multiple rows, mixed results
# ------------------------------------------------------------------


def test_multiple_rows_mixed() -> None:
    feed = {
        "attributions": pl.DataFrame({
            "csv_row_number": [2, 3, 4],
            "attribution_id": ["has-role", "no-role", "null-role"],
            "is_producer": [1, 0, None],
            "is_operator": [0, 0, None],
            "is_authority": [0, 0, None],
        }).cast({"is_producer": pl.Int64, "is_operator": pl.Int64, "is_authority": pl.Int64})
    }
    notices = validate_attribution_without_role(feed, CTX)
    assert len(notices) == 2
    ids = {n.fields["attribution_id"] for n in notices}
    assert ids == {"no-role", "null-role"}
