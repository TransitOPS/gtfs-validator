"""Tests for validate_fare_product_default_rider_categories."""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from gtfs_validator.context import ValidationContext
from gtfs_validator.notices import Severity
from gtfs_validator.validators.fare_product_default_rider_categories import (
    validate_fare_product_default_rider_categories,
)

CTX = ValidationContext(country_code="US", date_for_validation=date(2024, 1, 1))


def make_rider_categories(rows: list[dict]) -> pl.DataFrame:
    """Build a rider_categories DataFrame from a list of dicts.

    Keys: rider_category_id (str), is_default_fare_category (int).
    """
    return pl.DataFrame({
        "rider_category_id": [r["rider_category_id"] for r in rows],
        "is_default_fare_category": [r["is_default_fare_category"] for r in rows],
    })


def make_fare_products(rows: list[dict]) -> pl.DataFrame:
    """Build a fare_products DataFrame from a list of dicts.

    Keys: fare_product_id (str), rider_category_id (str | None),
    csv_row_number (int). fare_media_id is omitted unless needed.
    """
    return pl.DataFrame({
        "fare_product_id": [r["fare_product_id"] for r in rows],
        "rider_category_id": [r.get("rider_category_id") for r in rows],
        "csv_row_number": [r["csv_row_number"] for r in rows],
    })


def make_fare_products_with_media(rows: list[dict]) -> pl.DataFrame:
    """Build a fare_products DataFrame including fare_media_id column."""
    return pl.DataFrame({
        "fare_product_id": [r["fare_product_id"] for r in rows],
        "rider_category_id": [r.get("rider_category_id") for r in rows],
        "fare_media_id": [r.get("fare_media_id") for r in rows],
        "csv_row_number": [r["csv_row_number"] for r in rows],
    })


# ---------------------------------------------------------------------------
# Test 1
# ---------------------------------------------------------------------------

def test_multiple_default_rider_categories_triggers_notice() -> None:
    """Two distinct IS_DEFAULT rider categories for the same fare product → 1 ERROR notice."""
    rc = make_rider_categories([
        {"rider_category_id": "rider1", "is_default_fare_category": 1},
        {"rider_category_id": "rider2", "is_default_fare_category": 1},
        {"rider_category_id": "rider3", "is_default_fare_category": 0},
    ])
    fp = make_fare_products([
        {"fare_product_id": "fare1", "rider_category_id": "rider1", "csv_row_number": 1},
        {"fare_product_id": "fare1", "rider_category_id": "rider2", "csv_row_number": 2},
    ])
    notices = validate_fare_product_default_rider_categories(
        {"fare_products": fp, "rider_categories": rc}, CTX
    )
    assert len(notices) == 1
    n = notices[0]
    assert n.code == "fare_product_with_multiple_default_rider_categories"
    assert n.severity == Severity.ERROR
    assert n.fields["fare_product_id"] == "fare1"
    assert n.fields["csv_row_number1"] == 1
    assert n.fields["csv_row_number2"] == 2
    assert n.fields["rider_category_id1"] == "rider1"
    assert n.fields["rider_category_id2"] == "rider2"


# ---------------------------------------------------------------------------
# Test 2
# ---------------------------------------------------------------------------

def test_one_default_rider_category_no_notice() -> None:
    """Only one IS_DEFAULT rider category for a fare product → no notice."""
    rc = make_rider_categories([
        {"rider_category_id": "rider1", "is_default_fare_category": 1},
        {"rider_category_id": "rider2", "is_default_fare_category": 0},
        {"rider_category_id": "rider3", "is_default_fare_category": 0},
    ])
    fp = make_fare_products([
        {"fare_product_id": "fare1", "rider_category_id": "rider1", "csv_row_number": 1},
        {"fare_product_id": "fare1", "rider_category_id": "rider2", "csv_row_number": 2},
    ])
    notices = validate_fare_product_default_rider_categories(
        {"fare_products": fp, "rider_categories": rc}, CTX
    )
    assert notices == []


# ---------------------------------------------------------------------------
# Test 3
# ---------------------------------------------------------------------------

def test_default_categories_different_product_ids_no_notice() -> None:
    """Same default rider category across two different fare products → no notice."""
    rc = make_rider_categories([
        {"rider_category_id": "rider1", "is_default_fare_category": 1},
        {"rider_category_id": "rider2", "is_default_fare_category": 0},
        {"rider_category_id": "rider3", "is_default_fare_category": 0},
    ])
    fp = make_fare_products([
        {"fare_product_id": "fare1", "rider_category_id": "rider1", "csv_row_number": 1},
        {"fare_product_id": "fare2", "rider_category_id": "rider1", "csv_row_number": 2},
    ])
    notices = validate_fare_product_default_rider_categories(
        {"fare_products": fp, "rider_categories": rc}, CTX
    )
    assert notices == []


# ---------------------------------------------------------------------------
# Test 4
# ---------------------------------------------------------------------------

def test_default_categories_different_product_ids_triggers_notice() -> None:
    """fare2 has two distinct default rider categories; fare1 has one → 1 notice for fare2."""
    rc = make_rider_categories([
        {"rider_category_id": "rider1", "is_default_fare_category": 1},
        {"rider_category_id": "rider2", "is_default_fare_category": 1},
        {"rider_category_id": "rider3", "is_default_fare_category": 0},
    ])
    fp = make_fare_products([
        {"fare_product_id": "fare1", "rider_category_id": "rider1", "csv_row_number": 1},
        {"fare_product_id": "fare2", "rider_category_id": "rider1", "csv_row_number": 2},
        {"fare_product_id": "fare2", "rider_category_id": "rider2", "csv_row_number": 3},
    ])
    notices = validate_fare_product_default_rider_categories(
        {"fare_products": fp, "rider_categories": rc}, CTX
    )
    assert len(notices) == 1
    n = notices[0]
    assert n.fields["fare_product_id"] == "fare2"
    assert n.fields["csv_row_number1"] == 2
    assert n.fields["csv_row_number2"] == 3
    assert n.fields["rider_category_id1"] == "rider1"
    assert n.fields["rider_category_id2"] == "rider2"


# ---------------------------------------------------------------------------
# Test 5
# ---------------------------------------------------------------------------

def test_same_rider_category_different_fare_media_ids_no_notice() -> None:
    """Same rider_category_id repeated with different fare_media_id → de-duped to one default."""
    rc = make_rider_categories([
        {"rider_category_id": "rider1", "is_default_fare_category": 1},
        {"rider_category_id": "rider2", "is_default_fare_category": 0},
        {"rider_category_id": "rider3", "is_default_fare_category": 0},
    ])
    fp = make_fare_products_with_media([
        {"fare_product_id": "fare1", "rider_category_id": "rider1", "fare_media_id": "media1", "csv_row_number": 1},
        {"fare_product_id": "fare1", "rider_category_id": "rider1", "fare_media_id": "media2", "csv_row_number": 2},
        {"fare_product_id": "fare2", "rider_category_id": "rider1", "fare_media_id": "media2", "csv_row_number": 3},
    ])
    notices = validate_fare_product_default_rider_categories(
        {"fare_products": fp, "rider_categories": rc}, CTX
    )
    assert notices == []


# ---------------------------------------------------------------------------
# Test 6
# ---------------------------------------------------------------------------

def test_rider_categories_absent_skips() -> None:
    """No rider_categories key in feed → returns empty list."""
    fp = make_fare_products([
        {"fare_product_id": "fare1", "rider_category_id": "rider1", "csv_row_number": 1},
    ])
    notices = validate_fare_product_default_rider_categories(
        {"fare_products": fp}, CTX
    )
    assert notices == []


# ---------------------------------------------------------------------------
# Test 7
# ---------------------------------------------------------------------------

def test_fare_products_absent_skips() -> None:
    """No fare_products key in feed → returns empty list."""
    rc = make_rider_categories([
        {"rider_category_id": "rider1", "is_default_fare_category": 1},
    ])
    notices = validate_fare_product_default_rider_categories(
        {"rider_categories": rc}, CTX
    )
    assert notices == []


# ---------------------------------------------------------------------------
# Test 8
# ---------------------------------------------------------------------------

def test_rider_categories_empty_skips() -> None:
    """Empty rider_categories DataFrame → returns empty list."""
    rc = pl.DataFrame({
        "rider_category_id": pl.Series([], dtype=pl.Utf8),
        "is_default_fare_category": pl.Series([], dtype=pl.Int64),
    })
    fp = make_fare_products([
        {"fare_product_id": "fare1", "rider_category_id": "rider1", "csv_row_number": 1},
    ])
    notices = validate_fare_product_default_rider_categories(
        {"fare_products": fp, "rider_categories": rc}, CTX
    )
    assert notices == []


# ---------------------------------------------------------------------------
# Test 9
# ---------------------------------------------------------------------------

def test_fare_products_empty_skips() -> None:
    """Non-empty rider_categories but empty fare_products → returns empty list."""
    rc = make_rider_categories([
        {"rider_category_id": "rider1", "is_default_fare_category": 1},
    ])
    fp = pl.DataFrame({
        "fare_product_id": pl.Series([], dtype=pl.Utf8),
        "rider_category_id": pl.Series([], dtype=pl.Utf8),
        "csv_row_number": pl.Series([], dtype=pl.Int64),
    })
    notices = validate_fare_product_default_rider_categories(
        {"fare_products": fp, "rider_categories": rc}, CTX
    )
    assert notices == []


# ---------------------------------------------------------------------------
# Test 10
# ---------------------------------------------------------------------------

def test_both_tables_empty_skips() -> None:
    """Both tables empty → returns empty list."""
    rc = pl.DataFrame({
        "rider_category_id": pl.Series([], dtype=pl.Utf8),
        "is_default_fare_category": pl.Series([], dtype=pl.Int64),
    })
    fp = pl.DataFrame({
        "fare_product_id": pl.Series([], dtype=pl.Utf8),
        "rider_category_id": pl.Series([], dtype=pl.Utf8),
        "csv_row_number": pl.Series([], dtype=pl.Int64),
    })
    notices = validate_fare_product_default_rider_categories(
        {"fare_products": fp, "rider_categories": rc}, CTX
    )
    assert notices == []


# ---------------------------------------------------------------------------
# Test 11
# ---------------------------------------------------------------------------

def test_three_distinct_defaults_one_notice_first_two_only() -> None:
    """Three distinct IS_DEFAULT rider categories → exactly 1 notice referencing only first two."""
    rc = make_rider_categories([
        {"rider_category_id": "rider1", "is_default_fare_category": 1},
        {"rider_category_id": "rider2", "is_default_fare_category": 1},
        {"rider_category_id": "rider3", "is_default_fare_category": 1},
    ])
    fp = make_fare_products([
        {"fare_product_id": "fare1", "rider_category_id": "rider1", "csv_row_number": 1},
        {"fare_product_id": "fare1", "rider_category_id": "rider2", "csv_row_number": 2},
        {"fare_product_id": "fare1", "rider_category_id": "rider3", "csv_row_number": 3},
    ])
    notices = validate_fare_product_default_rider_categories(
        {"fare_products": fp, "rider_categories": rc}, CTX
    )
    assert len(notices) == 1
    n = notices[0]
    assert n.fields["fare_product_id"] == "fare1"
    assert n.fields["csv_row_number1"] == 1
    assert n.fields["csv_row_number2"] == 2
    assert n.fields["rider_category_id1"] == "rider1"
    assert n.fields["rider_category_id2"] == "rider2"
    # rider3/row3 must not appear
    assert n.fields.get("csv_row_number3") is None
    assert n.fields.get("rider_category_id3") is None


# ---------------------------------------------------------------------------
# Test 12
# ---------------------------------------------------------------------------

def test_two_products_each_with_two_defaults_two_notices() -> None:
    """Two fare products each with two distinct IS_DEFAULT categories → 2 notices."""
    rc = make_rider_categories([
        {"rider_category_id": "rider1", "is_default_fare_category": 1},
        {"rider_category_id": "rider2", "is_default_fare_category": 1},
        {"rider_category_id": "rider3", "is_default_fare_category": 1},
        {"rider_category_id": "rider4", "is_default_fare_category": 1},
    ])
    fp = make_fare_products([
        {"fare_product_id": "fareA", "rider_category_id": "rider1", "csv_row_number": 1},
        {"fare_product_id": "fareA", "rider_category_id": "rider2", "csv_row_number": 2},
        {"fare_product_id": "fareB", "rider_category_id": "rider3", "csv_row_number": 3},
        {"fare_product_id": "fareB", "rider_category_id": "rider4", "csv_row_number": 4},
    ])
    notices = validate_fare_product_default_rider_categories(
        {"fare_products": fp, "rider_categories": rc}, CTX
    )
    assert len(notices) == 2
    by_product = {n.fields["fare_product_id"]: n for n in notices}
    assert "fareA" in by_product
    assert "fareB" in by_product
    nA = by_product["fareA"]
    assert nA.fields["csv_row_number1"] == 1
    assert nA.fields["csv_row_number2"] == 2
    assert nA.fields["rider_category_id1"] == "rider1"
    assert nA.fields["rider_category_id2"] == "rider2"
    nB = by_product["fareB"]
    assert nB.fields["csv_row_number1"] == 3
    assert nB.fields["csv_row_number2"] == 4
    assert nB.fields["rider_category_id1"] == "rider3"
    assert nB.fields["rider_category_id2"] == "rider4"


# ---------------------------------------------------------------------------
# Test 13
# ---------------------------------------------------------------------------

def test_rider_category_id_not_in_rider_categories_no_notice() -> None:
    """Unknown rider_category_id (unresolved FK) is silently ignored → no notice."""
    rc = make_rider_categories([
        {"rider_category_id": "rider1", "is_default_fare_category": 1},
    ])
    fp = make_fare_products([
        {"fare_product_id": "fare1", "rider_category_id": "unknown_rider", "csv_row_number": 1},
        {"fare_product_id": "fare1", "rider_category_id": "rider1", "csv_row_number": 2},
    ])
    notices = validate_fare_product_default_rider_categories(
        {"fare_products": fp, "rider_categories": rc}, CTX
    )
    assert notices == []


# ---------------------------------------------------------------------------
# Test 14
# ---------------------------------------------------------------------------

def test_not_default_rider_category_ignored() -> None:
    """All rider categories NOT_DEFAULT → no notices."""
    rc = make_rider_categories([
        {"rider_category_id": "rider1", "is_default_fare_category": 0},
        {"rider_category_id": "rider2", "is_default_fare_category": 0},
    ])
    fp = make_fare_products([
        {"fare_product_id": "fare1", "rider_category_id": "rider1", "csv_row_number": 1},
        {"fare_product_id": "fare1", "rider_category_id": "rider2", "csv_row_number": 2},
    ])
    notices = validate_fare_product_default_rider_categories(
        {"fare_products": fp, "rider_categories": rc}, CTX
    )
    assert notices == []


# ---------------------------------------------------------------------------
# Test 15
# ---------------------------------------------------------------------------

def test_notice_fields_exact_values() -> None:
    """Verify all seven notice properties exactly for a basic violation."""
    rc = make_rider_categories([
        {"rider_category_id": "rider1", "is_default_fare_category": 1},
        {"rider_category_id": "rider2", "is_default_fare_category": 1},
    ])
    fp = make_fare_products([
        {"fare_product_id": "fare1", "rider_category_id": "rider1", "csv_row_number": 5},
        {"fare_product_id": "fare1", "rider_category_id": "rider2", "csv_row_number": 8},
    ])
    notices = validate_fare_product_default_rider_categories(
        {"fare_products": fp, "rider_categories": rc}, CTX
    )
    assert len(notices) == 1
    n = notices[0]
    assert n.code == "fare_product_with_multiple_default_rider_categories"
    assert n.severity == Severity.ERROR
    assert n.fields["fare_product_id"] == "fare1"
    assert n.fields["csv_row_number1"] == 5
    assert n.fields["csv_row_number2"] == 8
    assert n.fields["rider_category_id1"] == "rider1"
    assert n.fields["rider_category_id2"] == "rider2"


# ---------------------------------------------------------------------------
# Test 16
# ---------------------------------------------------------------------------

def test_row_order_preserved_first_two_by_csv_row_number() -> None:
    """DataFrame presented in reverse CSV order → sort by csv_row_number selects earliest two."""
    rc = make_rider_categories([
        {"rider_category_id": "rider1", "is_default_fare_category": 1},
        {"rider_category_id": "rider2", "is_default_fare_category": 1},
        {"rider_category_id": "rider3", "is_default_fare_category": 1},
    ])
    # Presented in reverse order (row 3, 2, 1)
    fp = make_fare_products([
        {"fare_product_id": "fare1", "rider_category_id": "rider3", "csv_row_number": 3},
        {"fare_product_id": "fare1", "rider_category_id": "rider2", "csv_row_number": 2},
        {"fare_product_id": "fare1", "rider_category_id": "rider1", "csv_row_number": 1},
    ])
    notices = validate_fare_product_default_rider_categories(
        {"fare_products": fp, "rider_categories": rc}, CTX
    )
    assert len(notices) == 1
    n = notices[0]
    assert n.fields["csv_row_number1"] == 1
    assert n.fields["csv_row_number2"] == 2
    assert n.fields["rider_category_id1"] == "rider1"
    assert n.fields["rider_category_id2"] == "rider2"


# ---------------------------------------------------------------------------
# Test 17
# ---------------------------------------------------------------------------

def test_null_rider_category_id_in_fare_products_ignored() -> None:
    """Null rider_category_id in fare_products is silently ignored → notice for rows 2 and 3."""
    rc = make_rider_categories([
        {"rider_category_id": "rider1", "is_default_fare_category": 1},
        {"rider_category_id": "rider2", "is_default_fare_category": 1},
    ])
    fp = make_fare_products([
        {"fare_product_id": "fare1", "rider_category_id": None, "csv_row_number": 1},
        {"fare_product_id": "fare1", "rider_category_id": "rider1", "csv_row_number": 2},
        {"fare_product_id": "fare1", "rider_category_id": "rider2", "csv_row_number": 3},
    ])
    notices = validate_fare_product_default_rider_categories(
        {"fare_products": fp, "rider_categories": rc}, CTX
    )
    assert len(notices) == 1
    n = notices[0]
    assert n.fields["fare_product_id"] == "fare1"
    assert n.fields["csv_row_number1"] == 2
    assert n.fields["csv_row_number2"] == 3
    assert n.fields["rider_category_id1"] == "rider1"
    assert n.fields["rider_category_id2"] == "rider2"
