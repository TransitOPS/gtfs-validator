"""Tests for ShapeUsageValidator."""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from gtfs_validator.context import ValidationContext
from gtfs_validator.notices import Severity
from gtfs_validator.validators.shape_usage import validate_shape_usage

CTX = ValidationContext(country_code="US", date_for_validation=date(2024, 1, 1))


def make_shapes(shape_ids: list[str], csv_row_numbers: list[int]) -> pl.DataFrame:
    """Minimal shapes DataFrame (two columns only)."""
    return pl.DataFrame(
        {
            "shape_id": shape_ids,
            "csv_row_number": csv_row_numbers,
        }
    )


def make_trips(shape_ids: list[str | None]) -> pl.DataFrame:
    """Minimal trips DataFrame."""
    return pl.DataFrame(
        {
            "shape_id": pl.Series(shape_ids, dtype=pl.String),
            "trip_id": [f"t{i}" for i in range(len(shape_ids))],
            "csv_row_number": list(range(1, len(shape_ids) + 1)),
        }
    )


def test_all_shapes_used_no_notice() -> None:
    """All shapes referenced by trips should produce no notices."""
    feed = {
        "shapes": make_shapes(["first shape id", "second shape id"], [1, 3]),
        "trips": make_trips(["first shape id", "second shape id"]),
    }
    notices = validate_shape_usage(feed, CTX)
    assert notices == []


def test_unused_shape_emits_notice() -> None:
    """An unreferenced shape should produce exactly one warning notice."""
    feed = {
        "shapes": make_shapes(["first shape id", "second shape id"], [1, 3]),
        "trips": make_trips(["first shape id"]),
    }
    notices = validate_shape_usage(feed, CTX)
    assert len(notices) == 1
    notice = notices[0]
    assert notice.code == "unused_shape"
    assert notice.severity == Severity.WARNING
    assert notice.fields["shape_id"] == "second shape id"
    assert notice.fields["csv_row_number"] == 3


def test_shapes_absent_no_notice() -> None:
    """Missing shapes table should produce no notices."""
    feed = {"trips": make_trips(["s1"])}
    notices = validate_shape_usage(feed, CTX)
    assert notices == []


def test_shapes_empty_no_notice() -> None:
    """Empty shapes table should produce no notices."""
    empty_shapes = pl.DataFrame(
        {"shape_id": pl.Series([], dtype=pl.String), "csv_row_number": pl.Series([], dtype=pl.Int64)}
    )
    feed = {"shapes": empty_shapes, "trips": make_trips(["s1"])}
    notices = validate_shape_usage(feed, CTX)
    assert notices == []


def test_trips_absent_all_shapes_unused() -> None:
    """When trips table is missing, all shapes are unused."""
    feed = {"shapes": make_shapes(["s1", "s2"], [1, 2])}
    notices = validate_shape_usage(feed, CTX)
    assert len(notices) == 2
    shape_ids = {n.fields["shape_id"] for n in notices}
    assert shape_ids == {"s1", "s2"}


def test_trips_empty_all_shapes_unused() -> None:
    """When trips table is empty, all shapes are unused."""
    empty_trips = pl.DataFrame(
        {
            "shape_id": pl.Series([], dtype=pl.String),
            "trip_id": pl.Series([], dtype=pl.String),
            "csv_row_number": pl.Series([], dtype=pl.Int64),
        }
    )
    feed = {"shapes": make_shapes(["s1"], [1]), "trips": empty_trips}
    notices = validate_shape_usage(feed, CTX)
    assert len(notices) == 1
    assert notices[0].fields["shape_id"] == "s1"


def test_shape_id_spans_multiple_rows_one_notice() -> None:
    """A shape_id spanning multiple shape-point rows should produce exactly one notice."""
    feed = {
        "shapes": make_shapes(["shape_a", "shape_a", "shape_a"], [1, 2, 3]),
        "trips": make_trips([]),
    }
    notices = validate_shape_usage(feed, CTX)
    assert len(notices) == 1
    assert notices[0].fields["shape_id"] == "shape_a"
    assert notices[0].fields["csv_row_number"] == 1


def test_csv_row_number_from_first_occurrence() -> None:
    """The reported csv_row_number must be from the first file-order occurrence, not minimum."""
    # First occurrence has row 5, second occurrence has row 3 (smaller integer but later in file)
    feed = {
        "shapes": make_shapes(["shape_a", "shape_a"], [5, 3]),
        "trips": make_trips([]),
    }
    notices = validate_shape_usage(feed, CTX)
    assert len(notices) == 1
    assert notices[0].fields["csv_row_number"] == 5


def test_null_shape_id_in_trips_does_not_contribute() -> None:
    """Null shape_id in trips must not count as a reference."""
    feed = {
        "shapes": make_shapes(["s1"], [1]),
        "trips": make_trips([None]),
    }
    notices = validate_shape_usage(feed, CTX)
    assert len(notices) == 1
    assert notices[0].fields["shape_id"] == "s1"


def test_empty_string_shape_id_in_trips_does_not_contribute() -> None:
    """Empty-string shape_id in trips must be excluded from the reference set."""
    feed = {
        "shapes": make_shapes(["s1"], [1]),
        "trips": make_trips([""]),
    }
    notices = validate_shape_usage(feed, CTX)
    assert len(notices) == 1
    assert notices[0].fields["shape_id"] == "s1"


def test_notice_fields_are_complete() -> None:
    """Notice fields must contain exactly the keys shape_id and csv_row_number."""
    feed = {
        "shapes": make_shapes(["s1"], [1]),
        "trips": make_trips([]),
    }
    notices = validate_shape_usage(feed, CTX)
    assert len(notices) == 1
    assert set(notices[0].fields.keys()) == {"shape_id", "csv_row_number"}


def test_ctx_unused() -> None:
    """A deliberately different ctx should produce the same result, confirming ctx has no effect."""
    feed = {
        "shapes": make_shapes(["s1"], [1]),
        "trips": make_trips([]),
    }
    ctx_alt = ValidationContext(country_code="GB", date_for_validation=date(2000, 6, 15))
    notices_normal = validate_shape_usage(feed, CTX)
    notices_alt = validate_shape_usage(feed, ctx_alt)
    assert len(notices_normal) == len(notices_alt)
    assert notices_normal[0].fields == notices_alt[0].fields
