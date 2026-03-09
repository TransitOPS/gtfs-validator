"""Tests for SingleShapePointValidator."""

from datetime import date

import polars as pl
import pytest

from gtfs_validator.context import ValidationContext
from gtfs_validator.notices import Severity
from gtfs_validator.validators.single_shape_point import validate_single_shape_point

CTX = ValidationContext(country_code="US", date_for_validation=date(2024, 1, 1))


def make_shapes(
    shape_ids: list[str],
    csv_row_numbers: list[int],
    sequences: list[int] | None = None,
) -> pl.DataFrame:
    """Minimal shapes DataFrame with the columns the validator reads."""
    n = len(shape_ids)
    return pl.DataFrame({
        "shape_id": shape_ids,
        "csv_row_number": csv_row_numbers,
        "shape_pt_sequence": sequences if sequences is not None else list(range(1, n + 1)),
    })


def test_shape_with_multiple_points_no_notice() -> None:
    """A shape with more than one point should not generate a notice."""
    shapes = make_shapes(
        shape_ids=["first shape id", "first shape id"],
        csv_row_numbers=[1, 2],
        sequences=[1, 2],
    )
    notices = validate_single_shape_point({"shapes": shapes}, CTX)
    assert notices == []


def test_single_point_shape_emits_notice() -> None:
    """A shape with exactly one point should generate a single_shape_point notice."""
    shapes = make_shapes(
        shape_ids=["first shape id", "first shape id", "second shape id"],
        csv_row_numbers=[1, 2, 3],
        sequences=[1, 2, 1],
    )
    notices = validate_single_shape_point({"shapes": shapes}, CTX)
    assert len(notices) == 1
    notice = notices[0]
    assert notice.code == "single_shape_point"
    assert notice.severity == Severity.WARNING
    assert notice.fields["shape_id"] == "second shape id"
    assert notice.fields["csv_row_number"] == 3


def test_shapes_absent_no_notice() -> None:
    """When 'shapes' is not in the feed, no notices should be emitted."""
    notices = validate_single_shape_point({}, CTX)
    assert notices == []


def test_shapes_empty_no_notice() -> None:
    """When shapes.txt is present but empty, no notices should be emitted."""
    shapes = pl.DataFrame({
        "shape_id": pl.Series([], dtype=pl.Utf8),
        "csv_row_number": pl.Series([], dtype=pl.Int64),
        "shape_pt_sequence": pl.Series([], dtype=pl.Int64),
    })
    notices = validate_single_shape_point({"shapes": shapes}, CTX)
    assert notices == []


def test_all_shapes_single_point_all_emit() -> None:
    """Every single-point shape should emit one notice each."""
    shapes = make_shapes(
        shape_ids=["s1", "s2", "s3"],
        csv_row_numbers=[1, 2, 3],
        sequences=[1, 1, 1],
    )
    notices = validate_single_shape_point({"shapes": shapes}, CTX)
    assert len(notices) == 3
    for notice in notices:
        assert notice.code == "single_shape_point"
        assert notice.severity == Severity.WARNING
    assert {n.fields["shape_id"] for n in notices} == {"s1", "s2", "s3"}


def test_all_shapes_multiple_points_no_notice() -> None:
    """All shapes with multiple points should produce no notices."""
    shapes = make_shapes(
        shape_ids=["s1", "s1", "s2", "s2"],
        csv_row_numbers=[1, 2, 3, 4],
        sequences=[1, 2, 1, 2],
    )
    notices = validate_single_shape_point({"shapes": shapes}, CTX)
    assert notices == []


def test_csv_row_number_is_sole_occurrence_row() -> None:
    """The csv_row_number field should match the row number of the only point row."""
    shapes = make_shapes(
        shape_ids=["s1"],
        csv_row_numbers=[42],
        sequences=[1],
    )
    notices = validate_single_shape_point({"shapes": shapes}, CTX)
    assert len(notices) == 1
    assert notices[0].fields["csv_row_number"] == 42


def test_notice_fields_are_complete() -> None:
    """Notice fields must contain exactly 'shape_id' and 'csv_row_number'."""
    shapes = make_shapes(
        shape_ids=["s1"],
        csv_row_numbers=[1],
        sequences=[1],
    )
    notices = validate_single_shape_point({"shapes": shapes}, CTX)
    assert len(notices) == 1
    assert set(notices[0].fields.keys()) == {"shape_id", "csv_row_number"}


def test_ctx_unused() -> None:
    """The ctx parameter has no effect on the validator output."""
    shapes = make_shapes(
        shape_ids=["s1"],
        csv_row_numbers=[1],
        sequences=[1],
    )
    ctx_wrong = ValidationContext(country_code="XX", date_for_validation=date(1900, 1, 1))
    notices_normal = validate_single_shape_point({"shapes": shapes}, CTX)
    notices_wrong = validate_single_shape_point({"shapes": shapes}, ctx_wrong)
    assert len(notices_normal) == len(notices_wrong) == 1
    assert notices_normal[0].fields == notices_wrong[0].fields


def test_mixed_single_and_multi_point_shapes() -> None:
    """Only single-point shapes should be flagged in a mixed feed."""
    shapes = make_shapes(
        shape_ids=["multi", "multi", "multi", "solo_a", "solo_b"],
        csv_row_numbers=[1, 2, 3, 4, 5],
        sequences=[1, 2, 3, 1, 1],
    )
    notices = validate_single_shape_point({"shapes": shapes}, CTX)
    assert len(notices) == 2
    flagged_ids = {n.fields["shape_id"] for n in notices}
    assert flagged_ids == {"solo_a", "solo_b"}
    assert "multi" not in flagged_ids
