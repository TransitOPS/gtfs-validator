"""StopTimesShapeDistTraveledPresenceValidator: validates that flex location rows
in stop_times.txt do not have shape_dist_traveled set."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from gtfs_validator.notices import Notice, Severity

if TYPE_CHECKING:
    from gtfs_validator.context import ValidationContext


def validate_stop_times_shape_dist_traveled_presence(
    feed: dict[str, pl.DataFrame],
    ctx: ValidationContext,
) -> list[Notice]:
    """Validate that flex location rows in stop_times do not have shape_dist_traveled.

    Emits ``forbidden_shape_dist_traveled`` for each row that:
    - has no stop_id (null), AND
    - has at least one flex location field (location_group_id or location_id)
      that is non-null, AND
    - has a non-null shape_dist_traveled.
    """
    # Guard 1: stop_times table absent
    if "stop_times" not in feed:
        return []

    st = feed["stop_times"]

    # Guard 2: shape_dist_traveled column absent
    if "shape_dist_traveled" not in st.columns:
        return []

    # Guard 3: neither flex location column present
    has_location_id_col = "location_id" in st.columns
    has_location_group_col = "location_group_id" in st.columns
    if not has_location_id_col and not has_location_group_col:
        return []

    # Guard 4: empty table
    if st.is_empty():
        return []

    # Build filter predicates

    # stop_id: if column absent, treat all rows as having no stop_id
    if "stop_id" in st.columns:
        has_stop_id_expr = pl.col("stop_id").is_not_null()
    else:
        has_stop_id_expr = pl.lit(False)

    # At least one flex location field non-null
    has_flex_expr: pl.Expr = pl.lit(False)
    if has_location_group_col:
        has_flex_expr = has_flex_expr | pl.col("location_group_id").is_not_null()
    if has_location_id_col:
        has_flex_expr = has_flex_expr | pl.col("location_id").is_not_null()

    # Violation = no stop_id AND has flex location AND shape_dist_traveled non-null
    violations = st.filter(
        ~has_stop_id_expr
        & has_flex_expr
        & pl.col("shape_dist_traveled").is_not_null()
    )

    notices: list[Notice] = []
    for row in violations.iter_rows(named=True):
        notices.append(Notice(
            code="forbidden_shape_dist_traveled",
            severity=Severity.ERROR,
            fields={
                "csv_row_number": row["csv_row_number"],
                "trip_id": row["trip_id"],
                "location_group_id": row.get("location_group_id") if has_location_group_col else None,
                "location_id": row.get("location_id") if has_location_id_col else None,
                "shape_dist_traveled": row["shape_dist_traveled"],
            },
        ))
    return notices
