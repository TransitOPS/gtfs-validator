"""StopTimesGeographyIdPresenceValidator: validates that each stop_times row has
exactly one geography identifier (stop_id, location_group_id, or location_id)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from gtfs_validator.notices import Notice, Severity

if TYPE_CHECKING:
    from gtfs_validator.context import ValidationContext


def _get_or_none(row: dict, col: str, present: bool) -> str | None:
    """Return the column value if the column is present, else None."""
    if not present:
        return None
    val = row.get(col)
    return val if val else None


def validate_stop_times_geography_id_presence(
    feed: dict[str, pl.DataFrame],
    ctx: ValidationContext,
) -> list[Notice]:
    """Validate that each stop_times row has exactly one geography identifier.

    Emits ``missing_required_field`` when none of stop_id, location_group_id,
    or location_id are present on a row.

    Emits ``forbidden_geography_id`` when more than one of the three geography
    ID fields is non-null and non-empty on the same row.
    """
    # Guard 1: stop_times table absent
    if "stop_times" not in feed or feed["stop_times"] is None:
        return []

    st = feed["stop_times"]

    # Guard 2: empty file
    if st.is_empty():
        return []

    # Guard 3: none of the three geography columns exist as file columns
    has_stop_id = "stop_id" in st.columns
    has_location_group = "location_group_id" in st.columns
    has_location_id = "location_id" in st.columns

    if not (has_stop_id or has_location_group or has_location_id):
        return []

    # Select only the columns we need
    select_cols = ["csv_row_number"]
    if has_stop_id:
        select_cols.append("stop_id")
    if has_location_group:
        select_cols.append("location_group_id")
    if has_location_id:
        select_cols.append("location_id")
    st = feed["stop_times"].select(select_cols)

    # Build presence expressions for each existing geography column
    flags: list[pl.Expr] = []
    if has_stop_id:
        flags.append(
            pl.col("stop_id").is_not_null() & (pl.col("stop_id") != "")
        )
    if has_location_group:
        flags.append(
            pl.col("location_group_id").is_not_null() & (pl.col("location_group_id") != "")
        )
    if has_location_id:
        flags.append(
            pl.col("location_id").is_not_null() & (pl.col("location_id") != "")
        )

    # Sum flags to get presence count per row
    presence_expr = sum(f.cast(pl.Int32) for f in flags)
    st = st.with_columns(presence_expr.alias("_presence"))

    notices: list[Notice] = []

    # Emit missing_required_field notices (presence == 0)
    missing = st.filter(pl.col("_presence") == 0)
    for row in missing.iter_rows(named=True):
        notices.append(Notice(
            code="missing_required_field",
            severity=Severity.ERROR,
            fields={
                "filename": "stop_times.txt",
                "csv_row_number": row["csv_row_number"],
                "field_name": "stop_id",
            },
        ))

    # Emit forbidden_geography_id notices (presence > 1)
    forbidden = st.filter(pl.col("_presence") > 1)
    for row in forbidden.iter_rows(named=True):
        notices.append(Notice(
            code="forbidden_geography_id",
            severity=Severity.ERROR,
            fields={
                "csv_row_number": row["csv_row_number"],
                "stop_id": _get_or_none(row, "stop_id", has_stop_id),
                "location_group_id": _get_or_none(row, "location_group_id", has_location_group),
                "location_id": _get_or_none(row, "location_id", has_location_id),
            },
        ))

    return notices
