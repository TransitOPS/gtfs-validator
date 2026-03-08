"""Validator: ParentStationValidator.

Checks:
- wrong_parent_location_type: a stop's parent_station has an unexpected location_type.
- unused_station: a station (location_type=1) has no STOP (location_type=0) children.
"""

from __future__ import annotations

import polars as pl

from gtfs_validator.context import ValidationContext
from gtfs_validator.notices import Notice, Severity

# Child location_type values that are checked for correct parent type
_CHECKED_CHILD_TYPES: frozenset[int] = frozenset({0, 2, 3, 4})

# Mapping from child location_type to required parent location_type
_EXPECTED_PARENT_TYPE: dict[int, int] = {
    0: 1,  # STOP requires STATION parent
    2: 1,  # ENTRANCE requires STATION parent
    3: 1,  # GENERIC_NODE requires STATION parent
    4: 0,  # BOARDING_AREA requires STOP parent
}


def validate_parent_station(
    feed: dict[str, pl.DataFrame],
    ctx: ValidationContext,
) -> list[Notice]:
    """Validate parent_station references in stops.txt."""
    stops = feed.get("stops")
    if stops is None or stops.is_empty():
        return []

    # Ensure required columns are present
    if "stop_id" not in stops.columns or "location_type" not in stops.columns:
        return []

    notices: list[Notice] = []

    # --- Check A: wrong_parent_location_type ---
    # Only proceed if parent_station column exists in the DataFrame
    if "parent_station" in stops.columns:

        # Build parent lookup table (all stops, selecting identifying columns)
        # Rename columns to avoid collision after the self-join
        parent_lookup = stops.select(["stop_id", "stop_name", "location_type", "csv_row_number"]).rename({
            "stop_id": "parent_station",
            "stop_name": "parentStopName",
            "location_type": "parentLocationType",
            "csv_row_number": "parentCsvRowNumber",
        })

        # Filter child rows: recognized types that have a non-null, non-empty parent_station
        children = stops.filter(
            pl.col("location_type").is_in(list(_CHECKED_CHILD_TYPES))
            & pl.col("parent_station").is_not_null()
            & (pl.col("parent_station") != "")
        )

        if not children.is_empty():
            # Inner join drops unresolvable parent references (FK violations handled elsewhere)
            joined = children.join(parent_lookup, on="parent_station", how="inner")

            # Add expected parent location_type column via when/then chain
            expected_expr = (
                pl.when(pl.col("location_type").is_in([0, 2, 3]))
                .then(pl.lit(1))
                .when(pl.col("location_type") == 4)
                .then(pl.lit(0))
                .otherwise(pl.lit(-1))  # should not occur; child pre-filter ensures recognized types
            )
            joined = joined.with_columns(expected_expr.alias("expectedLocationType"))

            # Filter mismatches: actual parent type != expected parent type
            mismatches = joined.filter(
                pl.col("parentLocationType") != pl.col("expectedLocationType")
            ).sort("csv_row_number")  # stable output order by child row number

            for row in mismatches.iter_rows(named=True):
                notices.append(Notice(
                    code="wrong_parent_location_type",
                    severity=Severity.ERROR,
                    fields={
                        "csvRowNumber": row["csv_row_number"],
                        "stopId": row["stop_id"],
                        "stopName": row.get("stop_name"),
                        "locationType": row["location_type"],
                        "parentCsvRowNumber": row["parentCsvRowNumber"],
                        "parentStation": row["parent_station"],
                        "parentStopName": row["parentStopName"],
                        "parentLocationType": row["parentLocationType"],
                        "expectedLocationType": row["expectedLocationType"],
                    },
                ))

    # --- Check B: unused_station ---
    # Stations: rows where location_type == 1
    stations = stops.filter(pl.col("location_type") == 1).select(
        ["stop_id", "stop_name", "csv_row_number"]
    )

    if not stations.is_empty():
        # Stop parents: STOP children (location_type == 0) with a non-null, non-empty parent_station
        if "parent_station" in stops.columns:
            stop_parents = stops.filter(
                (pl.col("location_type") == 0)
                & pl.col("parent_station").is_not_null()
                & (pl.col("parent_station") != "")
            ).select(pl.col("parent_station").alias("stop_id"))
        else:
            stop_parents = pl.DataFrame({"stop_id": []}, schema={"stop_id": pl.Utf8})

        # Anti-join: stations with no matching STOP child
        unused = stations.join(stop_parents, on="stop_id", how="anti").sort("csv_row_number")

        for row in unused.iter_rows(named=True):
            notices.append(Notice(
                code="unused_station",
                severity=Severity.INFO,
                fields={
                    "csvRowNumber": row["csv_row_number"],
                    "stopId": row["stop_id"],
                    "stopName": row.get("stop_name"),
                },
            ))

    return notices
