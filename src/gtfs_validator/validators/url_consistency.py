"""Validator: UrlConsistencyValidator.

Checks that route_url, stop_url, and agency_url are not shared across
different entity types, as this typically indicates a data entry error.
"""

from __future__ import annotations

import polars as pl

from gtfs_validator.context import ValidationContext
from gtfs_validator.notices import Notice, Severity


def _keyed(
    df: pl.DataFrame | None,
    url_col: str,
    keep_cols: list[str],
) -> pl.DataFrame | None:
    """Return a narrow DataFrame keyed by lowercase URL, or None if not applicable."""
    if df is None or df.is_empty() or url_col not in df.columns:
        return None
    # If the column dtype is Null (all values are null), str operations will fail.
    # Cast to String first so filtering and lowercasing work correctly.
    col_dtype = df.schema[url_col]
    if col_dtype == pl.Null:
        return None
    return (
        df.filter(pl.col(url_col).is_not_null())
        .with_columns(pl.col(url_col).cast(pl.String).str.to_lowercase().alias("_url_key"))
        .select(["_url_key"] + keep_cols)
    )


def validate_url_consistency(
    feed: dict[str, pl.DataFrame],
    ctx: ValidationContext,
) -> list[Notice]:
    """Emit warnings when route_url, stop_url, or agency_url values overlap.

    All three input tables (agency, routes, stops) are optional. The validator
    handles every absent/empty combination internally without raising errors.
    URL comparison is case-insensitive.
    """
    notices: list[Notice] = []

    agency = feed.get("agency")
    routes = feed.get("routes")
    stops = feed.get("stops")

    # Build keyed frames (only when the table and URL column are present and non-empty)
    agency_keyed = _keyed(agency, "agency_url", ["agency_name", "csv_row_number"])
    routes_keyed = _keyed(routes, "route_url", ["route_id", "route_url", "csv_row_number"])
    stops_keyed = _keyed(stops, "stop_url", ["stop_id", "stop_url", "csv_row_number"])

    # Check 1: route_url vs agency_url
    # After join: left cols = [_url_key, route_id, route_url, csv_row_number]
    #             right cols = [agency_name, csv_row_number_right]
    # (agency_name has no collision so keeps its name; csv_row_number collides -> _right suffix)
    if agency_keyed is not None and routes_keyed is not None:
        matches = routes_keyed.join(agency_keyed, on="_url_key", how="inner")
        for row in matches.iter_rows(named=True):
            notices.append(
                Notice(
                    code="same_route_and_agency_url",
                    severity=Severity.WARNING,
                    fields={
                        "route_csv_row_number": row["csv_row_number"],
                        "route_id": row["route_id"],
                        "agency_name": row["agency_name"],
                        "route_url": row["route_url"],
                        "agency_csv_row_number": row["csv_row_number_right"],
                    },
                )
            )

    # Check 2: stop_url vs agency_url
    # After join: left cols = [_url_key, stop_id, stop_url, csv_row_number]
    #             right cols = [agency_name, csv_row_number_right]
    if agency_keyed is not None and stops_keyed is not None:
        matches = stops_keyed.join(agency_keyed, on="_url_key", how="inner")
        for row in matches.iter_rows(named=True):
            notices.append(
                Notice(
                    code="same_stop_and_agency_url",
                    severity=Severity.WARNING,
                    fields={
                        "stop_csv_row_number": row["csv_row_number"],
                        "stop_id": row["stop_id"],
                        "agency_name": row["agency_name"],
                        "stop_url": row["stop_url"],
                        "agency_csv_row_number": row["csv_row_number_right"],
                    },
                )
            )

    # Check 3: stop_url vs route_url
    # After join: left cols = [_url_key, stop_id, stop_url, csv_row_number]
    #             right cols = [route_id, route_url_right, csv_row_number_right]
    # (stop_url and route_url don't collide with each other, but route_url from right
    #  has no name clash with stop_url; however route_url is on the right frame so
    #  it keeps its name; csv_row_number collides -> _right suffix)
    if routes_keyed is not None and stops_keyed is not None:
        matches = stops_keyed.join(routes_keyed, on="_url_key", how="inner")
        for row in matches.iter_rows(named=True):
            notices.append(
                Notice(
                    code="same_stop_and_route_url",
                    severity=Severity.WARNING,
                    fields={
                        "stop_csv_row_number": row["csv_row_number"],
                        "stop_id": row["stop_id"],
                        "stop_url": row["stop_url"],
                        "route_id": row["route_id"],
                        "route_csv_row_number": row["csv_row_number_right"],
                    },
                )
            )

    return notices