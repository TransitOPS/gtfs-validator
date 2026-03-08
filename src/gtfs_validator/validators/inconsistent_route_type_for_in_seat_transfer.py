"""Validator: inconsistent route_type for in-seat transfers."""

from __future__ import annotations

import polars as pl

from gtfs_validator.context import ValidationContext
from gtfs_validator.notices import Notice, Severity


def validate_inconsistent_route_type_for_in_seat_transfer(
    feed: dict[str, pl.DataFrame],
    ctx: ValidationContext,
) -> list[Notice]:
    """Flag in-seat transfers (transfer_type=4) connecting routes with different route_type."""
    transfers = feed.get("transfers")
    routes = feed.get("routes")
    if transfers is None or routes is None:
        return []
    if transfers.is_empty() or routes.is_empty():
        return []
    for col in ("from_route_id", "to_route_id", "transfer_type"):
        if col not in transfers.columns:
            return []

    # Filter to in-seat transfers only (transfer_type == 4)
    in_seat = transfers.filter(pl.col("transfer_type") == "4")
    if in_seat.is_empty():
        return []

    route_lookup = routes.select("route_id", "route_type")

    # Join twice: once for from_route_id, once for to_route_id
    joined = (
        in_seat.select("csvRowNumber", "from_route_id", "to_route_id")
        .join(
            route_lookup.rename({"route_id": "from_route_id", "route_type": "from_route_type"}),
            on="from_route_id",
            how="inner",
        )
        .join(
            route_lookup.rename({"route_id": "to_route_id", "route_type": "to_route_type"}),
            on="to_route_id",
            how="inner",
        )
    )

    # Filter to rows where route types differ
    bad = joined.filter(pl.col("from_route_type") != pl.col("to_route_type"))

    # Emit one notice per mismatched transfer
    notices: list[Notice] = []
    for row in bad.iter_rows(named=True):
        notices.append(Notice(
            code="inconsistent_route_type_for_in_seat_transfer",
            severity=Severity.WARNING,
            fields={
                "csv_row_number": row["csvRowNumber"],
                "from_route_id": row["from_route_id"],
                "to_route_id": row["to_route_id"],
                "from_route_type": int(row["from_route_type"]),
                "to_route_type": int(row["to_route_type"]),
            },
        ))
    return notices
