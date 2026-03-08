"""Validator: fare_leg_join_rules.txt FK and co-presence checks."""

from __future__ import annotations

import polars as pl

from gtfs_validator.context import ValidationContext
from gtfs_validator.notices import Notice, Severity


def validate_fare_leg_join_rule(
    feed: dict[str, pl.DataFrame],
    ctx: ValidationContext,
) -> list[Notice]:
    """Validate fare_leg_join_rules.txt network ID FKs and stop ID co-presence."""
    if "fare_leg_join_rules" not in feed or feed["fare_leg_join_rules"].is_empty():
        return []

    df = feed["fare_leg_join_rules"].with_row_index("_row_idx")

    # Build valid network ID lookup set (OR across networks.txt and routes.txt)
    valid_network_ids: set[str] = set()
    if "networks" in feed:
        col = feed["networks"]["network_id"].drop_nulls()
        valid_network_ids |= set(col.to_list())
    if "routes" in feed and "network_id" in feed["routes"].columns:
        routes_col = feed["routes"]["network_id"].drop_nulls().cast(pl.Utf8)
        valid_network_ids |= set(
            routes_col.filter(routes_col != "").to_list()
        )

    notices: list[Notice] = []

    # Check from_network_id and to_network_id FK
    for field_name in ("from_network_id", "to_network_id"):
        if field_name not in df.columns:
            continue

        violations = df.filter(
            pl.col(field_name).is_not_null()
            & (pl.col(field_name) != "")
            & ~pl.col(field_name).is_in(list(valid_network_ids))
        )

        for row in violations.iter_rows(named=True):
            notices.append(
                Notice(
                    code="foreign_key_violation",
                    severity=Severity.ERROR,
                    fields={
                        "child_filename": "fare_leg_join_rules.txt",
                        "child_field_name": field_name,
                        "parent_filename": "routes.txt or networks.txt",
                        "parent_field_name": "network_id",
                        "field_value": row[field_name],
                        "csv_row_number": row["_row_idx"],
                    },
                )
            )

    # Check from_stop_id / to_stop_id co-presence
    has_from_col = "from_stop_id" in df.columns
    has_to_col = "to_stop_id" in df.columns

    def is_present(col_name: str) -> pl.Expr:
        return pl.col(col_name).is_not_null() & (pl.col(col_name) != "")

    def is_absent(col_name: str) -> pl.Expr:
        return pl.col(col_name).is_null() | (pl.col(col_name) == "")

    if has_from_col and has_to_col:
        missing_to = df.filter(is_present("from_stop_id") & is_absent("to_stop_id"))
        for row in missing_to.iter_rows(named=True):
            notices.append(
                Notice(
                    code="missing_required_field",
                    severity=Severity.ERROR,
                    fields={
                        "filename": "fare_leg_join_rules.txt",
                        "csv_row_number": row["_row_idx"],
                        "field_name": "to_stop_id",
                    },
                )
            )

        missing_from = df.filter(is_present("to_stop_id") & is_absent("from_stop_id"))
        for row in missing_from.iter_rows(named=True):
            notices.append(
                Notice(
                    code="missing_required_field",
                    severity=Severity.ERROR,
                    fields={
                        "filename": "fare_leg_join_rules.txt",
                        "csv_row_number": row["_row_idx"],
                        "field_name": "from_stop_id",
                    },
                )
            )
    elif has_from_col and not has_to_col:
        missing_to = df.filter(is_present("from_stop_id"))
        for row in missing_to.iter_rows(named=True):
            notices.append(
                Notice(
                    code="missing_required_field",
                    severity=Severity.ERROR,
                    fields={
                        "filename": "fare_leg_join_rules.txt",
                        "csv_row_number": row["_row_idx"],
                        "field_name": "to_stop_id",
                    },
                )
            )
    elif has_to_col and not has_from_col:
        missing_from = df.filter(is_present("to_stop_id"))
        for row in missing_from.iter_rows(named=True):
            notices.append(
                Notice(
                    code="missing_required_field",
                    severity=Severity.ERROR,
                    fields={
                        "filename": "fare_leg_join_rules.txt",
                        "csv_row_number": row["_row_idx"],
                        "field_name": "from_stop_id",
                    },
                )
            )

    return notices
