"""Validator: fare_product_with_multiple_default_rider_categories."""

from __future__ import annotations

import polars as pl

from gtfs_validator.context import ValidationContext
from gtfs_validator.notices import Notice, Severity


def validate_fare_product_default_rider_categories(
    feed: dict[str, pl.DataFrame],
    ctx: ValidationContext,
) -> list[Notice]:
    """Emit an error for each fare_product_id linked to two or more distinct
    default rider categories (is_default_fare_category == 1).
    """
    # --- Skip guards ---
    if "rider_categories" not in feed or feed["rider_categories"].is_empty():
        return []
    if "fare_products" not in feed or feed["fare_products"].is_empty():
        return []

    df_fp = feed["fare_products"]
    df_rc = feed["rider_categories"].select(["rider_category_id", "is_default_fare_category"])

    # --- Step 1: join to bring is_default_fare_category into fare_products rows ---
    # Left join preserves all fare_products rows; non-matching rider_category_id
    # yields null for is_default_fare_category.
    joined = df_fp.join(df_rc, on="rider_category_id", how="left")

    # --- Step 2: filter to IS_DEFAULT rows only (is_default_fare_category == 1) ---
    # Null is_default_fare_category (no matching rider category) evaluates to
    # false in == comparison, so unresolved FK rows are silently dropped here.
    defaults = joined.filter(pl.col("is_default_fare_category") == 1)

    # --- Step 3: de-duplicate (fare_product_id, rider_category_id) pairs ---
    # Sort by csv_row_number first to ensure "first" means earliest CSV row.
    # unique(keep="first") retains the earliest row per pair after sort.
    # Re-sort after unique to restore global csv_row_number order.
    deduped = (
        defaults
        .sort("csv_row_number")
        .unique(subset=["fare_product_id", "rider_category_id"], keep="first")
        .sort("csv_row_number")
    )

    # --- Step 4: group by fare_product_id, emit notice if group size > 1 ---
    notices: list[Notice] = []
    for (fare_product_id,), group in deduped.group_by(["fare_product_id"]):
        group_sorted = group.sort("csv_row_number")
        if len(group_sorted) < 2:
            continue
        rows = list(group_sorted.iter_rows(named=True))
        r1 = rows[0]
        r2 = rows[1]
        notices.append(Notice(
            code="fare_product_with_multiple_default_rider_categories",
            severity=Severity.ERROR,
            fields={
                "fare_product_id": fare_product_id,
                "csv_row_number1": r1["csv_row_number"],
                "csv_row_number2": r2["csv_row_number"],
                "rider_category_id1": r1["rider_category_id"],
                "rider_category_id2": r2["rider_category_id"],
            },
        ))
    return notices
