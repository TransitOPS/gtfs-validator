"""Schema-driven load-time checks.

These replace Java's annotation-generated validators.  They run after each
table is loaded and operate on the loaded DataFrames using Polars expressions.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from zoneinfo import available_timezones

import polars as pl

from gtfs_validator.loading import TableStatus
from gtfs_validator.notices import Notice, NoticeContainer, Severity
from gtfs_validator.schemas import (
    TABLE_BY_FILENAME,
    ColumnDefinition,
    FieldType,
    TableDefinition,
)

_VALID_TIMEZONES: set[str] = available_timezones()
_VALID_TIMEZONES_LIST: list[str] = list(_VALID_TIMEZONES)


def _run_per_table(table_def: TableDefinition, df: pl.DataFrame) -> NoticeContainer:
    """Run all per-table load validators for a single table."""
    notices = NoticeContainer()
    _check_primary_key(df, table_def, notices)
    _check_end_ranges(df, table_def, notices)
    _check_mixed_case(df, table_def, notices)
    _check_near_origin(df, table_def, notices)
    _check_timezone_values(df, table_def, notices)
    return notices


def run_load_validators(
    feed: dict[str, pl.DataFrame],
    table_statuses: dict[str, TableStatus],
    num_threads: int = 1,
) -> NoticeContainer:
    """Execute all schema-driven load-time checks.

    Per-table checks (PK, end-range, mixed-case, near-origin, timezone) run in
    parallel via *num_threads* when > 1.  Foreign-key checks are always
    sequential because they require the full feed to be present.
    """
    notices = NoticeContainer()

    tables_to_validate: list[tuple[TableDefinition, pl.DataFrame]] = []
    for table_def in TABLE_BY_FILENAME.values():
        status = table_statuses.get(table_def.filename, TableStatus.MISSING_FILE)
        if status not in (TableStatus.PARSABLE, TableStatus.EMPTY_FILE):
            continue
        df = feed.get(table_def.filename)
        if df is None or df.height == 0:
            continue
        tables_to_validate.append((table_def, df))

    if num_threads <= 1 or len(tables_to_validate) <= 1:
        for table_def, df in tables_to_validate:
            notices.merge(_run_per_table(table_def, df))
    else:
        with ThreadPoolExecutor(max_workers=num_threads) as pool:
            futures = {
                pool.submit(_run_per_table, td, df): None
                for td, df in tables_to_validate
            }
            for fut in as_completed(futures):
                notices.merge(fut.result())

    # Foreign key checks require cross-table access; always sequential.
    _check_all_foreign_keys(feed, table_statuses, notices)

    return notices


# ---------------------------------------------------------------------------
# Primary key uniqueness
# ---------------------------------------------------------------------------


def _check_primary_key(
    df: pl.DataFrame,
    table_def: TableDefinition,
    notices: NoticeContainer,
) -> None:
    pk_cols = table_def.primary_key_columns()
    if not pk_cols:
        return

    # Only check columns that exist in the DataFrame.
    pk_cols = [c for c in pk_cols if c in df.columns]
    if not pk_cols:
        return

    if len(pk_cols) == 1:
        col = pk_cols[0]
        dups = df.filter(pl.col(col).is_duplicated())
        if dups.height > 0:
            notices.add(Notice(
                code="duplicate_key",
                severity=Severity.ERROR,
                fields={
                    "filename": table_def.filename,
                    "fieldName": col,
                    "count": dups.height,
                },
            ))
    else:
        grouped = df.group_by(pk_cols).len()
        dups = grouped.filter(pl.col("len") > 1)
        if dups.height > 0:
            notices.add(Notice(
                code="duplicate_key",
                severity=Severity.ERROR,
                fields={
                    "filename": table_def.filename,
                    "fieldName": ",".join(pk_cols),
                    "count": dups.height,
                },
            ))


# ---------------------------------------------------------------------------
# Foreign key existence
# ---------------------------------------------------------------------------


def _check_all_foreign_keys(
    feed: dict[str, pl.DataFrame],
    table_statuses: dict[str, TableStatus],
    notices: NoticeContainer,
) -> None:
    for table_def in TABLE_BY_FILENAME.values():
        source_status = table_statuses.get(table_def.filename, TableStatus.MISSING_FILE)
        if source_status not in (TableStatus.PARSABLE, TableStatus.EMPTY_FILE):
            continue

        source_df = feed.get(table_def.filename)
        if source_df is None or source_df.height == 0:
            continue

        for col_def in table_def.columns:
            if col_def.foreign_key is None:
                continue
            if col_def.name not in source_df.columns:
                continue

            fk = col_def.foreign_key
            target_status = table_statuses.get(fk.table, TableStatus.MISSING_FILE)
            if target_status in (TableStatus.UNPARSABLE_ROWS, TableStatus.MISSING_FILE):
                continue

            target_df = feed.get(fk.table)
            if target_df is None:
                continue
            if fk.field not in target_df.columns:
                continue

            _check_foreign_key(
                source_df,
                table_def.filename,
                col_def.name,
                target_df,
                fk.table,
                fk.field,
                notices,
            )


def _check_foreign_key(
    source_df: pl.DataFrame,
    source_filename: str,
    source_field: str,
    target_df: pl.DataFrame,
    target_filename: str,
    target_field: str,
    notices: NoticeContainer,
) -> None:
    # Filter out null source values.
    source_non_null = source_df.filter(pl.col(source_field).is_not_null())
    if source_non_null.height == 0:
        return

    target_values = target_df.select(pl.col(target_field).unique())

    orphans = source_non_null.join(
        target_values,
        left_on=source_field,
        right_on=target_field,
        how="anti",
    )

    if orphans.height > 0:
        notices.add(Notice(
            code="foreign_key_violation",
            severity=Severity.ERROR,
            fields={
                "filename": source_filename,
                "fieldName": source_field,
                "foreignFilename": target_filename,
                "foreignFieldName": target_field,
                "count": orphans.height,
            },
        ))


# ---------------------------------------------------------------------------
# End-range ordering
# ---------------------------------------------------------------------------


def _check_end_ranges(
    df: pl.DataFrame,
    table_def: TableDefinition,
    notices: NoticeContainer,
) -> None:
    for col_def in table_def.columns:
        if col_def.end_range is None:
            continue
        er = col_def.end_range
        start_col = col_def.name
        end_col = er.field

        if start_col not in df.columns or end_col not in df.columns:
            continue

        both_present = df.filter(
            pl.col(start_col).is_not_null() & pl.col(end_col).is_not_null()
        )
        if both_present.height == 0:
            continue

        if er.allow_equal:
            bad = both_present.filter(pl.col(start_col) > pl.col(end_col))
        else:
            bad = both_present.filter(pl.col(start_col) >= pl.col(end_col))

        if bad.height > 0:
            notices.add(Notice(
                code="start_and_end_range_out_of_order",
                severity=Severity.ERROR,
                fields={
                    "filename": table_def.filename,
                    "startField": start_col,
                    "endField": end_col,
                    "count": bad.height,
                },
            ))


# ---------------------------------------------------------------------------
# Mixed case
# ---------------------------------------------------------------------------


def _check_mixed_case(
    df: pl.DataFrame,
    table_def: TableDefinition,
    notices: NoticeContainer,
) -> None:
    for col_def in table_def.columns:
        if not col_def.mixed_case:
            continue
        if col_def.name not in df.columns:
            continue

        non_null = df.filter(pl.col(col_def.name).is_not_null())
        if non_null.height == 0:
            continue

        all_upper_count = non_null.select(
            (pl.col(col_def.name).str.to_uppercase() == pl.col(col_def.name)).sum()
        ).item()

        if all_upper_count > 0:
            notices.add(Notice(
                code="mixed_case_recommended_field",
                severity=Severity.WARNING,
                fields={
                    "filename": table_def.filename,
                    "fieldName": col_def.name,
                    "count": all_upper_count,
                },
            ))


# ---------------------------------------------------------------------------
# Near-origin check
# ---------------------------------------------------------------------------


def _check_near_origin(
    df: pl.DataFrame,
    table_def: TableDefinition,
    notices: NoticeContainer,
) -> None:
    lat_cols = [c for c in table_def.columns if c.field_type == FieldType.LATITUDE]
    lon_cols = [c for c in table_def.columns if c.field_type == FieldType.LONGITUDE]

    if not lat_cols or not lon_cols:
        return

    lat_col = lat_cols[0].name
    lon_col = lon_cols[0].name

    if lat_col not in df.columns or lon_col not in df.columns:
        return

    near_origin = df.filter(
        pl.col(lat_col).is_not_null()
        & pl.col(lon_col).is_not_null()
        & (pl.col(lat_col).abs() < 1.0)
        & (pl.col(lon_col).abs() < 1.0)
    )

    if near_origin.height > 0:
        notices.add(Notice(
            code="point_near_origin",
            severity=Severity.WARNING,
            fields={
                "filename": table_def.filename,
                "count": near_origin.height,
            },
        ))


# ---------------------------------------------------------------------------
# Timezone validation
# ---------------------------------------------------------------------------


def _check_timezone_values(
    df: pl.DataFrame,
    table_def: TableDefinition,
    notices: NoticeContainer,
) -> None:
    for col_def in table_def.columns:
        if col_def.field_type != FieldType.TIMEZONE:
            continue
        if col_def.name not in df.columns:
            continue

        non_null = df.filter(pl.col(col_def.name).is_not_null())
        if non_null.height == 0:
            continue

        bad_count = non_null.select(
            (~pl.col(col_def.name).is_in(_VALID_TIMEZONES_LIST)).sum()
        ).item()
        if bad_count > 0:
            notices.add(Notice(
                code="invalid_timezone",
                severity=Severity.ERROR,
                fields={
                    "filename": table_def.filename,
                    "fieldName": col_def.name,
                    "count": bad_count,
                },
            ))
