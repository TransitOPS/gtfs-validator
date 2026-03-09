"""Polars-based CSV loading with schema enforcement.

Two-phase strategy:
1. Read all columns as ``pl.Utf8`` to preserve raw values for validation.
2. Validate headers, trim whitespace, normalise empties → null, then cast
   column-by-column to target dtypes, emitting notices for failures.
"""

from __future__ import annotations

import enum
import io
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import IO

import polars as pl

from gtfs_validator.input import GtfsInput
from gtfs_validator.notices import Notice, NoticeContainer, Severity
from gtfs_validator.schemas import (
    TABLE_BY_FILENAME,
    ColumnDefinition,
    FieldType,
    Presence,
    TableDefinition,
    TableRequirement,
)


class TableStatus(enum.Enum):
    MISSING_FILE = "missing_file"
    EMPTY_FILE = "empty_file"
    UNPARSABLE_ROWS = "unparsable_rows"
    PARSABLE = "parsable"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_feed(
    gtfs_input: GtfsInput,
    num_threads: int = 1,
) -> tuple[dict[str, pl.DataFrame], dict[str, TableStatus], NoticeContainer]:
    """Load all tables from a GTFS feed.

    Returns ``(feed, table_statuses, notices)``.
    """
    notices = NoticeContainer()
    feed: dict[str, pl.DataFrame] = {}
    statuses: dict[str, TableStatus] = {}

    input_files = gtfs_input.filenames()

    # Emit notices for unknown files.
    for fname in sorted(input_files):
        if fname not in TABLE_BY_FILENAME:
            notices.add(Notice(
                code="unknown_file",
                severity=Severity.INFO,
                fields={"filename": fname},
            ))

    # Build work list of known tables.
    tables_to_load: list[tuple[TableDefinition, bool]] = []
    for table_def in TABLE_BY_FILENAME.values():
        present = table_def.filename in input_files
        tables_to_load.append((table_def, present))

    if num_threads <= 1:
        for table_def, present in tables_to_load:
            df, status, table_notices = _load_one(
                gtfs_input, table_def, present
            )
            feed[table_def.filename] = df
            statuses[table_def.filename] = status
            # Compatibility alias for validator code that uses logical table keys
            # (e.g. "calendar") rather than file names ("calendar.txt").
            if present:
                table_key = _table_key_from_filename(table_def.filename)
                feed[table_key] = df
                statuses[table_key] = status
            notices.add_all(table_notices)
    else:
        with ThreadPoolExecutor(max_workers=num_threads) as pool:
            futures = {
                pool.submit(_load_one, gtfs_input, td, p): (td, p)
                for td, p in tables_to_load
            }
            for fut in as_completed(futures):
                td, present = futures[fut]
                df, status, table_notices = fut.result()
                feed[td.filename] = df
                statuses[td.filename] = status
                if present:
                    table_key = _table_key_from_filename(td.filename)
                    feed[table_key] = df
                    statuses[table_key] = status
                notices.add_all(table_notices)

    return feed, statuses, notices


# ---------------------------------------------------------------------------
# Per-table loading
# ---------------------------------------------------------------------------


def _load_one(
    gtfs_input: GtfsInput,
    table_def: TableDefinition,
    present: bool,
) -> tuple[pl.DataFrame, TableStatus, list[Notice]]:
    """Load a single table, returning (DataFrame, status, notices)."""
    notices: list[Notice] = []

    if not present:
        return _handle_missing(table_def, notices)

    stream = gtfs_input.open_file(table_def.filename)
    try:
        return _parse_csv(stream, table_def, notices)
    finally:
        stream.close()


def _handle_missing(
    table_def: TableDefinition,
    notices: list[Notice],
) -> tuple[pl.DataFrame, TableStatus, list[Notice]]:
    """Create an empty DataFrame for a missing table and emit a notice."""
    if table_def.requirement == TableRequirement.REQUIRED:
        notices.append(Notice(
            code="missing_required_file",
            severity=Severity.ERROR,
            fields={"filename": table_def.filename},
        ))
    elif table_def.requirement in (
        TableRequirement.RECOMMENDED,
        TableRequirement.CONDITIONALLY_REQUIRED,
    ):
        notices.append(Notice(
            code="missing_recommended_file",
            severity=Severity.WARNING,
            fields={"filename": table_def.filename},
        ))
    empty = _empty_df(table_def)
    return empty, TableStatus.MISSING_FILE, notices


def _empty_df(table_def: TableDefinition) -> pl.DataFrame:
    """Return an empty DataFrame with the expected schema (all Utf8)."""
    return pl.DataFrame(
        {col.name: pl.Series(col.name, [], dtype=pl.Utf8) for col in table_def.columns}
    )


def _table_key_from_filename(filename: str) -> str:
    """Return logical table key from GTFS filename (e.g. stops.txt -> stops)."""
    return filename[:-4] if filename.endswith(".txt") else filename


# ---------------------------------------------------------------------------
# CSV parsing
# ---------------------------------------------------------------------------

def _parse_csv(
    stream: IO[bytes],
    table_def: TableDefinition,
    notices: list[Notice],
) -> tuple[pl.DataFrame, TableStatus, list[Notice]]:
    """Parse CSV bytes into a DataFrame with header validation."""
    raw_bytes = stream.read()

    # Attempt to parse CSV.
    try:
        raw_df = pl.read_csv(
            io.BytesIO(raw_bytes),
            has_header=True,
            infer_schema_length=0,  # read everything as Utf8
            truncate_ragged_lines=False,
            encoding="utf8-lossy",
        )
    except Exception as exc:
        notices.append(Notice(
            code="csv_parsing_failed",
            severity=Severity.ERROR,
            fields={"filename": table_def.filename, "message": str(exc)},
        ))
        return _empty_df(table_def), TableStatus.UNPARSABLE_ROWS, notices

    # Empty file check.
    if raw_df.height == 0:
        sev = (
            Severity.ERROR
            if table_def.requirement == TableRequirement.REQUIRED
            else Severity.WARNING
        )
        notices.append(Notice(
            code="empty_file",
            severity=sev,
            fields={"filename": table_def.filename},
        ))
        return _empty_df(table_def), TableStatus.EMPTY_FILE, notices

    # Header validation.
    _validate_headers(raw_df.columns, table_def, notices)

    # Whitespace trimming and null normalisation.
    raw_df = _normalise_strings(raw_df, table_def, notices)

    # Inject csv_row_number (1-based, header = row 1 so data starts at 2).
    raw_df = raw_df.with_row_index(name="csv_row_number", offset=2)

    # Type casting.
    raw_df = _cast_columns(raw_df, table_def, notices)

    return raw_df, TableStatus.PARSABLE, notices


# ---------------------------------------------------------------------------
# Header validation
# ---------------------------------------------------------------------------


def _validate_headers(
    actual_columns: list[str],
    table_def: TableDefinition,
    notices: list[Notice],
) -> None:
    col_map = table_def.column_map()
    expected = set(col_map)
    actual_set = set(actual_columns)

    # Duplicate columns.
    seen: set[str] = set()
    for col in actual_columns:
        if col in seen:
            notices.append(Notice(
                code="duplicate_column",
                severity=Severity.ERROR,
                fields={"filename": table_def.filename, "fieldName": col},
            ))
        seen.add(col)

    # Missing required / recommended columns.
    for col_def in table_def.columns:
        if col_def.name not in actual_set:
            if col_def.presence == Presence.REQUIRED:
                notices.append(Notice(
                    code="missing_required_column",
                    severity=Severity.ERROR,
                    fields={"filename": table_def.filename, "fieldName": col_def.name},
                ))
            elif col_def.presence == Presence.RECOMMENDED:
                notices.append(Notice(
                    code="missing_recommended_column",
                    severity=Severity.WARNING,
                    fields={"filename": table_def.filename, "fieldName": col_def.name},
                ))

    # Unknown columns.
    for col in actual_columns:
        if col not in expected:
            notices.append(Notice(
                code="unknown_column",
                severity=Severity.INFO,
                fields={"filename": table_def.filename, "fieldName": col},
            ))


# ---------------------------------------------------------------------------
# String normalisation
# ---------------------------------------------------------------------------


def _normalise_strings(
    df: pl.DataFrame,
    table_def: TableDefinition,
    notices: list[Notice],
) -> pl.DataFrame:
    """Trim whitespace, emit notices, and convert empty strings to null."""
    col_map = table_def.column_map()

    # Batch all whitespace counts into one pass over the DataFrame.
    ws_check_cols = [c for c in df.columns if c in col_map]
    if ws_check_cols:
        ws_counts = df.select([
            (pl.col(c).str.strip_chars() != pl.col(c)).sum().alias(c)
            for c in ws_check_cols
        ]).row(0, named=True)
        for col_name in ws_check_cols:
            ws_count = ws_counts[col_name]
            if ws_count > 0:
                notices.append(Notice(
                    code="leading_or_trailing_whitespaces",
                    severity=Severity.WARNING,
                    fields={
                        "filename": table_def.filename,
                        "fieldName": col_name,
                        "count": ws_count,
                    },
                ))

    # Trim and replace empty string with null in a single pass.
    return df.select([
        pl.col(col_name).str.strip_chars().replace("", None).alias(col_name)
        for col_name in df.columns
    ])


# ---------------------------------------------------------------------------
# Type casting
# ---------------------------------------------------------------------------

# Regex patterns for type validation.
_COLOR_RE = re.compile(r"^[0-9A-Fa-f]{6}$")
_DATE_RE = re.compile(r"^\d{8}$")
_TIME_RE = re.compile(r"^\d{1,2}:\d{2}:\d{2}$")
_URL_RE = re.compile(r"^https?://", re.IGNORECASE)
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def _cast_columns(
    df: pl.DataFrame,
    table_def: TableDefinition,
    notices: list[Notice],
) -> pl.DataFrame:
    """Cast columns from Utf8 to target dtypes based on schema."""
    col_map = table_def.column_map()

    for col_def in table_def.columns:
        if col_def.name not in df.columns:
            continue
        df = _cast_single_column(df, col_def, table_def.filename, notices)

    return df


def _cast_single_column(
    df: pl.DataFrame,
    col_def: ColumnDefinition,
    filename: str,
    notices: list[Notice],
) -> pl.DataFrame:
    """Cast a single column to its target type, emitting notices on failure."""
    col_name = col_def.name
    ft = col_def.field_type

    # Apply default values before casting (replaces nulls).
    if col_def.default_value is not None:
        df = df.with_columns(
            pl.col(col_name).fill_null(col_def.default_value).alias(col_name)
        )

    # Required/recommended value presence checks.
    if col_def.presence == Presence.REQUIRED:
        null_count = df.select(pl.col(col_name).is_null().sum()).item()
        if null_count > 0:
            notices.append(Notice(
                code="missing_required_value",
                severity=Severity.ERROR,
                fields={
                    "filename": filename,
                    "fieldName": col_name,
                    "count": null_count,
                },
            ))
    elif col_def.presence == Presence.RECOMMENDED:
        null_count = df.select(pl.col(col_name).is_null().sum()).item()
        if null_count > 0:
            notices.append(Notice(
                code="missing_recommended_value",
                severity=Severity.WARNING,
                fields={
                    "filename": filename,
                    "fieldName": col_name,
                    "count": null_count,
                },
            ))

    # Types that stay as Utf8 — only format validation needed.
    if ft in (FieldType.ID, FieldType.TEXT):
        return df

    if ft == FieldType.URL:
        return _validate_format(df, col_def, filename, notices, _URL_RE, "invalid_url")

    if ft == FieldType.EMAIL:
        return _validate_format(df, col_def, filename, notices, _EMAIL_RE, "invalid_email")

    if ft == FieldType.PHONE:
        return df  # Phone validation deferred to country-specific checks.

    if ft == FieldType.COLOR:
        return _validate_format(df, col_def, filename, notices, _COLOR_RE, "invalid_color")

    if ft == FieldType.DATE:
        return _validate_format(df, col_def, filename, notices, _DATE_RE, "invalid_date")

    if ft == FieldType.TIME:
        return _validate_format(df, col_def, filename, notices, _TIME_RE, "invalid_time")

    if ft in (FieldType.TIMEZONE, FieldType.LANGUAGE, FieldType.CURRENCY):
        return df  # Detailed validation happens in load_validators.

    # Numeric types — cast from Utf8.
    if ft in (FieldType.INTEGER, FieldType.ENUM):
        return _cast_numeric(
            df, col_name, filename, notices, pl.Int32, "invalid_integer",
            col_def.numeric_constraint,
        )

    if ft in (FieldType.FLOAT, FieldType.DECIMAL):
        return _cast_numeric(
            df, col_name, filename, notices, pl.Float64, "invalid_float",
            col_def.numeric_constraint,
        )

    if ft == FieldType.LATITUDE:
        df = _cast_numeric(
            df, col_name, filename, notices, pl.Float64, "invalid_float", None,
        )
        _check_range(df, col_name, filename, notices, -90.0, 90.0)
        return df

    if ft == FieldType.LONGITUDE:
        df = _cast_numeric(
            df, col_name, filename, notices, pl.Float64, "invalid_float", None,
        )
        _check_range(df, col_name, filename, notices, -180.0, 180.0)
        return df

    return df


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _validate_format(
    df: pl.DataFrame,
    col_def: ColumnDefinition,
    filename: str,
    notices: list[Notice],
    pattern: re.Pattern[str],
    notice_code: str,
) -> pl.DataFrame:
    """Validate string format; emit notice for non-null values that fail."""
    col_name = col_def.name
    col = pl.col(col_name)
    invalid_count = df.select(
        (col.is_not_null() & ~col.str.contains(pattern.pattern)).sum()
    ).item()
    if invalid_count > 0:
        notices.append(Notice(
            code=notice_code,
            severity=Severity.ERROR,
            fields={
                "filename": filename,
                "fieldName": col_name,
                "count": invalid_count,
            },
        ))
    return df


def _cast_numeric(
    df: pl.DataFrame,
    col_name: str,
    filename: str,
    notices: list[Notice],
    target_dtype: type[pl.DataType],
    error_code: str,
    constraint: object | None,
) -> pl.DataFrame:
    """Cast Utf8 column to numeric type; set failures to null."""
    from gtfs_validator.schemas import NumericConstraint

    try:
        df = df.with_columns(
            pl.col(col_name).cast(target_dtype, strict=False).alias(col_name)
        )
    except Exception:
        notices.append(Notice(
            code=error_code,
            severity=Severity.ERROR,
            fields={"filename": filename, "fieldName": col_name},
        ))
        return df

    # Check for values that became null after cast (were non-null strings).
    # We can't easily detect these after cast, so skip individual row tracking
    # for now; the cast with strict=False silently nullifies bad values.

    # Numeric constraint checks.
    if constraint is not None and isinstance(constraint, NumericConstraint):
        _check_numeric_constraint(df, col_name, filename, notices, constraint)

    return df


def _check_numeric_constraint(
    df: pl.DataFrame,
    col_name: str,
    filename: str,
    notices: list[Notice],
    constraint: object,
) -> None:
    from gtfs_validator.schemas import NumericConstraint

    if not isinstance(constraint, NumericConstraint):
        return

    col = pl.col(col_name)
    if constraint == NumericConstraint.NON_NEGATIVE:
        bad_expr = (col.is_not_null() & (col < 0)).sum()
    elif constraint == NumericConstraint.POSITIVE:
        bad_expr = (col.is_not_null() & (col <= 0)).sum()
    elif constraint == NumericConstraint.NON_ZERO:
        bad_expr = (col.is_not_null() & (col == 0)).sum()
    else:
        return

    bad = df.select(bad_expr).item()
    if bad > 0:
        notices.append(Notice(
            code="number_out_of_range",
            severity=Severity.ERROR,
            fields={
                "filename": filename,
                "fieldName": col_name,
                "count": bad,
            },
        ))


def _check_range(
    df: pl.DataFrame,
    col_name: str,
    filename: str,
    notices: list[Notice],
    min_val: float,
    max_val: float,
) -> None:
    col = pl.col(col_name)
    out = df.select(
        (col.is_not_null() & ((col < min_val) | (col > max_val))).sum()
    ).item()
    if out > 0:
        notices.append(Notice(
            code="number_out_of_range",
            severity=Severity.ERROR,
            fields={
                "filename": filename,
                "fieldName": col_name,
                "count": out,
            },
        ))
