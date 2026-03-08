"""Validator: TranslationFieldAndReferenceValidator.

Checks that each row in translations.txt:
- Uses the standard format (table_name present and not all-null)
- Has required fields (table_name, field_name, language)
- Uses either field_value OR record_id/record_sub_id (not both)
- References a known, present parent table
- Has matching entity in the parent table for the given record_id/record_sub_id
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from gtfs_validator.notices import Notice, Severity

if TYPE_CHECKING:
    from gtfs_validator.context import ValidationContext

# (record_id_column, record_sub_id_column) — None means the field is not used
TRANSLATION_KEYS: dict[str, tuple[str | None, str | None]] = {
    "agency":          ("agency_id",       None),
    "stops":           ("stop_id",         None),
    "routes":          ("route_id",        None),
    "trips":           ("trip_id",         None),
    "calendar":        ("service_id",      None),
    "fare_attributes": ("fare_id",         None),
    "levels":          ("level_id",        None),
    "pathways":        ("pathway_id",      None),
    "attributions":    ("attribution_id",  None),
    "stop_times":      ("trip_id",         "stop_sequence"),
    "calendar_dates":  ("service_id",      "date"),
    "shapes":          ("shape_id",        "shape_pt_sequence"),
    "frequencies":     ("trip_id",         "headway_secs"),
    "transfers":       ("from_stop_id",    "to_stop_id"),
    "feed_info":       (None,              None),
    "fare_rules":      (None,              None),
}

_INTEGER_SUB_ID_COLS: frozenset[str] = frozenset({
    "stop_sequence",
    "shape_pt_sequence",
    "headway_secs",
})


def _is_missing_or_unexpected(
    *,
    has_value: bool,
    expected_present: bool,
    field_name: str,
    field_value: str | None,
    csv_row_number: int,
    notices: list[Notice],
) -> bool:
    """Return True and append a notice if there is a mismatch; False if correct."""
    if has_value == expected_present:
        return False
    if has_value and not expected_present:
        notices.append(Notice(
            code="translation_unexpected_value",
            severity=Severity.ERROR,
            fields={
                "csvRowNumber": csv_row_number,
                "fieldName": field_name,
                "fieldValue": field_value or "",
            },
        ))
        return True
    # not has_value and expected_present
    notices.append(Notice(
        code="missing_required_field",
        severity=Severity.ERROR,
        fields={
            "filename": "translations.txt",
            "csvRowNumber": csv_row_number,
            "fieldName": field_name,
        },
    ))
    return True


def _entity_exists(
    *,
    entity_set: set[tuple[str | None, str | None]],
    record_id: str | None,
    record_sub_id: str | None,
    sub_col: str | None,
) -> bool:
    """Check whether an entity with the given keys exists in the pre-built lookup set."""
    if not entity_set:
        return False

    # 0-key table: always exists if the table is non-empty
    if record_id is None or record_id == "":
        return (None, None) in entity_set

    # 1-key table
    if sub_col is None:
        return (record_id, None) in entity_set

    # 2-key table: normalize sub-id for integer columns
    if sub_col in _INTEGER_SUB_ID_COLS:
        try:
            parsed_sub = str(int(record_sub_id or ""))
        except (ValueError, TypeError):
            return False   # unparseable → FK violation
    else:
        parsed_sub = record_sub_id

    return (record_id, parsed_sub) in entity_set


def _validate_parent_table(
    *,
    row_num: int,
    table_name: str,
    feed: dict[str, pl.DataFrame],
    lookup: dict[str, set[tuple[str | None, str | None]]],
    notices: list[Notice],
    check_reference: bool,
    record_id: str | None,
    record_sub_id: str | None,
) -> None:
    """Steps B and C: check parent table existence and reference integrity."""
    # Step B: resolve parent table
    if table_name not in TRANSLATION_KEYS or table_name not in feed:
        notices.append(Notice(
            code="translation_unknown_table_name",
            severity=Severity.WARNING,
            fields={
                "csvRowNumber": row_num,
                "tableName": table_name,
            },
        ))
        return

    if not check_reference:
        return

    # Step C: reference integrity
    id_col, sub_col = TRANSLATION_KEYS[table_name]
    record_id_expected = id_col is not None
    record_sub_id_expected = sub_col is not None

    has_record_id = record_id is not None and record_id != ""
    has_record_sub_id = record_sub_id is not None and record_sub_id != ""

    # Check record_id presence vs expectation
    mismatch = _is_missing_or_unexpected(
        has_value=has_record_id,
        expected_present=record_id_expected,
        field_name="record_id",
        field_value=record_id,
        csv_row_number=row_num,
        notices=notices,
    )
    if mismatch:
        return

    # Check record_sub_id presence vs expectation
    mismatch = _is_missing_or_unexpected(
        has_value=has_record_sub_id,
        expected_present=record_sub_id_expected,
        field_name="record_sub_id",
        field_value=record_sub_id,
        csv_row_number=row_num,
        notices=notices,
    )
    if mismatch:
        return

    # Entity lookup
    entity_set = lookup.get(table_name, set())
    found = _entity_exists(
        entity_set=entity_set,
        record_id=record_id,
        record_sub_id=record_sub_id,
        sub_col=sub_col,
    )
    if not found:
        notices.append(Notice(
            code="translation_foreign_key_violation",
            severity=Severity.ERROR,
            fields={
                "csvRowNumber": row_num,
                "tableName": table_name,
                "recordId": record_id or "",
                "recordSubId": record_sub_id or "",
            },
        ))


def validate_translation_field_and_reference(
    feed: dict[str, pl.DataFrame],
    ctx: ValidationContext,
) -> list[Notice]:
    """Validate translations.txt field values and record references."""
    if "translations" not in feed:
        return []
    translations = feed["translations"]
    if translations.is_empty():
        return []
    # Legacy format guard: standard format requires the "table_name" column.
    # If the column is absent or entirely null, treat as legacy format and skip.
    if "table_name" not in translations.columns:
        return []
    if translations["table_name"].is_null().all():
        return []

    # Phase 2: required-fields scan
    notices: list[Notice] = []
    any_missing = False

    for row in translations.iter_rows(named=True):
        row_num = row["csvRowNumber"]
        for field_name in ("table_name", "field_name", "language"):
            val = row.get(field_name)
            if val is None or val == "":
                notices.append(Notice(
                    code="missing_required_field",
                    severity=Severity.ERROR,
                    fields={
                        "filename": "translations.txt",
                        "csvRowNumber": row_num,
                        "fieldName": field_name,
                    },
                ))
                any_missing = True

    if any_missing:
        return notices

    # Phase 3 setup: pre-build entity lookup sets
    _lookup: dict[str, set[tuple[str | None, str | None]]] = {}

    for tname, (id_col, sub_col) in TRANSLATION_KEYS.items():
        df = feed.get(tname)
        if df is None or df.is_empty():
            continue
        keys: set[tuple[str | None, str | None]] = set()
        if id_col is None:
            # 0-key table: any non-empty table counts as having the single entity
            keys.add((None, None))
        elif sub_col is None:
            # 1-key table
            if id_col in df.columns:
                for (val,) in df.select([id_col]).iter_rows():
                    if val is not None:
                        keys.add((val, None))
        else:
            # 2-key table
            cols_present = {id_col, sub_col}.issubset(df.columns)
            if cols_present:
                for (id_val, sub_val) in df.select([id_col, sub_col]).iter_rows():
                    if id_val is not None:
                        # Store sub_val as string for consistent comparison
                        keys.add((str(id_val), str(sub_val) if sub_val is not None else None))
        _lookup[tname] = keys

    # Phase 3: per-row validation loop
    for row in translations.iter_rows(named=True):
        row_num = row["csvRowNumber"]
        table_name = row.get("table_name") or ""
        field_value = row.get("field_value")       # None if absent or null
        record_id = row.get("record_id")           # None if absent or null
        record_sub_id = row.get("record_sub_id")   # None if absent or null

        has_field_value = field_value is not None and field_value != ""
        has_record_id = record_id is not None and record_id != ""
        has_record_sub_id = record_sub_id is not None and record_sub_id != ""

        # Step A: field_value mutual exclusion
        if has_field_value:
            if has_record_id:
                notices.append(Notice(
                    code="translation_unexpected_value",
                    severity=Severity.ERROR,
                    fields={
                        "csvRowNumber": row_num,
                        "fieldName": "record_id",
                        "fieldValue": record_id,
                    },
                ))
            if has_record_sub_id:
                notices.append(Notice(
                    code="translation_unexpected_value",
                    severity=Severity.ERROR,
                    fields={
                        "csvRowNumber": row_num,
                        "fieldName": "record_sub_id",
                        "fieldValue": record_sub_id,
                    },
                ))
            # Step B still runs (parent table unknown check) but Step C is skipped
            _validate_parent_table(
                row_num=row_num,
                table_name=table_name,
                feed=feed,
                lookup=_lookup,
                notices=notices,
                check_reference=False,
                record_id=record_id,
                record_sub_id=record_sub_id,
            )
        else:
            _validate_parent_table(
                row_num=row_num,
                table_name=table_name,
                feed=feed,
                lookup=_lookup,
                notices=notices,
                check_reference=True,
                record_id=record_id,
                record_sub_id=record_sub_id,
            )

    return notices
