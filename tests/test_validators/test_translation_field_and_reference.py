"""Tests for TranslationFieldAndReferenceValidator."""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from gtfs_validator.context import ValidationContext
from gtfs_validator.notices import Notice, Severity
from gtfs_validator.validators.translation_field_and_reference import (
    validate_translation_field_and_reference,
)

CTX = ValidationContext(country_code="US", date_for_validation=date(2024, 1, 1))

# Standard translation columns
_TRANS_COLS = [
    "table_name",
    "field_name",
    "language",
    "translation",
    "record_id",
    "record_sub_id",
    "field_value",
    "csvRowNumber",
]


def make_feed(**overrides: pl.DataFrame) -> dict[str, pl.DataFrame]:
    """Return a minimal feed seeded with standard fixture data, with any overrides applied."""
    feed: dict[str, pl.DataFrame] = {
        "agency": pl.DataFrame({
            "agency_id": ["agency0"],
            "csvRowNumber": [2],
        }),
        "stop_times": pl.DataFrame({
            "trip_id": ["trip0"],
            "stop_sequence": [0],
            "csvRowNumber": [2],
        }),
        "feed_info": pl.DataFrame({
            "feed_lang": ["en-CA"],
            "csvRowNumber": [2],
        }),
    }
    feed.update(overrides)
    return feed


def make_translations(**kwargs: object) -> pl.DataFrame:
    """Build a one-row translations DataFrame with all standard columns.

    Keyword args set field values; anything not provided defaults to None.
    csvRowNumber defaults to 2.
    """
    row: dict[str, object] = {col: None for col in _TRANS_COLS}
    row["csvRowNumber"] = 2
    row.update(kwargs)
    return pl.DataFrame({col: [row[col]] for col in _TRANS_COLS})


# ---------------------------------------------------------------------------
# Guard / legacy format tests
# ---------------------------------------------------------------------------


def test_no_translations_table() -> None:
    """Feed without translations key yields no notices."""
    feed = make_feed()
    result = validate_translation_field_and_reference(feed, CTX)
    assert result == []


def test_empty_translations_table() -> None:
    """Empty translations DataFrame yields no notices."""
    feed = make_feed(translations=pl.DataFrame({col: [] for col in _TRANS_COLS}))
    result = validate_translation_field_and_reference(feed, CTX)
    assert result == []


def test_legacy_format_yields_no_notice() -> None:
    """Translations without table_name column treated as legacy — no notices."""
    translations = pl.DataFrame({
        "record_id": [None],
        "translation": [None],
    })
    feed = make_feed(translations=translations)
    result = validate_translation_field_and_reference(feed, CTX)
    assert result == []


def test_all_null_table_name_column() -> None:
    """Translations with all-null table_name treated as legacy — no notices."""
    translations = pl.DataFrame({col: [None] for col in _TRANS_COLS})
    feed = make_feed(translations=translations)
    result = validate_translation_field_and_reference(feed, CTX)
    assert result == []


# ---------------------------------------------------------------------------
# Phase 2: required fields scan
# ---------------------------------------------------------------------------


def test_missing_required_standard_fields_yields_notices() -> None:
    """Row missing table_name, field_name, and language yields three notices."""
    translations = pl.DataFrame({col: [None] for col in _TRANS_COLS} | {"table_name": ["agency"], "csvRowNumber": [2]})
    # Override to all None (including table_name back to None for this test)
    translations = pl.DataFrame({col: [None] for col in _TRANS_COLS} | {"csvRowNumber": [2]})
    # But we need table_name to be non-null somewhere for guard to pass — use a second row
    # Actually the spec says: standard format detected when table_name col is present AND not all-null
    # So we need at least one non-null table_name to enter the validator,
    # but the row we test has it null.
    translations = pl.DataFrame({
        col: [None] for col in _TRANS_COLS
    } | {"csvRowNumber": [2]})
    # All-null table_name would trigger the legacy guard... need at least one non-null.
    # The test mirrors Java missingRequiredStandardFields_yieldsNotice which has table_name present
    # but null ON THIS ROW. The guard checks if ALL are null. So with 1 row all null,
    # the guard fires and we get []. We need 2 rows: 1 with real table_name, 1 missing it.
    # But the spec says "one row where table_name, field_name, language are all None" and
    # expects 3 notices... Let's re-read: guard says translations["table_name"].is_null().all()
    # With one row and table_name=None, all() is True → guard fires → return [].
    # So the test must have table_name present (not null) but still missing field_name/language.
    # Wait — re-read spec test case: "one row where table_name, field_name, and language are all None"
    # Expected: exactly three missing_required_field notices.
    # This conflicts with the guard. Unless the guard only checks all-null, and the phase-2 loop
    # also checks for empty string. Let me re-check: guard is translations["table_name"].is_null().all()
    # If table_name is None (null) on the only row, is_null().all() = True → return [].
    # That means the Java test must have table_name as empty string "", not null.
    translations = pl.DataFrame({
        "table_name": [""],
        "field_name": [None],
        "language": [None],
        "translation": [None],
        "record_id": [None],
        "record_sub_id": [None],
        "field_value": [None],
        "csvRowNumber": [2],
    })
    feed = make_feed(translations=translations)
    result = validate_translation_field_and_reference(feed, CTX)
    codes = [(n.code, n.fields["fieldName"]) for n in result]
    assert len(result) == 3
    assert ("missing_required_field", "table_name") in codes
    assert ("missing_required_field", "field_name") in codes
    assert ("missing_required_field", "language") in codes
    # Early return: no FK or unknown-table notices
    assert all(n.code == "missing_required_field" for n in result)


def test_missing_required_fields_multiple_rows() -> None:
    """Two rows both missing field_name and language; first also missing table_name."""
    translations = pl.DataFrame({
        "table_name": ["", "agency"],
        "field_name": [None, None],
        "language": [None, None],
        "translation": [None, None],
        "record_id": [None, None],
        "record_sub_id": [None, None],
        "field_value": [None, None],
        "csvRowNumber": [2, 3],
    })
    feed = make_feed(translations=translations)
    result = validate_translation_field_and_reference(feed, CTX)
    # Row 1: 3 notices (table_name, field_name, language)
    # Row 2: 2 notices (field_name, language)
    assert len(result) == 5
    assert all(n.code == "missing_required_field" for n in result)


# ---------------------------------------------------------------------------
# Phase 3: per-row validation
# ---------------------------------------------------------------------------


def test_wrong_table_name_yields_notice() -> None:
    """Unknown table_name yields translation_unknown_table_name warning."""
    translations = make_translations(table_name="wrong", field_name="any", language="en")
    feed = make_feed(translations=translations)
    result = validate_translation_field_and_reference(feed, CTX)
    assert len(result) == 1
    assert result[0].code == "translation_unknown_table_name"
    assert result[0].severity == Severity.WARNING
    assert result[0].fields["tableName"] == "wrong"


def test_field_value_defined_yields_no_notice() -> None:
    """field_value set with no record_id/record_sub_id and known table → no notices."""
    translations = make_translations(
        table_name="agency",
        field_name="any",
        field_value="any",
        language="en",
    )
    feed = make_feed(translations=translations)
    result = validate_translation_field_and_reference(feed, CTX)
    assert result == []


def test_record_id_and_field_value_defined_yields_notice() -> None:
    """record_id and field_value both set → unexpected_value for record_id."""
    translations = make_translations(
        table_name="agency",
        field_name="any",
        record_id="id",
        field_value="any",
        language="en",
    )
    feed = make_feed(translations=translations)
    result = validate_translation_field_and_reference(feed, CTX)
    assert len(result) == 1
    assert result[0].code == "translation_unexpected_value"
    assert result[0].fields["fieldName"] == "record_id"
    assert result[0].fields["fieldValue"] == "id"


def test_record_sub_id_and_field_value_defined_yields_notice() -> None:
    """record_sub_id and field_value both set → unexpected_value for record_sub_id."""
    translations = make_translations(
        table_name="agency",
        field_name="any",
        record_sub_id="sub_id",
        field_value="any",
        language="en",
    )
    feed = make_feed(translations=translations)
    result = validate_translation_field_and_reference(feed, CTX)
    assert len(result) == 1
    assert result[0].code == "translation_unexpected_value"
    assert result[0].fields["fieldName"] == "record_sub_id"
    assert result[0].fields["fieldValue"] == "sub_id"


def test_both_record_id_and_record_sub_id_with_field_value() -> None:
    """field_value plus record_id and record_sub_id → two unexpected_value notices."""
    translations = make_translations(
        table_name="agency",
        field_name="any",
        record_id="r",
        record_sub_id="s",
        field_value="v",
        language="en",
    )
    feed = make_feed(translations=translations)
    result = validate_translation_field_and_reference(feed, CTX)
    assert len(result) == 2
    field_names = {n.fields["fieldName"] for n in result}
    assert field_names == {"record_id", "record_sub_id"}
    assert all(n.code == "translation_unexpected_value" for n in result)


def test_no_record_id_for_single_key_table_yields_notice() -> None:
    """Single-key table (agency) with no record_id → missing_required_field."""
    translations = make_translations(
        table_name="agency",
        field_name="any",
        language="en",
    )
    feed = make_feed(translations=translations)
    result = validate_translation_field_and_reference(feed, CTX)
    assert len(result) == 1
    assert result[0].code == "missing_required_field"
    assert result[0].fields["fieldName"] == "record_id"


def test_wrong_record_id_for_single_key_table_yields_notice() -> None:
    """record_id not found in agency → translation_foreign_key_violation."""
    translations = make_translations(
        table_name="agency",
        field_name="any",
        record_id="any",
        language="en",
    )
    feed = make_feed(translations=translations)
    result = validate_translation_field_and_reference(feed, CTX)
    assert len(result) == 1
    assert result[0].code == "translation_foreign_key_violation"
    assert result[0].fields["tableName"] == "agency"
    assert result[0].fields["recordId"] == "any"
    assert result[0].fields["recordSubId"] == ""


def test_feed_info_translation_yields_no_notice() -> None:
    """0-key table feed_info with no ID fields → no notices."""
    translations = make_translations(
        table_name="feed_info",
        field_name="any",
        language="en",
    )
    feed = make_feed(translations=translations)
    result = validate_translation_field_and_reference(feed, CTX)
    assert result == []


def test_unexpected_record_id_for_feed_info_yields_notice() -> None:
    """feed_info (0-key) with record_id → unexpected_value."""
    translations = make_translations(
        table_name="feed_info",
        field_name="any",
        language="en",
        record_id="feed-id",
    )
    feed = make_feed(translations=translations)
    result = validate_translation_field_and_reference(feed, CTX)
    assert len(result) == 1
    assert result[0].code == "translation_unexpected_value"
    assert result[0].fields["fieldName"] == "record_id"
    assert result[0].fields["fieldValue"] == "feed-id"


def test_agency_translation_yields_no_notice() -> None:
    """Valid record_id matching seeded agency entity → no notices."""
    translations = make_translations(
        table_name="agency",
        field_name="any",
        record_id="agency0",
        language="en",
    )
    feed = make_feed(translations=translations)
    result = validate_translation_field_and_reference(feed, CTX)
    assert result == []


def test_stop_time_translation_yields_no_notice() -> None:
    """Two-key table stop_times with valid record_id and record_sub_id → no notices."""
    translations = make_translations(
        table_name="stop_times",
        field_name="any",
        record_id="trip0",
        record_sub_id="0",
        language="en",
    )
    feed = make_feed(translations=translations)
    result = validate_translation_field_and_reference(feed, CTX)
    assert result == []


def test_unparsable_integer_for_stop_time_yields_notice() -> None:
    """Non-integer record_sub_id for stop_times → translation_foreign_key_violation."""
    translations = make_translations(
        table_name="stop_times",
        field_name="any",
        record_id="trip0",
        record_sub_id="not-an-int",
        language="en",
    )
    feed = make_feed(translations=translations)
    result = validate_translation_field_and_reference(feed, CTX)
    assert len(result) == 1
    assert result[0].code == "translation_foreign_key_violation"
    assert result[0].fields["tableName"] == "stop_times"
    assert result[0].fields["recordId"] == "trip0"
    assert result[0].fields["recordSubId"] == "not-an-int"


def test_unknown_table_name_present_but_empty_parent() -> None:
    """Known table_name with empty parent DataFrame → FK violation (entity not found)."""
    translations = make_translations(
        table_name="agency",
        field_name="any",
        record_id="agency0",
        language="en",
    )
    feed = make_feed(
        translations=translations,
        agency=pl.DataFrame({"agency_id": [], "csvRowNumber": []}),
    )
    result = validate_translation_field_and_reference(feed, CTX)
    assert len(result) == 1
    assert result[0].code == "translation_foreign_key_violation"


def test_table_absent_from_feed() -> None:
    """Known table_name not present in feed → translation_unknown_table_name."""
    translations = make_translations(
        table_name="stops",
        field_name="any",
        record_id="stop0",
        language="en",
    )
    # stops not in feed
    feed = make_feed(translations=translations)
    del feed["agency"]  # also remove agency to simplify (stops is the one we test)
    # Actually don't remove agency — just ensure stops is absent (it is by default)
    feed2 = make_feed(translations=translations)
    # stops is not in the default feed
    result = validate_translation_field_and_reference(feed2, CTX)
    assert len(result) == 1
    assert result[0].code == "translation_unknown_table_name"
    assert result[0].fields["tableName"] == "stops"


def test_missing_record_sub_id_for_two_key_table() -> None:
    """Two-key table stop_times missing record_sub_id → missing_required_field."""
    translations = make_translations(
        table_name="stop_times",
        field_name="any",
        record_id="trip0",
        language="en",
    )
    feed = make_feed(translations=translations)
    result = validate_translation_field_and_reference(feed, CTX)
    assert len(result) == 1
    assert result[0].code == "missing_required_field"
    assert result[0].fields["fieldName"] == "record_sub_id"


def test_unexpected_record_sub_id_for_single_key_table() -> None:
    """Single-key table (agency) with record_sub_id → unexpected_value for record_sub_id."""
    translations = make_translations(
        table_name="agency",
        field_name="any",
        record_id="agency0",
        record_sub_id="extra",
        language="en",
    )
    feed = make_feed(translations=translations)
    result = validate_translation_field_and_reference(feed, CTX)
    assert len(result) == 1
    assert result[0].code == "translation_unexpected_value"
    assert result[0].fields["fieldName"] == "record_sub_id"
    assert result[0].fields["fieldValue"] == "extra"


def test_multiple_rows_with_mixed_errors() -> None:
    """Three rows: valid, invalid FK, unknown table → FK + unknown notices."""
    translations = pl.DataFrame({
        "table_name": ["agency", "agency", "bogus"],
        "field_name": ["agency_name", "agency_name", "x"],
        "language": ["en", "en", "en"],
        "translation": [None, None, None],
        "record_id": ["agency0", "does_not_exist", None],
        "record_sub_id": [None, None, None],
        "field_value": [None, None, None],
        "csvRowNumber": [2, 3, 4],
    })
    feed = make_feed(translations=translations)
    result = validate_translation_field_and_reference(feed, CTX)
    assert len(result) == 2
    codes = {n.code for n in result}
    assert "translation_foreign_key_violation" in codes
    assert "translation_unknown_table_name" in codes
    # Row 1 (agency0) should produce no notice
    fk_notices = [n for n in result if n.code == "translation_foreign_key_violation"]
    assert len(fk_notices) == 1
    assert fk_notices[0].fields["recordId"] == "does_not_exist"
