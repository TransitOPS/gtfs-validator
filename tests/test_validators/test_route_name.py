"""Tests for validate_route_name."""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from gtfs_validator.context import ValidationContext
from gtfs_validator.notices import Severity
from gtfs_validator.validators.route_name import validate_route_name

CTX = ValidationContext(country_code="US", date_for_validation=date(2024, 1, 1))


def make_feed(rows: list[dict]) -> dict[str, pl.DataFrame]:
    """Build a minimal feed dict with a routes DataFrame."""
    schema = {
        "route_id": pl.Utf8,
        "csv_row_number": pl.Int64,
        "route_short_name": pl.Utf8,
        "route_long_name": pl.Utf8,
        "route_desc": pl.Utf8,
    }
    if not rows:
        return {"routes": pl.DataFrame(schema=schema)}
    return {"routes": pl.DataFrame(rows).cast(schema)}


def make_route(
    short: str | None,
    long_: str | None,
    desc: str | None,
    route_id: str = "r1",
    csv_row_number: int = 2,
) -> dict[str, pl.DataFrame]:
    return make_feed([{
        "route_id": route_id,
        "csv_row_number": csv_row_number,
        "route_short_name": short,
        "route_long_name": long_,
        "route_desc": desc,
    }])


# ---------------------------------------------------------------------------
# Group 1: route_both_short_and_long_name_missing
# ---------------------------------------------------------------------------

def test_both_names_null_emits_error():
    notices = validate_route_name(make_route(None, None, None), CTX)
    assert len(notices) == 1
    n = notices[0]
    assert n.code == "route_both_short_and_long_name_missing"
    assert n.severity == Severity.ERROR
    assert n.fields["route_id"] == "r1"
    assert n.fields["csv_row_number"] == 2


def test_only_short_name_no_missing_notice():
    notices = validate_route_name(make_route("S", None, None), CTX)
    codes = [n.code for n in notices]
    assert "route_both_short_and_long_name_missing" not in codes


def test_only_long_name_no_missing_notice():
    notices = validate_route_name(make_route(None, "Long", None), CTX)
    codes = [n.code for n in notices]
    assert "route_both_short_and_long_name_missing" not in codes


def test_both_names_present_no_missing_notice():
    notices = validate_route_name(make_route("S", "Long", None), CTX)
    codes = [n.code for n in notices]
    assert "route_both_short_and_long_name_missing" not in codes


# ---------------------------------------------------------------------------
# Group 2: route_short_name_too_long
# ---------------------------------------------------------------------------

def test_short_name_17_chars_emits_warning():
    notices = validate_route_name(make_route("THISISMYSHORTNAME", None, None), CTX)
    matches = [n for n in notices if n.code == "route_short_name_too_long"]
    assert len(matches) == 1
    n = matches[0]
    assert n.severity == Severity.WARNING
    assert n.fields["route_id"] == "r1"
    assert n.fields["csv_row_number"] == 2
    assert n.fields["route_short_name"] == "THISISMYSHORTNAME"


def test_short_name_2_chars_no_warning():
    notices = validate_route_name(make_route("SH", None, None), CTX)
    codes = [n.code for n in notices]
    assert "route_short_name_too_long" not in codes


def test_short_name_exactly_12_chars_no_warning():
    notices = validate_route_name(make_route("ABCDEFGHIJKL", None, None), CTX)
    codes = [n.code for n in notices]
    assert "route_short_name_too_long" not in codes


def test_short_name_13_chars_emits_warning():
    notices = validate_route_name(make_route("ABCDEFGHIJKLM", None, None), CTX)
    matches = [n for n in notices if n.code == "route_short_name_too_long"]
    assert len(matches) == 1


# ---------------------------------------------------------------------------
# Group 3: route_long_name_contains_short_name
# ---------------------------------------------------------------------------

def test_names_identical_emits_warning():
    notices = validate_route_name(make_route("S", "S", None), CTX)
    matches = [n for n in notices if n.code == "route_long_name_contains_short_name"]
    assert len(matches) == 1
    n = matches[0]
    assert n.severity == Severity.WARNING
    assert n.fields["route_short_name"] == "S"
    assert n.fields["route_long_name"] == "S"


def test_names_identical_case_insensitive_emits_warning():
    notices = validate_route_name(make_route("SA", "Sa", None), CTX)
    matches = [n for n in notices if n.code == "route_long_name_contains_short_name"]
    assert len(matches) == 1
    assert matches[0].fields["route_short_name"] == "SA"
    assert matches[0].fields["route_long_name"] == "Sa"


def test_long_name_unrelated_no_warning():
    notices = validate_route_name(make_route("S", "Long", None), CTX)
    codes = [n.code for n in notices]
    assert "route_long_name_contains_short_name" not in codes


def test_long_name_short_plus_space_plus_content():
    notices = validate_route_name(make_route("L1", "L1 Long Name", None), CTX)
    matches = [n for n in notices if n.code == "route_long_name_contains_short_name"]
    assert len(matches) == 1


def test_long_name_short_plus_hyphen():
    notices = validate_route_name(make_route("L1", "L1-Long Name", None), CTX)
    matches = [n for n in notices if n.code == "route_long_name_contains_short_name"]
    assert len(matches) == 1


def test_long_name_short_plus_hyphen_space():
    notices = validate_route_name(make_route("L1", "L1- Long Name", None), CTX)
    matches = [n for n in notices if n.code == "route_long_name_contains_short_name"]
    assert len(matches) == 1


def test_long_name_short_plus_space_hyphen():
    notices = validate_route_name(make_route("L1", "L1 - Long Name", None), CTX)
    matches = [n for n in notices if n.code == "route_long_name_contains_short_name"]
    assert len(matches) == 1


def test_long_name_short_plus_close_paren():
    notices = validate_route_name(make_route("L1", "L1)Long Name", None), CTX)
    matches = [n for n in notices if n.code == "route_long_name_contains_short_name"]
    assert len(matches) == 1


def test_long_name_short_plus_close_paren_space():
    notices = validate_route_name(make_route("L1", "L1) Long Name", None), CTX)
    matches = [n for n in notices if n.code == "route_long_name_contains_short_name"]
    assert len(matches) == 1


def test_long_name_short_plus_space_close_paren():
    notices = validate_route_name(make_route("L1", "L1 ) Long Name", None), CTX)
    matches = [n for n in notices if n.code == "route_long_name_contains_short_name"]
    assert len(matches) == 1


def test_long_name_short_plus_open_paren():
    notices = validate_route_name(make_route("L1", "L1(Long Name)", None), CTX)
    matches = [n for n in notices if n.code == "route_long_name_contains_short_name"]
    assert len(matches) == 1


def test_long_name_short_plus_space_open_paren():
    notices = validate_route_name(make_route("L1", "L1 (Long Name)", None), CTX)
    matches = [n for n in notices if n.code == "route_long_name_contains_short_name"]
    assert len(matches) == 1


def test_long_name_short_plus_open_paren_space():
    notices = validate_route_name(make_route("L1", "L1( Long Name)", None), CTX)
    matches = [n for n in notices if n.code == "route_long_name_contains_short_name"]
    assert len(matches) == 1


def test_long_name_starts_with_short_but_remainder_is_letter_no_warning():
    notices = validate_route_name(make_route("BAN", "Bankford", None), CTX)
    codes = [n.code for n in notices]
    assert "route_long_name_contains_short_name" not in codes


def test_only_long_name_no_short_name_skips_prefix_check():
    notices = validate_route_name(make_route(None, "Long Name", None), CTX)
    codes = [n.code for n in notices]
    assert "route_long_name_contains_short_name" not in codes


# ---------------------------------------------------------------------------
# Group 4: same_name_and_description_for_route
# ---------------------------------------------------------------------------

def test_desc_equals_short_name_emits_warning():
    notices = validate_route_name(make_route("duplicate", None, "duplicate"), CTX)
    matches = [n for n in notices if n.code == "same_name_and_description_for_route"]
    assert len(matches) == 1
    n = matches[0]
    assert n.severity == Severity.WARNING
    assert n.fields["csv_row_number"] == 2
    assert n.fields["route_id"] == "r1"
    assert n.fields["route_desc"] == "duplicate"
    assert n.fields["specified_field"] == "route_short_name"


def test_desc_equals_short_name_case_insensitive():
    notices = validate_route_name(make_route("DuplicATE", None, "duplicate"), CTX)
    matches = [n for n in notices if n.code == "same_name_and_description_for_route"]
    assert len(matches) == 1
    assert matches[0].fields["specified_field"] == "route_short_name"
    assert matches[0].fields["route_desc"] == "duplicate"


def test_desc_equals_long_name_emits_warning():
    notices = validate_route_name(make_route(None, "duplicate", "duplicate"), CTX)
    matches = [n for n in notices if n.code == "same_name_and_description_for_route"]
    assert len(matches) == 1
    assert matches[0].fields["specified_field"] == "route_long_name"
    assert matches[0].fields["route_desc"] == "duplicate"


def test_desc_equals_long_name_case_insensitive_desc_mixed_case():
    notices = validate_route_name(make_route(None, "duplicate", "DuplicATE"), CTX)
    matches = [n for n in notices if n.code == "same_name_and_description_for_route"]
    assert len(matches) == 1
    assert matches[0].fields["specified_field"] == "route_long_name"
    assert matches[0].fields["route_desc"] == "DuplicATE"


def test_desc_different_from_short_name_no_notice():
    notices = validate_route_name(make_route("short name", None, "desc"), CTX)
    codes = [n.code for n in notices]
    assert "same_name_and_description_for_route" not in codes


def test_desc_different_from_long_name_no_notice():
    notices = validate_route_name(make_route(None, "long name", "desc"), CTX)
    codes = [n.code for n in notices]
    assert "same_name_and_description_for_route" not in codes


def test_all_names_different_from_desc_no_notice():
    notices = validate_route_name(make_route("short name", "long name", "desc"), CTX)
    codes = [n.code for n in notices]
    assert "same_name_and_description_for_route" not in codes


def test_desc_null_no_notice():
    notices = validate_route_name(make_route("S", "Long", None), CTX)
    codes = [n.code for n in notices]
    assert "same_name_and_description_for_route" not in codes


# ---------------------------------------------------------------------------
# Group 5: short-circuit and multi-notice interactions
# ---------------------------------------------------------------------------

def test_all_three_equal_emits_two_notices():
    notices = validate_route_name(make_route("duplicate", "duplicate", "duplicate"), CTX)
    codes = [n.code for n in notices]
    assert codes.count("route_long_name_contains_short_name") == 1
    assert codes.count("same_name_and_description_for_route") == 1
    desc_notices = [n for n in notices if n.code == "same_name_and_description_for_route"]
    assert desc_notices[0].fields["specified_field"] == "route_short_name"
    assert len(notices) == 2


def test_desc_matches_short_name_suppresses_long_name_desc_check():
    notices = validate_route_name(make_route("SAME", "Different", "same"), CTX)
    matches = [n for n in notices if n.code == "same_name_and_description_for_route"]
    assert len(matches) == 1
    assert matches[0].fields["specified_field"] == "route_short_name"


def test_short_name_too_long_and_names_equal_both_emit():
    notices = validate_route_name(make_route("TOOLONGSHORTNAME", "TOOLONGSHORTNAME", None), CTX)
    codes = [n.code for n in notices]
    assert "route_short_name_too_long" in codes
    assert "route_long_name_contains_short_name" in codes
    assert len(notices) == 2


# ---------------------------------------------------------------------------
# Group 6: table-level guards
# ---------------------------------------------------------------------------

def test_missing_routes_table_returns_empty():
    notices = validate_route_name({}, CTX)
    assert notices == []


def test_empty_routes_table_returns_empty():
    notices = validate_route_name(make_feed([]), CTX)
    assert notices == []


def test_multiple_rows_each_checked_independently():
    feed = make_feed([
        {
            "route_id": "r1",
            "csv_row_number": 2,
            "route_short_name": None,
            "route_long_name": None,
            "route_desc": None,
        },
        {
            "route_id": "r2",
            "csv_row_number": 3,
            "route_short_name": "TOOLONGNAME123",
            "route_long_name": None,
            "route_desc": None,
        },
    ])
    notices = validate_route_name(feed, CTX)
    assert len(notices) == 2
    missing = [n for n in notices if n.code == "route_both_short_and_long_name_missing"]
    too_long = [n for n in notices if n.code == "route_short_name_too_long"]
    assert len(missing) == 1
    assert missing[0].fields["route_id"] == "r1"
    assert missing[0].fields["csv_row_number"] == 2
    assert len(too_long) == 1
    assert too_long[0].fields["route_id"] == "r2"
    assert too_long[0].fields["csv_row_number"] == 3
