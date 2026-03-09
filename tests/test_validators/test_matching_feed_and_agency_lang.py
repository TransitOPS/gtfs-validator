"""Tests for MatchingFeedAndAgencyLangValidator."""

from datetime import date

import polars as pl
import pytest

from gtfs_validator.context import ValidationContext
from gtfs_validator.notices import Severity
from gtfs_validator.validators.matching_feed_and_agency_lang import (
    validate_matching_feed_and_agency_lang,
)

CTX = ValidationContext(country_code="US", date_for_validation=date(2026, 3, 8))


def make_feed_info(feed_lang: str | None) -> pl.DataFrame:
    """Build a feed_info DataFrame with a single row."""
    return pl.DataFrame(
        {
            "csvRowNumber": [1],
            "feed_publisher_name": ["Test Publisher"],
            "feed_publisher_url": ["http://example.com"],
            "feed_lang": [feed_lang],
        },
        schema={
            "csvRowNumber": pl.Int64,
            "feed_publisher_name": pl.Utf8,
            "feed_publisher_url": pl.Utf8,
            "feed_lang": pl.Utf8,
        },
    )


def make_agency(rows: list[dict]) -> pl.DataFrame:
    """Build an agency DataFrame from row dicts."""
    schema = {
        "csvRowNumber": pl.Int64,
        "agency_id": pl.Utf8,
        "agency_name": pl.Utf8,
        "agency_lang": pl.Utf8,
    }
    data: dict[str, list] = {col: [] for col in schema}
    for r in rows:
        for col in schema:
            data[col].append(r.get(col))
    return pl.DataFrame(data, schema=schema)


def test_no_feed_info_no_notice():
    feed = {"agency": make_agency([])}
    assert validate_matching_feed_and_agency_lang(feed, CTX) == []


def test_empty_feed_info_no_notice():
    feed = {
        "feed_info": pl.DataFrame(
            schema={"csvRowNumber": pl.Int64, "feed_lang": pl.Utf8}
        ),
        "agency": make_agency([]),
    }
    assert validate_matching_feed_and_agency_lang(feed, CTX) == []


def test_no_feed_lang_no_notice():
    feed = {
        "feed_info": make_feed_info(None),
        "agency": make_agency([
            {"csvRowNumber": 2, "agency_id": "a1", "agency_name": "A1", "agency_lang": "en"},
            {"csvRowNumber": 3, "agency_id": "a2", "agency_name": "A2", "agency_lang": "fr"},
        ]),
    }
    assert validate_matching_feed_and_agency_lang(feed, CTX) == []


def test_feed_lang_empty_string_no_notice():
    feed = {
        "feed_info": make_feed_info(""),
        "agency": make_agency([
            {"csvRowNumber": 2, "agency_id": "a1", "agency_name": "A1", "agency_lang": "en"},
        ]),
    }
    assert validate_matching_feed_and_agency_lang(feed, CTX) == []


def test_language_mismatch():
    feed = {
        "feed_info": make_feed_info("fr-FR"),
        "agency": make_agency([
            {"csvRowNumber": 2, "agency_id": "agencyCa", "agency_name": "agencyCa name", "agency_lang": "fr-CA"},
            {"csvRowNumber": 3, "agency_id": "agencyFr", "agency_name": "agencyFr name", "agency_lang": "fr-FR"},
            {"csvRowNumber": 4, "agency_id": "agencyNull", "agency_name": "agencyNull name", "agency_lang": None},
        ]),
    }
    notices = validate_matching_feed_and_agency_lang(feed, CTX)
    assert len(notices) == 1
    n = notices[0]
    assert n.code == "feed_info_lang_and_agency_lang_mismatch"
    assert n.severity == Severity.WARNING
    assert n.fields["csvRowNumber"] == 2
    assert n.fields["agencyId"] == "agencyCa"
    assert n.fields["agencyName"] == "agencyCa name"
    assert n.fields["agencyLang"] == "fr-CA"
    assert n.fields["feedLang"] == "fr-FR"


def test_multilanguage_feed_different_agencies_no_notice():
    feed = {
        "feed_info": make_feed_info("mul"),
        "agency": make_agency([
            {"csvRowNumber": 2, "agency_id": "a1", "agency_name": "A1", "agency_lang": "en"},
            {"csvRowNumber": 3, "agency_id": "a2", "agency_name": "A2", "agency_lang": "fr"},
        ]),
    }
    assert validate_matching_feed_and_agency_lang(feed, CTX) == []


def test_multilanguage_feed_single_agency_no_notice():
    feed = {
        "feed_info": make_feed_info("mul"),
        "agency": make_agency([
            {"csvRowNumber": 2, "agency_id": "a1", "agency_name": "A1", "agency_lang": "en"},
        ]),
    }
    assert validate_matching_feed_and_agency_lang(feed, CTX) == []


def test_agency_table_missing_no_notice():
    feed = {"feed_info": make_feed_info("en")}
    assert validate_matching_feed_and_agency_lang(feed, CTX) == []


def test_agency_table_empty_no_notice():
    feed = {"feed_info": make_feed_info("en"), "agency": make_agency([])}
    assert validate_matching_feed_and_agency_lang(feed, CTX) == []


def test_all_agencies_match_no_notice():
    feed = {
        "feed_info": make_feed_info("en"),
        "agency": make_agency([
            {"csvRowNumber": 2, "agency_id": "a1", "agency_name": "A1", "agency_lang": "en"},
            {"csvRowNumber": 3, "agency_id": "a2", "agency_name": "A2", "agency_lang": "en"},
        ]),
    }
    assert validate_matching_feed_and_agency_lang(feed, CTX) == []


def test_multiple_mismatches():
    feed = {
        "feed_info": make_feed_info("en"),
        "agency": make_agency([
            {"csvRowNumber": 2, "agency_id": "a1", "agency_name": "A1", "agency_lang": "fr"},
            {"csvRowNumber": 3, "agency_id": "a2", "agency_name": "A2", "agency_lang": "de"},
            {"csvRowNumber": 4, "agency_id": "a3", "agency_name": "A3", "agency_lang": "en"},
        ]),
    }
    notices = validate_matching_feed_and_agency_lang(feed, CTX)
    assert len(notices) == 2
    assert all(n.code == "feed_info_lang_and_agency_lang_mismatch" for n in notices)
    assert all(n.severity == Severity.WARNING for n in notices)


def test_case_insensitive_match_no_notice():
    feed = {
        "feed_info": make_feed_info("EN"),
        "agency": make_agency([
            {"csvRowNumber": 2, "agency_id": "a1", "agency_name": "A1", "agency_lang": "en"},
        ]),
    }
    assert validate_matching_feed_and_agency_lang(feed, CTX) == []


def test_mul_case_insensitive_no_notice():
    feed = {
        "feed_info": make_feed_info("MUL"),
        "agency": make_agency([
            {"csvRowNumber": 2, "agency_id": "a1", "agency_name": "A1", "agency_lang": "en"},
            {"csvRowNumber": 3, "agency_id": "a2", "agency_name": "A2", "agency_lang": "fr"},
        ]),
    }
    assert validate_matching_feed_and_agency_lang(feed, CTX) == []


def test_agency_lang_column_missing_no_notice():
    feed = {
        "feed_info": make_feed_info("en"),
        "agency": pl.DataFrame({
            "csvRowNumber": [1],
            "agency_id": ["a1"],
            "agency_name": ["Agency 1"],
        }),
    }
    assert validate_matching_feed_and_agency_lang(feed, CTX) == []
