"""Tests for agency_consistency validator."""

from __future__ import annotations

from datetime import date

import polars as pl

from gtfs_validator.context import ValidationContext
from gtfs_validator.notices import Severity
from gtfs_validator.validators.agency_consistency import validate_agency_consistency

CTX = ValidationContext(country_code="US", date_for_validation=date(2024, 6, 1))


def _agency_df(**overrides: list) -> pl.DataFrame:  # type: ignore[type-arg]
    """Build a minimal agency DataFrame with sensible defaults."""
    defaults: dict[str, list] = {  # type: ignore[type-arg]
        "agency_id": ["a1"],
        "agency_name": ["Test Agency"],
        "agency_url": ["http://example.com"],
        "agency_timezone": ["America/Montreal"],
        "agency_lang": ["en"],
    }
    defaults.update(overrides)
    return pl.DataFrame(defaults)


# ------------------------------------------------------------------
# Single-agency tests
# ------------------------------------------------------------------


def test_single_agency_with_agency_id() -> None:
    feed = {"agency": _agency_df(agency_id=["a1"])}
    notices = validate_agency_consistency(feed, CTX)
    assert len(notices) == 0


def test_single_agency_missing_agency_id() -> None:
    feed = {"agency": _agency_df(agency_id=[None])}
    notices = validate_agency_consistency(feed, CTX)
    assert len(notices) == 1
    n = notices[0]
    assert n.code == "missing_recommended_field"
    assert n.severity == Severity.WARNING
    assert n.fields["filename"] == "agency.txt"
    assert n.fields["field_name"] == "agency_id"
    assert n.fields["csv_row_number"] == 1


# ------------------------------------------------------------------
# Multi-agency: agency_id required
# ------------------------------------------------------------------


def test_multi_agency_missing_agency_id() -> None:
    feed = {
        "agency": _agency_df(
            agency_id=["first agency", None],
            agency_name=["First", "agency name"],
            agency_url=["http://a.com", "http://b.com"],
            agency_timezone=["America/Montreal", "America/Montreal"],
            agency_lang=["en", "en"],
        )
    }
    notices = validate_agency_consistency(feed, CTX)
    assert len(notices) == 1
    n = notices[0]
    assert n.code == "missing_required_agency_id"
    assert n.severity == Severity.ERROR
    assert n.fields["agency_name"] == "agency name"


# ------------------------------------------------------------------
# Timezone consistency
# ------------------------------------------------------------------


def test_inconsistent_timezone() -> None:
    feed = {
        "agency": _agency_df(
            agency_id=["a1", "a2"],
            agency_name=["A1", "A2"],
            agency_url=["http://a.com", "http://b.com"],
            agency_timezone=["America/Bogota", "America/Montreal"],
            agency_lang=["en", "en"],
        )
    }
    notices = validate_agency_consistency(feed, CTX)
    tz_notices = [n for n in notices if n.code == "inconsistent_agency_timezone"]
    assert len(tz_notices) == 1
    n = tz_notices[0]
    assert n.severity == Severity.ERROR
    assert n.fields["expected"] == "America/Bogota"
    assert n.fields["actual"] == "America/Montreal"


def test_consistent_timezone() -> None:
    feed = {
        "agency": _agency_df(
            agency_id=["a1", "a2"],
            agency_name=["A1", "A2"],
            agency_url=["http://a.com", "http://b.com"],
            agency_timezone=["America/Montreal", "America/Montreal"],
            agency_lang=["en", "en"],
        )
    }
    notices = validate_agency_consistency(feed, CTX)
    assert len(notices) == 0


# ------------------------------------------------------------------
# Language consistency
# ------------------------------------------------------------------


def test_inconsistent_language() -> None:
    feed = {
        "agency": _agency_df(
            agency_id=["a1", "a2"],
            agency_name=["A1", "A2"],
            agency_url=["http://a.com", "http://b.com"],
            agency_timezone=["America/Montreal", "America/Montreal"],
            agency_lang=["en", "fr"],
        )
    }
    notices = validate_agency_consistency(feed, CTX)
    lang_notices = [n for n in notices if n.code == "inconsistent_agency_lang"]
    assert len(lang_notices) == 1
    n = lang_notices[0]
    assert n.severity == Severity.WARNING
    assert n.fields["expected"] == "en"
    assert n.fields["actual"] == "fr"


def test_consistent_language() -> None:
    feed = {
        "agency": _agency_df(
            agency_id=["a1", "a2"],
            agency_name=["A1", "A2"],
            agency_url=["http://a.com", "http://b.com"],
            agency_timezone=["America/Montreal", "America/Montreal"],
            agency_lang=["en", "en"],
        )
    }
    notices = validate_agency_consistency(feed, CTX)
    assert len(notices) == 0


def test_some_languages_null_rest_consistent() -> None:
    feed = {
        "agency": _agency_df(
            agency_id=["a1", "a2", "a3"],
            agency_name=["A1", "A2", "A3"],
            agency_url=["http://a.com", "http://b.com", "http://c.com"],
            agency_timezone=["America/Montreal", "America/Montreal", "America/Montreal"],
            agency_lang=[None, "fr", "fr"],
        )
    }
    notices = validate_agency_consistency(feed, CTX)
    assert len(notices) == 0


# ------------------------------------------------------------------
# Edge cases
# ------------------------------------------------------------------


def test_agency_table_missing() -> None:
    feed: dict[str, pl.DataFrame] = {}
    notices = validate_agency_consistency(feed, CTX)
    assert len(notices) == 0


def test_agency_table_empty() -> None:
    feed = {
        "agency": pl.DataFrame(
            schema={
                "agency_id": pl.Utf8,
                "agency_name": pl.Utf8,
                "agency_url": pl.Utf8,
                "agency_timezone": pl.Utf8,
                "agency_lang": pl.Utf8,
            }
        )
    }
    notices = validate_agency_consistency(feed, CTX)
    assert len(notices) == 0
