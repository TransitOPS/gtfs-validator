"""Tests for validate_fare_media_name."""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from gtfs_validator.context import ValidationContext
from gtfs_validator.notices import Severity
from gtfs_validator.validators.fare_media_name import validate_fare_media_name

CTX = ValidationContext(country_code="US", date_for_validation=date(2024, 1, 1))


def make_fare_media(**kwargs) -> pl.DataFrame:
    """Build a fare_media DataFrame with sensible defaults."""
    defaults: dict = {
        "fare_media_id": ["m1"],
        "fare_media_name": ["Go! Pass"],
        "fare_media_type": [2],
    }
    defaults.update(kwargs)
    return pl.DataFrame(defaults)


def test_transit_card_with_name_no_notices() -> None:
    feed = {"fare_media": make_fare_media(fare_media_type=[2], fare_media_name=["Go! Pass"])}
    assert validate_fare_media_name(feed, CTX) == []


def test_transit_card_without_name_yields_notice() -> None:
    feed = {"fare_media": make_fare_media(fare_media_type=[2], fare_media_name=[None])}
    notices = validate_fare_media_name(feed, CTX)
    assert len(notices) == 1
    n = notices[0]
    assert n.code == "missing_recommended_field"
    assert n.severity == Severity.WARNING
    assert n.fields["filename"] == "fare_media.txt"
    assert n.fields["field_name"] == "fare_media_name"
    assert n.fields["csv_row_number"] == 0


def test_paper_ticket_with_name_no_notices() -> None:
    feed = {"fare_media": make_fare_media(fare_media_type=[1], fare_media_name=["Some Ticket"])}
    assert validate_fare_media_name(feed, CTX) == []


def test_paper_ticket_without_name_no_notices() -> None:
    feed = {"fare_media": make_fare_media(fare_media_type=[1], fare_media_name=[None])}
    assert validate_fare_media_name(feed, CTX) == []


def test_mobile_app_with_name_no_notices() -> None:
    feed = {"fare_media": make_fare_media(fare_media_type=[4], fare_media_name=["MyApp"])}
    assert validate_fare_media_name(feed, CTX) == []


def test_mobile_app_without_name_yields_notice() -> None:
    feed = {"fare_media": make_fare_media(fare_media_type=[4], fare_media_name=[None])}
    notices = validate_fare_media_name(feed, CTX)
    assert len(notices) == 1
    assert notices[0].fields["field_name"] == "fare_media_name"


def test_none_type_without_name_no_notices() -> None:
    feed = {"fare_media": make_fare_media(fare_media_type=[0], fare_media_name=[None])}
    assert validate_fare_media_name(feed, CTX) == []


def test_contactless_emv_without_name_no_notices() -> None:
    feed = {"fare_media": make_fare_media(fare_media_type=[3], fare_media_name=[None])}
    assert validate_fare_media_name(feed, CTX) == []


def test_empty_string_name_for_transit_card_yields_notice() -> None:
    feed = {"fare_media": make_fare_media(fare_media_type=[2], fare_media_name=[""])}
    notices = validate_fare_media_name(feed, CTX)
    assert len(notices) == 1
    assert notices[0].code == "missing_recommended_field"


def test_fare_media_absent_skips() -> None:
    assert validate_fare_media_name({}, CTX) == []


def test_fare_media_empty_skips() -> None:
    empty = pl.DataFrame(
        {"fare_media_id": [], "fare_media_name": [], "fare_media_type": []},
        schema={"fare_media_id": pl.Utf8, "fare_media_name": pl.Utf8, "fare_media_type": pl.Int32},
    )
    assert validate_fare_media_name({"fare_media": empty}, CTX) == []


def test_multiple_rows_two_violations() -> None:
    df = pl.DataFrame(
        {
            "fare_media_id": ["m1", "m2"],
            "fare_media_name": [None, None],
            "fare_media_type": [2, 2],
        }
    )
    notices = validate_fare_media_name({"fare_media": df}, CTX)
    assert len(notices) == 2
    assert notices[0].fields["csv_row_number"] == 0
    assert notices[1].fields["csv_row_number"] == 1


def test_multiple_rows_mixed_types_one_violation() -> None:
    df = pl.DataFrame(
        {
            "fare_media_id": ["m1", "m2", "m3"],
            "fare_media_name": [None, None, "App"],
            "fare_media_type": [2, 1, 4],
        }
    )
    notices = validate_fare_media_name({"fare_media": df}, CTX)
    assert len(notices) == 1
    assert notices[0].fields["csv_row_number"] == 0


def test_null_fare_media_type_no_notices() -> None:
    df = pl.DataFrame(
        {
            "fare_media_id": ["m1"],
            "fare_media_name": [None],
            "fare_media_type": [None],
        },
        schema={"fare_media_id": pl.Utf8, "fare_media_name": pl.Utf8, "fare_media_type": pl.Int32},
    )
    assert validate_fare_media_name({"fare_media": df}, CTX) == []


def test_notice_fields_exact_values() -> None:
    feed = {"fare_media": make_fare_media(fare_media_type=[2], fare_media_name=[None])}
    notices = validate_fare_media_name(feed, CTX)
    assert len(notices) == 1
    n = notices[0]
    assert n.code == "missing_recommended_field"
    assert n.severity == Severity.WARNING
    assert n.fields["filename"] == "fare_media.txt"
    assert n.fields["field_name"] == "fare_media_name"
    assert n.fields["csv_row_number"] == 0
