"""Tests for the duplicate_fare_media validator."""

from datetime import date

import polars as pl

from gtfs_validator.context import ValidationContext
from gtfs_validator.notices import Severity
from gtfs_validator.validators.duplicate_fare_media import validate_duplicate_fare_media

CTX = ValidationContext(country_code="US", date_for_validation=date(2024, 1, 1))


def make_fare_media(rows: list[dict]) -> pl.DataFrame:
    """Create a fare_media DataFrame with fare_media_id,
    fare_media_name, fare_media_type columns."""
    if not rows:
        return pl.DataFrame(
            schema={
                "fare_media_id": pl.Utf8,
                "fare_media_name": pl.Utf8,
                "fare_media_type": pl.Int64,
            }
        )
    return pl.DataFrame(rows).cast({
        "fare_media_id": pl.Utf8,
        "fare_media_name": pl.Utf8,
        "fare_media_type": pl.Int64,
    })


def test_unique_entries_no_notice():
    feed = {"fare_media": make_fare_media([
        {"fare_media_id": "a", "fare_media_name": "Transit Card", "fare_media_type": 2},
        {"fare_media_id": "b", "fare_media_name": "Transit App", "fare_media_type": 4},
    ])}
    assert validate_duplicate_fare_media(feed, CTX) == []


def test_duplicate_name_and_type():
    feed = {"fare_media": make_fare_media([
        {"fare_media_id": "a", "fare_media_name": "Transit Card", "fare_media_type": 2},
        {"fare_media_id": "b", "fare_media_name": "Transit Card", "fare_media_type": 2},
    ])}
    notices = validate_duplicate_fare_media(feed, CTX)
    assert len(notices) == 1
    n = notices[0]
    assert n.code == "duplicate_fare_media"
    assert n.severity == Severity.WARNING
    assert n.fields["csv_row_number_1"] == 1
    assert n.fields["fare_media_id_1"] == "a"
    assert n.fields["csv_row_number_2"] == 2
    assert n.fields["fare_media_id_2"] == "b"


def test_same_type_different_names_no_notice():
    feed = {"fare_media": make_fare_media([
        {"fare_media_id": "a", "fare_media_name": "Transit Card A", "fare_media_type": 2},
        {"fare_media_id": "b", "fare_media_name": "Transit Card B", "fare_media_type": 2},
    ])}
    assert validate_duplicate_fare_media(feed, CTX) == []


def test_same_name_different_types_no_notice():
    feed = {"fare_media": make_fare_media([
        {"fare_media_id": "a", "fare_media_name": "Card", "fare_media_type": 2},
        {"fare_media_id": "b", "fare_media_name": "Card", "fare_media_type": 3},
    ])}
    assert validate_duplicate_fare_media(feed, CTX) == []


def test_empty_table_no_notice():
    feed = {"fare_media": make_fare_media([])}
    assert validate_duplicate_fare_media(feed, CTX) == []


def test_three_way_duplicate():
    feed = {"fare_media": make_fare_media([
        {"fare_media_id": "a", "fare_media_name": "Card", "fare_media_type": 2},
        {"fare_media_id": "b", "fare_media_name": "Card", "fare_media_type": 2},
        {"fare_media_id": "c", "fare_media_name": "Card", "fare_media_type": 2},
    ])}
    notices = validate_duplicate_fare_media(feed, CTX)
    assert len(notices) == 2
    assert notices[0].fields["csv_row_number_1"] == 1
    assert notices[0].fields["fare_media_id_1"] == "a"
    assert notices[0].fields["csv_row_number_2"] == 2
    assert notices[0].fields["fare_media_id_2"] == "b"
    assert notices[1].fields["csv_row_number_1"] == 1
    assert notices[1].fields["fare_media_id_1"] == "a"
    assert notices[1].fields["csv_row_number_2"] == 3
    assert notices[1].fields["fare_media_id_2"] == "c"


def test_null_name_match():
    feed = {"fare_media": make_fare_media([
        {"fare_media_id": "a", "fare_media_name": None, "fare_media_type": 0},
        {"fare_media_id": "b", "fare_media_name": None, "fare_media_type": 0},
    ])}
    notices = validate_duplicate_fare_media(feed, CTX)
    assert len(notices) == 1
    assert notices[0].fields["csv_row_number_1"] == 1
    assert notices[0].fields["fare_media_id_1"] == "a"
    assert notices[0].fields["csv_row_number_2"] == 2
    assert notices[0].fields["fare_media_id_2"] == "b"


def test_empty_string_name_match():
    feed = {"fare_media": make_fare_media([
        {"fare_media_id": "a", "fare_media_name": "", "fare_media_type": 0},
        {"fare_media_id": "b", "fare_media_name": "", "fare_media_type": 0},
    ])}
    notices = validate_duplicate_fare_media(feed, CTX)
    assert len(notices) == 1
    assert notices[0].code == "duplicate_fare_media"


def test_missing_table_no_notice():
    feed: dict[str, pl.DataFrame] = {}
    assert validate_duplicate_fare_media(feed, CTX) == []


def test_single_row_no_notice():
    feed = {"fare_media": make_fare_media([
        {"fare_media_id": "a", "fare_media_name": "Card", "fare_media_type": 2},
    ])}
    assert validate_duplicate_fare_media(feed, CTX) == []


def test_multiple_duplicate_groups():
    feed = {"fare_media": make_fare_media([
        {"fare_media_id": "a", "fare_media_name": "Card", "fare_media_type": 2},
        {"fare_media_id": "b", "fare_media_name": "Card", "fare_media_type": 2},
        {"fare_media_id": "c", "fare_media_name": "App", "fare_media_type": 4},
        {"fare_media_id": "d", "fare_media_name": "App", "fare_media_type": 4},
    ])}
    notices = validate_duplicate_fare_media(feed, CTX)
    assert len(notices) == 2
    # Sort by csv_row_number_2 for deterministic assertion
    notices.sort(key=lambda n: n.fields["csv_row_number_2"])
    assert notices[0].fields["fare_media_id_1"] == "a"
    assert notices[0].fields["fare_media_id_2"] == "b"
    assert notices[1].fields["fare_media_id_1"] == "c"
    assert notices[1].fields["fare_media_id_2"] == "d"


def test_notice_severity_is_warning():
    feed = {"fare_media": make_fare_media([
        {"fare_media_id": "a", "fare_media_name": "Card", "fare_media_type": 2},
        {"fare_media_id": "b", "fare_media_name": "Card", "fare_media_type": 2},
    ])}
    notices = validate_duplicate_fare_media(feed, CTX)
    assert all(n.severity == Severity.WARNING for n in notices)


def test_notice_code():
    feed = {"fare_media": make_fare_media([
        {"fare_media_id": "a", "fare_media_name": "Card", "fare_media_type": 2},
        {"fare_media_id": "b", "fare_media_name": "Card", "fare_media_type": 2},
    ])}
    notices = validate_duplicate_fare_media(feed, CTX)
    assert all(n.code == "duplicate_fare_media" for n in notices)
