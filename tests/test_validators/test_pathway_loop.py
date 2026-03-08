"""Tests for validate_pathway_loop."""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from gtfs_validator.context import ValidationContext
from gtfs_validator.notices import Severity
from gtfs_validator.validators.pathway_loop import validate_pathway_loop

CTX = ValidationContext(country_code="US", date_for_validation=date(2026, 3, 8))

PATHWAYS_SCHEMA = {
    "csv_row_number": pl.Int64,
    "pathway_id": pl.Utf8,
    "from_stop_id": pl.Utf8,
    "to_stop_id": pl.Utf8,
}


def make_feed(rows: list[dict]) -> dict[str, pl.DataFrame]:
    return {"pathways": pl.DataFrame(rows, schema=PATHWAYS_SCHEMA)}


def test_loop_yields_notice() -> None:
    feed = make_feed([
        {"csv_row_number": 2, "pathway_id": "pw1", "from_stop_id": "platform1", "to_stop_id": "platform1"},
    ])
    notices = validate_pathway_loop(feed, CTX)
    assert len(notices) == 1
    n = notices[0]
    assert n.code == "pathway_loop"
    assert n.severity == Severity.WARNING
    assert n.fields == {"csvRowNumber": 2, "pathwayId": "pw1", "stopId": "platform1"}


def test_no_loop_yields_no_notice() -> None:
    feed = make_feed([
        {"csv_row_number": 2, "pathway_id": "pw1", "from_stop_id": "platform1", "to_stop_id": "entrance2"},
    ])
    assert validate_pathway_loop(feed, CTX) == []


def test_no_endpoints_yields_no_notice() -> None:
    feed = make_feed([
        {"csv_row_number": 2, "pathway_id": "pw1", "from_stop_id": None, "to_stop_id": None},
    ])
    assert validate_pathway_loop(feed, CTX) == []


def test_null_from_stop_id_yields_no_notice() -> None:
    feed = make_feed([
        {"csv_row_number": 3, "pathway_id": "pw2", "from_stop_id": None, "to_stop_id": "platform1"},
    ])
    assert validate_pathway_loop(feed, CTX) == []


def test_null_to_stop_id_yields_no_notice() -> None:
    feed = make_feed([
        {"csv_row_number": 4, "pathway_id": "pw3", "from_stop_id": "platform1", "to_stop_id": None},
    ])
    assert validate_pathway_loop(feed, CTX) == []


def test_multiple_loop_rows_yield_one_notice_each() -> None:
    feed = make_feed([
        {"csv_row_number": 2, "pathway_id": "pw1", "from_stop_id": "s1", "to_stop_id": "s1"},
        {"csv_row_number": 3, "pathway_id": "pw2", "from_stop_id": "s1", "to_stop_id": "s2"},
        {"csv_row_number": 4, "pathway_id": "pw3", "from_stop_id": "s3", "to_stop_id": "s3"},
    ])
    notices = validate_pathway_loop(feed, CTX)
    assert len(notices) == 2
    assert notices[0].fields == {"csvRowNumber": 2, "pathwayId": "pw1", "stopId": "s1"}
    assert notices[1].fields == {"csvRowNumber": 4, "pathwayId": "pw3", "stopId": "s3"}


def test_pathways_absent_yields_no_notice() -> None:
    assert validate_pathway_loop({}, CTX) == []


def test_pathways_empty_yields_no_notice() -> None:
    feed = {"pathways": pl.DataFrame(schema=PATHWAYS_SCHEMA)}
    assert validate_pathway_loop(feed, CTX) == []
