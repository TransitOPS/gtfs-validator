"""Tests for validate_pathway_stop_access."""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from gtfs_validator.context import ValidationContext
from gtfs_validator.notices import Severity
from gtfs_validator.validators.pathway_stop_access import validate_pathway_stop_access


@pytest.fixture()
def ctx() -> ValidationContext:
    return ValidationContext(country_code="US", date_for_validation=date(2024, 1, 1))


def _stops(rows: list[dict]) -> pl.DataFrame:
    """Build a minimal stops DataFrame."""
    return pl.DataFrame(rows, schema={"stop_id": pl.Utf8, "stop_access": pl.Int64, "platform_code": pl.Utf8})


def _stops_no_access(rows: list[dict]) -> pl.DataFrame:
    """Build a stops DataFrame without stop_access column."""
    return pl.DataFrame(rows, schema={"stop_id": pl.Utf8, "platform_code": pl.Utf8})


def _pathways(rows: list[dict]) -> pl.DataFrame:
    """Build a minimal pathways DataFrame."""
    return pl.DataFrame(
        rows,
        schema={
            "csv_row_number": pl.Int64,
            "pathway_id": pl.Utf8,
            "from_stop_id": pl.Utf8,
            "to_stop_id": pl.Utf8,
        },
    )


def test_to_stop_has_external_access_yields_notice(ctx: ValidationContext) -> None:
    stops = _stops([
        {"stop_id": "stop1", "stop_access": 0, "platform_code": "platform1"},
        {"stop_id": "stop2", "stop_access": 1, "platform_code": "platform2"},
    ])
    pathways = _pathways([
        {"csv_row_number": 12, "pathway_id": "pathway12", "from_stop_id": "stop1", "to_stop_id": "stop2"},
    ])
    feed = {"stops": stops, "pathways": pathways}
    notices = validate_pathway_stop_access(feed, ctx)

    assert len(notices) == 1
    n = notices[0]
    assert n.code == "pathway_to_stop_with_access_outside_of_station_pathways"
    assert n.severity == Severity.ERROR
    assert n.fields["csvRowNumber"] == 12
    assert n.fields["platformCode"] == "platform2"
    assert n.fields["pathwayId"] == "pathway12"
    assert n.fields["stopId"] == "stop2"


def test_from_stop_has_external_access_yields_notice(ctx: ValidationContext) -> None:
    stops = _stops([
        {"stop_id": "stop1", "stop_access": 1, "platform_code": "P1"},
        {"stop_id": "stop2", "stop_access": 0, "platform_code": None},
    ])
    pathways = _pathways([
        {"csv_row_number": 5, "pathway_id": "pw5", "from_stop_id": "stop1", "to_stop_id": "stop2"},
    ])
    feed = {"stops": stops, "pathways": pathways}
    notices = validate_pathway_stop_access(feed, ctx)

    assert len(notices) == 1
    assert notices[0].fields["stopId"] == "stop1"
    assert notices[0].fields["platformCode"] == "P1"


def test_both_endpoints_external_access_yields_two_notices(ctx: ValidationContext) -> None:
    stops = _stops([
        {"stop_id": "stopA", "stop_access": 1, "platform_code": "PA"},
        {"stop_id": "stopB", "stop_access": 1, "platform_code": "PB"},
    ])
    pathways = _pathways([
        {"csv_row_number": 3, "pathway_id": "pwAB", "from_stop_id": "stopA", "to_stop_id": "stopB"},
    ])
    feed = {"stops": stops, "pathways": pathways}
    notices = validate_pathway_stop_access(feed, ctx)

    assert len(notices) == 2
    stop_ids = {n.fields["stopId"] for n in notices}
    assert stop_ids == {"stopA", "stopB"}
    assert all(n.fields["csvRowNumber"] == 3 for n in notices)


def test_same_stop_both_endpoints_yields_one_notice(ctx: ValidationContext) -> None:
    stops = _stops([
        {"stop_id": "stopX", "stop_access": 1, "platform_code": "PX"},
    ])
    pathways = _pathways([
        {"csv_row_number": 7, "pathway_id": "loop", "from_stop_id": "stopX", "to_stop_id": "stopX"},
    ])
    feed = {"stops": stops, "pathways": pathways}
    notices = validate_pathway_stop_access(feed, ctx)

    assert len(notices) == 1
    assert notices[0].fields["stopId"] == "stopX"


def test_no_stops_have_external_access_yields_no_notices(ctx: ValidationContext) -> None:
    stops = _stops([
        {"stop_id": "s1", "stop_access": 0, "platform_code": "P1"},
        {"stop_id": "s2", "stop_access": 0, "platform_code": "P2"},
    ])
    pathways = _pathways([
        {"csv_row_number": 1, "pathway_id": "pw1", "from_stop_id": "s1", "to_stop_id": "s2"},
    ])
    feed = {"stops": stops, "pathways": pathways}
    notices = validate_pathway_stop_access(feed, ctx)

    assert notices == []


def test_stop_access_null_yields_no_notices(ctx: ValidationContext) -> None:
    stops = pl.DataFrame(
        [{"stop_id": "sNull", "stop_access": None, "platform_code": "PX"}],
        schema={"stop_id": pl.Utf8, "stop_access": pl.Int64, "platform_code": pl.Utf8},
    )
    pathways = _pathways([
        {"csv_row_number": 1, "pathway_id": "pw1", "from_stop_id": "sNull", "to_stop_id": "sNull"},
    ])
    feed = {"stops": stops, "pathways": pathways}
    notices = validate_pathway_stop_access(feed, ctx)

    assert notices == []


def test_missing_stop_access_column_yields_no_notices(ctx: ValidationContext) -> None:
    stops = _stops_no_access([
        {"stop_id": "s1", "platform_code": "P1"},
    ])
    pathways = _pathways([
        {"csv_row_number": 1, "pathway_id": "pw1", "from_stop_id": "s1", "to_stop_id": "s1"},
    ])
    feed = {"stops": stops, "pathways": pathways}
    notices = validate_pathway_stop_access(feed, ctx)

    assert notices == []


def test_missing_stops_table_yields_no_notices(ctx: ValidationContext) -> None:
    pathways = _pathways([
        {"csv_row_number": 1, "pathway_id": "pw1", "from_stop_id": "s1", "to_stop_id": "s2"},
    ])
    feed = {"pathways": pathways}
    notices = validate_pathway_stop_access(feed, ctx)

    assert notices == []


def test_missing_pathways_table_yields_no_notices(ctx: ValidationContext) -> None:
    stops = _stops([
        {"stop_id": "s1", "stop_access": 1, "platform_code": "P1"},
    ])
    feed = {"stops": stops}
    notices = validate_pathway_stop_access(feed, ctx)

    assert notices == []


def test_platform_code_null_propagates_to_notice(ctx: ValidationContext) -> None:
    stops = pl.DataFrame(
        [{"stop_id": "sX", "stop_access": 1, "platform_code": None}],
        schema={"stop_id": pl.Utf8, "stop_access": pl.Int64, "platform_code": pl.Utf8},
    )
    pathways = _pathways([
        {"csv_row_number": 2, "pathway_id": "pw2", "from_stop_id": "other", "to_stop_id": "sX"},
    ])
    feed = {"stops": stops, "pathways": pathways}
    notices = validate_pathway_stop_access(feed, ctx)

    assert len(notices) == 1
    assert notices[0].fields["platformCode"] is None


def test_from_or_to_stop_id_null_is_skipped(ctx: ValidationContext) -> None:
    stops = _stops([
        {"stop_id": "sX", "stop_access": 1, "platform_code": "PX"},
    ])
    # from_stop_id is None; to_stop_id references external-access stop
    pathways = pl.DataFrame(
        [{"csv_row_number": 9, "pathway_id": "pw9", "from_stop_id": None, "to_stop_id": "sX"}],
        schema={
            "csv_row_number": pl.Int64,
            "pathway_id": pl.Utf8,
            "from_stop_id": pl.Utf8,
            "to_stop_id": pl.Utf8,
        },
    )
    feed = {"stops": stops, "pathways": pathways}
    notices = validate_pathway_stop_access(feed, ctx)

    assert len(notices) == 1
    assert notices[0].fields["stopId"] == "sX"
