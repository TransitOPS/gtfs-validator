"""Tests for MissingLevelIdValidator."""

from datetime import date

import polars as pl
import pytest

from gtfs_validator.context import ValidationContext
from gtfs_validator.notices import Severity
from gtfs_validator.validators.missing_level_id import validate_missing_level_id

CTX = ValidationContext(country_code="US", date_for_validation=date(2026, 3, 8))


def make_stops(
    rows: list[tuple[str, str | None, str | None]],
    row_start: int = 2,
) -> pl.DataFrame:
    """Build a stops DataFrame. Each row is (stop_id, stop_name, level_id)."""
    return pl.DataFrame(
        {
            "csv_row_number": list(range(row_start, row_start + len(rows))),
            "stop_id": [r[0] for r in rows],
            "stop_name": [r[1] for r in rows],
            "level_id": [r[2] for r in rows],
        },
        schema={
            "csv_row_number": pl.Int64,
            "stop_id": pl.Utf8,
            "stop_name": pl.Utf8,
            "level_id": pl.Utf8,
        },
    )


def make_pathways(
    rows: list[tuple[str, str, int]],
    row_start: int = 2,
) -> pl.DataFrame:
    """Build a pathways DataFrame. Each row is (from_stop_id, to_stop_id, pathway_mode)."""
    return pl.DataFrame(
        {
            "csv_row_number": list(range(row_start, row_start + len(rows))),
            "from_stop_id": [r[0] for r in rows],
            "to_stop_id": [r[1] for r in rows],
            "pathway_mode": [r[2] for r in rows],
        },
        schema={
            "csv_row_number": pl.Int64,
            "from_stop_id": pl.Utf8,
            "to_stop_id": pl.Utf8,
            "pathway_mode": pl.Int64,
        },
    )


def test_elevator_no_level_id_yields_notice() -> None:
    stops = make_stops([("s1", "Stop 1", None), ("s2", "Stop 2", None)])
    pathways = make_pathways([("s1", "s2", 5)])
    feed = {"stops": stops, "pathways": pathways}

    notices = validate_missing_level_id(feed, CTX)

    assert len(notices) == 2
    for n in notices:
        assert n.code == "missing_level_id"
        assert n.severity == Severity.ERROR
    assert {n.fields["stopId"] for n in notices} == {"s1", "s2"}


def test_elevator_no_level_id_reports_stop_only_once() -> None:
    stops = make_stops(
        [("s1", "Stop 1", None), ("s2", "Stop 2", None), ("s3", "Stop 3", None)]
    )
    # s2 appears in both pathways
    pathways = make_pathways([("s1", "s2", 5), ("s2", "s3", 5)])
    feed = {"stops": stops, "pathways": pathways}

    notices = validate_missing_level_id(feed, CTX)

    assert len(notices) == 3
    assert {n.fields["stopId"] for n in notices} == {"s1", "s2", "s3"}


def test_elevator_level_id_provided_no_notice() -> None:
    stops = make_stops([("s1", "Stop 1", "L1"), ("s2", "Stop 2", "L2")])
    pathways = make_pathways([("s1", "s2", 5)])
    feed = {"stops": stops, "pathways": pathways}

    notices = validate_missing_level_id(feed, CTX)

    assert notices == []


def test_walkway_no_level_id_no_notice() -> None:
    stops = make_stops([("s1", "Stop 1", None), ("s2", "Stop 2", None)])
    pathways = make_pathways([("s1", "s2", 1)])
    feed = {"stops": stops, "pathways": pathways}

    notices = validate_missing_level_id(feed, CTX)

    assert notices == []


def test_walkway_level_id_provided_no_notice() -> None:
    stops = make_stops([("s1", "Stop 1", "L1"), ("s2", "Stop 2", "L2")])
    pathways = make_pathways([("s1", "s2", 1)])
    feed = {"stops": stops, "pathways": pathways}

    notices = validate_missing_level_id(feed, CTX)

    assert notices == []


def test_stairs_no_level_id_no_notice() -> None:
    stops = make_stops([("s1", "Stop 1", None), ("s2", "Stop 2", None)])
    pathways = make_pathways([("s1", "s2", 2)])
    feed = {"stops": stops, "pathways": pathways}

    notices = validate_missing_level_id(feed, CTX)

    assert notices == []


def test_stairs_level_id_provided_no_notice() -> None:
    stops = make_stops([("s1", "Stop 1", "L1"), ("s2", "Stop 2", "L2")])
    pathways = make_pathways([("s1", "s2", 2)])
    feed = {"stops": stops, "pathways": pathways}

    notices = validate_missing_level_id(feed, CTX)

    assert notices == []


def test_moving_sidewalk_no_level_id_no_notice() -> None:
    stops = make_stops([("s1", "Stop 1", None), ("s2", "Stop 2", None)])
    pathways = make_pathways([("s1", "s2", 3)])
    feed = {"stops": stops, "pathways": pathways}

    notices = validate_missing_level_id(feed, CTX)

    assert notices == []


def test_moving_sidewalk_level_id_provided_no_notice() -> None:
    stops = make_stops([("s1", "Stop 1", "L1"), ("s2", "Stop 2", "L2")])
    pathways = make_pathways([("s1", "s2", 3)])
    feed = {"stops": stops, "pathways": pathways}

    notices = validate_missing_level_id(feed, CTX)

    assert notices == []


def test_escalator_no_level_id_no_notice() -> None:
    stops = make_stops([("s1", "Stop 1", None), ("s2", "Stop 2", None)])
    pathways = make_pathways([("s1", "s2", 4)])
    feed = {"stops": stops, "pathways": pathways}

    notices = validate_missing_level_id(feed, CTX)

    assert notices == []


def test_escalator_level_id_provided_no_notice() -> None:
    stops = make_stops([("s1", "Stop 1", "L1"), ("s2", "Stop 2", "L2")])
    pathways = make_pathways([("s1", "s2", 4)])
    feed = {"stops": stops, "pathways": pathways}

    notices = validate_missing_level_id(feed, CTX)

    assert notices == []


def test_fare_gate_no_level_id_no_notice() -> None:
    stops = make_stops([("s1", "Stop 1", None), ("s2", "Stop 2", None)])
    pathways = make_pathways([("s1", "s2", 6)])
    feed = {"stops": stops, "pathways": pathways}

    notices = validate_missing_level_id(feed, CTX)

    assert notices == []


def test_fare_gate_level_id_provided_no_notice() -> None:
    stops = make_stops([("s1", "Stop 1", "L1"), ("s2", "Stop 2", "L2")])
    pathways = make_pathways([("s1", "s2", 6)])
    feed = {"stops": stops, "pathways": pathways}

    notices = validate_missing_level_id(feed, CTX)

    assert notices == []


def test_exit_gate_no_level_id_no_notice() -> None:
    stops = make_stops([("s1", "Stop 1", None), ("s2", "Stop 2", None)])
    pathways = make_pathways([("s1", "s2", 7)])
    feed = {"stops": stops, "pathways": pathways}

    notices = validate_missing_level_id(feed, CTX)

    assert notices == []


def test_exit_gate_level_id_provided_no_notice() -> None:
    stops = make_stops([("s1", "Stop 1", "L1"), ("s2", "Stop 2", "L2")])
    pathways = make_pathways([("s1", "s2", 7)])
    feed = {"stops": stops, "pathways": pathways}

    notices = validate_missing_level_id(feed, CTX)

    assert notices == []


def test_pathways_absent_no_notices() -> None:
    feed = {"stops": make_stops([("s1", "Stop 1", None)])}

    notices = validate_missing_level_id(feed, CTX)

    assert notices == []


def test_stops_absent_no_notices() -> None:
    feed = {"pathways": make_pathways([("s1", "s2", 5)])}

    notices = validate_missing_level_id(feed, CTX)

    assert notices == []


def test_pathways_empty_no_notices() -> None:
    feed = {
        "pathways": make_pathways([]),
        "stops": make_stops([("s1", "Stop 1", None)]),
    }

    notices = validate_missing_level_id(feed, CTX)

    assert notices == []


def test_stops_empty_no_notices() -> None:
    feed = {
        "pathways": make_pathways([("s1", "s2", 5)]),
        "stops": make_stops([]),
    }

    notices = validate_missing_level_id(feed, CTX)

    assert notices == []


def test_dangling_stop_id_no_notice() -> None:
    stops = make_stops([("s1", "Stop 1", None)])
    pathways = make_pathways([("s1", "s_dangling", 5)])
    feed = {"stops": stops, "pathways": pathways}

    notices = validate_missing_level_id(feed, CTX)

    assert len(notices) == 1
    assert notices[0].fields["stopId"] == "s1"


def test_mixed_pathway_modes_only_elevator_triggers_notice() -> None:
    stops = make_stops(
        [("sa", "Stop A", None), ("sb", "Stop B", None), ("sc", "Stop C", None)]
    )
    # pathway A: sa->sb is WALKWAY (mode=1); pathway B: sb->sc is ELEVATOR (mode=5)
    pathways = make_pathways([("sa", "sb", 1), ("sb", "sc", 5)])
    feed = {"stops": stops, "pathways": pathways}

    notices = validate_missing_level_id(feed, CTX)

    assert len(notices) == 2
    assert {n.fields["stopId"] for n in notices} == {"sb", "sc"}


def test_notice_fields_csv_row_number_and_stop_name() -> None:
    stops = make_stops([("s1", "Platform 5", None)], row_start=7)
    # s2 is not in stops (dangling FK)
    pathways = make_pathways([("s1", "s2", 5)])
    feed = {"stops": stops, "pathways": pathways}

    notices = validate_missing_level_id(feed, CTX)

    assert len(notices) == 1
    n = notices[0]
    assert n.fields["csvRowNumber"] == 7
    assert n.fields["stopId"] == "s1"
    assert n.fields["stopName"] == "Platform 5"
