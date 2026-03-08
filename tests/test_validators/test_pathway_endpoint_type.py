"""Tests for validate_pathway_endpoint_type."""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from gtfs_validator.context import ValidationContext
from gtfs_validator.notices import Severity
from gtfs_validator.validators.pathway_endpoint_type import validate_pathway_endpoint_type

CTX = ValidationContext(country_code="US", date_for_validation=date(2026, 3, 8))

STOPS_SCHEMA = {
    "stop_id": pl.Utf8,
    "location_type": pl.Int64,
    "parent_station": pl.Utf8,
}

PATHWAYS_SCHEMA = {
    "csv_row_number": pl.Int64,
    "pathway_id": pl.Utf8,
    "from_stop_id": pl.Utf8,
    "to_stop_id": pl.Utf8,
}


def make_pathway(row: int, pid: str, from_id: str, to_id: str) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "csv_row_number": [row],
            "pathway_id": [pid],
            "from_stop_id": [from_id],
            "to_stop_id": [to_id],
        },
        schema=PATHWAYS_SCHEMA,
    )


def make_stops(*tuples: tuple) -> pl.DataFrame:
    """Build a stops DataFrame from (stop_id, location_type, parent_station | None) tuples."""
    stop_ids = []
    location_types: list[int | None] = []
    parent_stations: list[str | None] = []
    for t in tuples:
        stop_ids.append(t[0])
        location_types.append(t[1])
        parent_stations.append(t[2] if len(t) > 2 else None)
    return pl.DataFrame(
        {
            "stop_id": stop_ids,
            "location_type": location_types,
            "parent_station": parent_stations,
        },
        schema=STOPS_SCHEMA,
    )


def test_platform_endpoints_yields_no_notice() -> None:
    pathway = make_pathway(2003, "pathway2003", "stop2", "stop3")
    stops = make_stops(("stop2", 0, None), ("stop3", 0, None))
    result = validate_pathway_endpoint_type({"pathways": pathway, "stops": stops}, CTX)
    assert result == []


def test_entrance_endpoints_yields_no_notice() -> None:
    pathway = make_pathway(2003, "pathway2003", "stop2", "stop3")
    stops = make_stops(("stop2", 2, None), ("stop3", 2, None))
    result = validate_pathway_endpoint_type({"pathways": pathway, "stops": stops}, CTX)
    assert result == []


def test_generic_node_endpoints_yields_no_notice() -> None:
    pathway = make_pathway(2003, "pathway2003", "stop2", "stop3")
    stops = make_stops(("stop2", 3, None), ("stop3", 3, None))
    result = validate_pathway_endpoint_type({"pathways": pathway, "stops": stops}, CTX)
    assert result == []


def test_boarding_area_endpoints_yields_no_notice() -> None:
    pathway = make_pathway(2003, "pathway2003", "stop2", "stop3")
    stops = make_stops(("stop2", 4, None), ("stop3", 4, None))
    result = validate_pathway_endpoint_type({"pathways": pathway, "stops": stops}, CTX)
    assert result == []


def test_station_endpoints_yields_two_notices() -> None:
    pathway = make_pathway(2003, "pathway2003", "stop2", "stop3")
    stops = make_stops(("stop2", 1, None), ("stop3", 1, None))
    result = validate_pathway_endpoint_type({"pathways": pathway, "stops": stops}, CTX)
    assert len(result) == 2
    assert result[0].code == "pathway_to_wrong_location_type"
    assert result[0].severity == Severity.ERROR
    assert result[0].fields == {
        "csvRowNumber": 2003,
        "pathwayId": "pathway2003",
        "fieldName": "from_stop_id",
        "stopId": "stop2",
    }
    assert result[1].code == "pathway_to_wrong_location_type"
    assert result[1].severity == Severity.ERROR
    assert result[1].fields == {
        "csvRowNumber": 2003,
        "pathwayId": "pathway2003",
        "fieldName": "to_stop_id",
        "stopId": "stop3",
    }


def test_platform_with_boarding_areas_endpoints_yields_two_notices() -> None:
    pathway = make_pathway(2003, "pathway2003", "stop2", "stop3")
    stops = make_stops(
        ("stop2", 0, None),
        ("stop3", 0, None),
        ("stop22", 4, "stop2"),
        ("stop33", 4, "stop3"),
    )
    result = validate_pathway_endpoint_type({"pathways": pathway, "stops": stops}, CTX)
    assert len(result) == 2
    assert result[0].code == "pathway_to_platform_with_boarding_areas"
    assert result[0].severity == Severity.ERROR
    assert result[0].fields == {
        "csvRowNumber": 2003,
        "pathwayId": "pathway2003",
        "fieldName": "from_stop_id",
        "stopId": "stop2",
    }
    assert result[1].code == "pathway_to_platform_with_boarding_areas"
    assert result[1].severity == Severity.ERROR
    assert result[1].fields == {
        "csvRowNumber": 2003,
        "pathwayId": "pathway2003",
        "fieldName": "to_stop_id",
        "stopId": "stop3",
    }


def test_one_station_one_platform_with_children_yields_two_notices() -> None:
    pathway = make_pathway(1, "pw1", "sta", "plat")
    stops = make_stops(("sta", 1, None), ("plat", 0, None), ("child", 0, "plat"))
    result = validate_pathway_endpoint_type({"pathways": pathway, "stops": stops}, CTX)
    assert len(result) == 2
    assert result[0].code == "pathway_to_wrong_location_type"
    assert result[0].fields["fieldName"] == "from_stop_id"
    assert result[0].fields["stopId"] == "sta"
    assert result[1].code == "pathway_to_platform_with_boarding_areas"
    assert result[1].fields["fieldName"] == "to_stop_id"
    assert result[1].fields["stopId"] == "plat"


def test_broken_fk_endpoint_silently_skipped() -> None:
    pathway = make_pathway(5, "pw5", "nonexistent", "stop2")
    stops = make_stops(("stop2", 1, None))
    result = validate_pathway_endpoint_type({"pathways": pathway, "stops": stops}, CTX)
    assert len(result) == 1
    assert result[0].code == "pathway_to_wrong_location_type"
    assert result[0].fields["fieldName"] == "to_stop_id"
    assert result[0].fields["stopId"] == "stop2"


def test_pathways_absent_yields_no_notice() -> None:
    result = validate_pathway_endpoint_type({}, CTX)
    assert result == []


def test_pathways_empty_yields_no_notice() -> None:
    stops = make_stops(("stop2", 0, None))
    feed = {
        "pathways": pl.DataFrame(schema=PATHWAYS_SCHEMA),
        "stops": stops,
    }
    result = validate_pathway_endpoint_type(feed, CTX)
    assert result == []


def test_stops_absent_yields_no_notice() -> None:
    pathway = make_pathway(1, "pw1", "stop2", "stop3")
    result = validate_pathway_endpoint_type({"pathways": pathway}, CTX)
    assert result == []


def test_stops_empty_yields_no_notice() -> None:
    pathway = make_pathway(1, "pw1", "stop2", "stop3")
    feed = {
        "pathways": pathway,
        "stops": pl.DataFrame(schema=STOPS_SCHEMA),
    }
    result = validate_pathway_endpoint_type(feed, CTX)
    assert result == []


def test_stop_with_non_boarding_area_child_still_flags_platform() -> None:
    pathway = make_pathway(1, "pw1", "plat", "entrance")
    stops = make_stops(("plat", 0, None), ("entrance", 2, "plat"))
    result = validate_pathway_endpoint_type({"pathways": pathway, "stops": stops}, CTX)
    assert len(result) == 1
    assert result[0].code == "pathway_to_platform_with_boarding_areas"
    assert result[0].fields["fieldName"] == "from_stop_id"
    assert result[0].fields["stopId"] == "plat"


def test_notice_order_matches_pathway_row_and_field_order() -> None:
    # Three pathway rows: row=10 (from=STATION, to=STOP/no-child),
    #                     row=20 (from=STOP/no-child, to=STATION),
    #                     row=30 (from=STATION, to=STATION)
    pathways = pl.DataFrame(
        {
            "csv_row_number": [10, 20, 30],
            "pathway_id": ["pw10", "pw20", "pw30"],
            "from_stop_id": ["sta", "stp", "sta"],
            "to_stop_id": ["stp", "sta", "sta"],
        },
        schema=PATHWAYS_SCHEMA,
    )
    stops = make_stops(("sta", 1, None), ("stp", 0, None))
    result = validate_pathway_endpoint_type({"pathways": pathways, "stops": stops}, CTX)
    assert len(result) == 4
    # row 10 / from_stop_id = station
    assert result[0].fields["csvRowNumber"] == 10
    assert result[0].fields["fieldName"] == "from_stop_id"
    assert result[0].code == "pathway_to_wrong_location_type"
    # row 20 / to_stop_id = station
    assert result[1].fields["csvRowNumber"] == 20
    assert result[1].fields["fieldName"] == "to_stop_id"
    assert result[1].code == "pathway_to_wrong_location_type"
    # row 30 / from_stop_id = station
    assert result[2].fields["csvRowNumber"] == 30
    assert result[2].fields["fieldName"] == "from_stop_id"
    assert result[2].code == "pathway_to_wrong_location_type"
    # row 30 / to_stop_id = station
    assert result[3].fields["csvRowNumber"] == 30
    assert result[3].fields["fieldName"] == "to_stop_id"
    assert result[3].code == "pathway_to_wrong_location_type"
