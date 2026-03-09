"""Tests for validate_pathway_reachable_location."""

from datetime import date

import polars as pl
import pytest

from gtfs_validator.context import ValidationContext
from gtfs_validator.notices import Severity
from gtfs_validator.validators.pathway_reachable_location import (
    validate_pathway_reachable_location,
)

CTX = ValidationContext(country_code="US", date_for_validation=date(2026, 3, 8))

STOPS_SCHEMA = {
    "csvRowNumber": pl.Int64,
    "stop_id": pl.Utf8,
    "stop_name": pl.Utf8,
    "location_type": pl.Int64,
    "parent_station": pl.Utf8,
}

PATHWAYS_SCHEMA = {
    "from_stop_id": pl.Utf8,
    "to_stop_id": pl.Utf8,
    "is_bidirectional": pl.Int64,
}


def _rows_to_dicts(rows, schema):
    keys = list(schema.keys())
    return [dict(zip(keys, row)) for row in rows]


def make_feed(stop_rows, pathway_rows):
    return {
        "stops": pl.DataFrame(
            _rows_to_dicts(stop_rows, STOPS_SCHEMA), schema=STOPS_SCHEMA
        ),
        "pathways": pl.DataFrame(
            _rows_to_dicts(pathway_rows, PATHWAYS_SCHEMA), schema=PATHWAYS_SCHEMA
        ),
    }


def test_unreachable_platforms_yields_notices():
    """Platforms without full entrance/exit reachability emit notices."""
    feed = make_feed(
        stop_rows=[
            (10, "station", None, 1, None),
            (11, "platform11", None, 0, "station"),
            (12, "platform12", None, 0, "station"),
            (13, "platform13", None, 0, "station"),
            (14, "entrance14", None, 2, "station"),
        ],
        pathway_rows=[
            ("platform11", "entrance14", 0),
            ("entrance14", "platform12", 0),
        ],
    )
    notices = validate_pathway_reachable_location(feed, CTX)
    assert len(notices) == 3
    by_stop = {n.fields["stopId"]: n for n in notices}
    assert set(by_stop.keys()) == {"platform11", "platform12", "platform13"}

    n11 = by_stop["platform11"]
    assert n11.code == "pathway_unreachable_location"
    assert n11.severity == Severity.ERROR
    assert n11.fields["hasEntrance"] is False
    assert n11.fields["hasExit"] is True
    assert n11.fields["csvRowNumber"] == 11
    assert n11.fields["locationType"] == 0
    assert n11.fields["parentStation"] == "station"

    n12 = by_stop["platform12"]
    assert n12.fields["hasEntrance"] is True
    assert n12.fields["hasExit"] is False
    assert n12.fields["csvRowNumber"] == 12

    n13 = by_stop["platform13"]
    assert n13.fields["hasEntrance"] is False
    assert n13.fields["hasExit"] is False
    assert n13.fields["csvRowNumber"] == 13


def test_unreachable_generic_nodes_yields_notices():
    """Generic nodes without full entrance/exit reachability emit notices."""
    feed = make_feed(
        stop_rows=[
            (10, "station", None, 1, None),
            (11, "node11", None, 3, "station"),
            (12, "node12", None, 3, "station"),
            (13, "node13", None, 3, "station"),
            (14, "entrance14", None, 2, "station"),
        ],
        pathway_rows=[
            ("node11", "entrance14", 0),
            ("entrance14", "node12", 0),
        ],
    )
    notices = validate_pathway_reachable_location(feed, CTX)
    assert len(notices) == 3
    by_stop = {n.fields["stopId"]: n for n in notices}
    assert set(by_stop.keys()) == {"node11", "node12", "node13"}

    assert by_stop["node11"].fields["hasEntrance"] is False
    assert by_stop["node11"].fields["hasExit"] is True
    assert by_stop["node11"].fields["locationType"] == 3

    assert by_stop["node12"].fields["hasEntrance"] is True
    assert by_stop["node12"].fields["hasExit"] is False

    assert by_stop["node13"].fields["hasEntrance"] is False
    assert by_stop["node13"].fields["hasExit"] is False


def test_reachable_platform_yields_no_notice():
    """A platform connected bidirectionally via two unidirectional pathways is reachable."""
    feed = make_feed(
        stop_rows=[
            (10, "station", None, 1, None),
            (11, "platform11", None, 0, "station"),
            (12, "entrance12", None, 2, "station"),
        ],
        pathway_rows=[
            ("platform11", "entrance12", 0),
            ("entrance12", "platform11", 0),
        ],
    )
    notices = validate_pathway_reachable_location(feed, CTX)
    assert notices == []


def test_bidirectional_pathway_yields_no_notice():
    """A single bidirectional pathway makes the platform fully reachable."""
    feed = make_feed(
        stop_rows=[
            (10, "station", None, 1, None),
            (11, "platform11", None, 0, "station"),
            (12, "entrance12", None, 2, "station"),
        ],
        pathway_rows=[
            ("platform11", "entrance12", 1),
        ],
    )
    notices = validate_pathway_reachable_location(feed, CTX)
    assert notices == []


def test_unidirectional_pathway_yields_notice():
    """A platform→entrance unidirectional pathway means no entrance can reach the platform."""
    feed = make_feed(
        stop_rows=[
            (10, "station", None, 1, None),
            (11, "platform11", None, 0, "station"),
            (12, "entrance12", None, 2, "station"),
        ],
        pathway_rows=[
            ("platform11", "entrance12", 0),
        ],
    )
    notices = validate_pathway_reachable_location(feed, CTX)
    assert len(notices) == 1
    n = notices[0]
    assert n.fields["stopId"] == "platform11"
    assert n.fields["hasEntrance"] is False
    assert n.fields["hasExit"] is True


def test_reachable_boarding_areas_yields_no_notice():
    """Reachable boarding areas under a platform with children emit no notice."""
    feed = make_feed(
        stop_rows=[
            (10, "station", None, 1, None),
            (11, "platform11", None, 0, "station"),
            (12, "boarding_area12", None, 4, "platform11"),
            (13, "boarding_area13", None, 4, "platform11"),
            (14, "entrance14", None, 2, "station"),
        ],
        pathway_rows=[
            ("boarding_area12", "entrance14", 1),
            ("boarding_area13", "entrance14", 1),
        ],
    )
    notices = validate_pathway_reachable_location(feed, CTX)
    assert notices == []


def test_unreachable_boarding_area_yields_notice():
    """A boarding area without any pathway connection emits a notice."""
    feed = make_feed(
        stop_rows=[
            (10, "station", None, 1, None),
            (11, "platform11", None, 0, "station"),
            (12, "boarding_area12", None, 4, "platform11"),
            (13, "boarding_area13", None, 4, "platform11"),
            (14, "entrance14", None, 2, "station"),
        ],
        pathway_rows=[
            ("boarding_area12", "entrance14", 1),
        ],
    )
    notices = validate_pathway_reachable_location(feed, CTX)
    assert len(notices) == 1
    n = notices[0]
    assert n.fields["stopId"] == "boarding_area13"
    assert n.fields["hasEntrance"] is False
    assert n.fields["hasExit"] is False
    assert n.fields["locationType"] == 4


def test_station_without_pathways_yields_no_notice():
    """A station with no pathway-connected children is skipped entirely."""
    feed = make_feed(
        stop_rows=[
            (1, "stationA", None, 1, None),
            (2, "platform2", None, 0, "stationA"),
            (3, "entrance3", None, 2, "stationA"),
            (4, "stationB", None, 1, None),
            (5, "platform5", None, 0, "stationB"),
        ],
        pathway_rows=[
            ("platform2", "entrance3", 1),
        ],
    )
    notices = validate_pathway_reachable_location(feed, CTX)
    stop_ids = {n.fields["stopId"] for n in notices}
    assert "platform5" not in stop_ids


def test_stop_without_parent_chain_yields_no_notice():
    """A lone stop with no parent station is not subject to the check."""
    feed = make_feed(
        stop_rows=[
            (1, "island", None, 0, None),
        ],
        pathway_rows=[
            ("island", "island", 1),
        ],
    )
    notices = validate_pathway_reachable_location(feed, CTX)
    assert notices == []


def test_pathways_absent_yields_no_notice():
    """Early return when pathways table is absent from the feed."""
    stops = pl.DataFrame(
        [{"csvRowNumber": 1, "stop_id": "platform1", "stop_name": None, "location_type": 0, "parent_station": "station"}],
        schema=STOPS_SCHEMA,
    )
    feed = {"stops": stops}
    notices = validate_pathway_reachable_location(feed, CTX)
    assert notices == []


def test_pathways_empty_yields_no_notice():
    """Empty pathways table means no station has pathways — all stops skipped."""
    feed = make_feed(
        stop_rows=[
            (10, "station", None, 1, None),
            (11, "platform11", None, 0, "station"),
            (12, "entrance12", None, 2, "station"),
        ],
        pathway_rows=[],
    )
    notices = validate_pathway_reachable_location(feed, CTX)
    assert notices == []


def test_stops_absent_yields_no_notice():
    """Early return when stops table is absent from the feed."""
    pathways = pl.DataFrame(
        [{"from_stop_id": "A", "to_stop_id": "B", "is_bidirectional": 1}],
        schema=PATHWAYS_SCHEMA,
    )
    feed = {"pathways": pathways}
    notices = validate_pathway_reachable_location(feed, CTX)
    assert notices == []


def test_entrance_yields_no_notice():
    """Entrances are not eligible for the reachability notice."""
    feed = make_feed(
        stop_rows=[
            (1, "station", None, 1, None),
            (2, "entrance2", None, 2, "station"),
        ],
        pathway_rows=[
            ("entrance2", "entrance2", 1),
        ],
    )
    notices = validate_pathway_reachable_location(feed, CTX)
    assert notices == []


def test_station_stop_yields_no_notice():
    """Stations (location_type=1) are never eligible notice targets."""
    feed = make_feed(
        stop_rows=[
            (1, "station1", None, 1, None),
            (2, "entrance2", None, 2, "station1"),
        ],
        pathway_rows=[
            ("entrance2", "station1", 1),
        ],
    )
    notices = validate_pathway_reachable_location(feed, CTX)
    assert notices == []
