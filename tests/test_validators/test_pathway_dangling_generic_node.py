"""Tests for validate_pathway_dangling_generic_node."""

from datetime import date

import polars as pl
import pytest

from gtfs_validator.context import ValidationContext
from gtfs_validator.notices import Severity
from gtfs_validator.validators.pathway_dangling_generic_node import (
    validate_pathway_dangling_generic_node,
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
}

# Shared stop rows used by most tests
# Row 10: STATION,      stop_id="stop10"
# Row 11: STOP,         stop_id="stop11", parent="stop10"
# Row 12: GENERIC_NODE, stop_id="stop12", parent="stop10"
# Row 13: ENTRANCE,     stop_id="stop13", parent="stop10"
BASE_STOPS = pl.DataFrame(
    {
        "csvRowNumber": [10, 11, 12, 13],
        "stop_id": ["stop10", "stop11", "stop12", "stop13"],
        "stop_name": ["Station", "Platform", "Node", "Entrance"],
        "location_type": [1, 0, 3, 2],
        "parent_station": [None, "stop10", "stop10", "stop10"],
    },
    schema=STOPS_SCHEMA,
)


def make_pathways(*edges: tuple[str, str]) -> pl.DataFrame:
    """Build a pathways DataFrame from (from_stop_id, to_stop_id) tuples."""
    return pl.DataFrame(
        {
            "from_stop_id": [e[0] for e in edges],
            "to_stop_id": [e[1] for e in edges],
        },
        schema=PATHWAYS_SCHEMA,
    )


def test_dangling_generic_node_yields_notice() -> None:
    """GENERIC_NODE with exactly one neighbor emits a WARNING notice."""
    feed = {
        "stops": BASE_STOPS,
        "pathways": make_pathways(("stop11", "stop13"), ("stop11", "stop12")),
    }
    notices = validate_pathway_dangling_generic_node(feed, CTX)
    assert len(notices) == 1
    n = notices[0]
    assert n.code == "pathway_dangling_generic_node"
    assert n.severity == Severity.WARNING
    assert n.fields["csvRowNumber"] == 12
    assert n.fields["stopId"] == "stop12"
    assert n.fields["stopName"] == "Node"
    assert n.fields["parentStation"] == "stop10"


def test_dangling_generic_node_with_two_pathways_yields_notice() -> None:
    """Multi-edge to same neighbor is deduplicated; still exactly one neighbor."""
    feed = {
        "stops": BASE_STOPS,
        "pathways": make_pathways(
            ("stop11", "stop13"),
            ("stop11", "stop12"),
            ("stop12", "stop11"),
        ),
    }
    notices = validate_pathway_dangling_generic_node(feed, CTX)
    assert len(notices) == 1
    assert notices[0].fields["csvRowNumber"] == 12


def test_isolated_generic_node_yields_no_notice() -> None:
    """GENERIC_NODE with no neighbors (zero) is not flagged."""
    feed = {
        "stops": BASE_STOPS,
        "pathways": make_pathways(("stop11", "stop13")),
    }
    notices = validate_pathway_dangling_generic_node(feed, CTX)
    assert notices == []


def test_pass_through_generic_node_yields_no_notice() -> None:
    """GENERIC_NODE with two distinct neighbors is not flagged."""
    feed = {
        "stops": BASE_STOPS,
        "pathways": make_pathways(("stop11", "stop12"), ("stop12", "stop13")),
    }
    notices = validate_pathway_dangling_generic_node(feed, CTX)
    assert notices == []


def test_dangling_platform_and_entrance_yields_no_notice() -> None:
    """No GENERIC_NODE rows → validator returns early, no notices."""
    stops_no_generic = pl.DataFrame(
        {
            "csvRowNumber": [10, 11, 12],
            "stop_id": ["stop10", "stop11", "stop12"],
            "stop_name": ["Station", "Platform", "Entrance"],
            "location_type": [1, 0, 2],
            "parent_station": [None, "stop10", "stop10"],
        },
        schema=STOPS_SCHEMA,
    )
    feed = {
        "stops": stops_no_generic,
        "pathways": make_pathways(("stop11", "stop12")),
    }
    notices = validate_pathway_dangling_generic_node(feed, CTX)
    assert notices == []


def test_stops_absent_yields_no_notice() -> None:
    """No stops key in feed → returns early, no notices."""
    notices = validate_pathway_dangling_generic_node({}, CTX)
    assert notices == []


def test_stops_empty_yields_no_notice() -> None:
    """Empty stops DataFrame → returns early, no notices."""
    feed = {"stops": pl.DataFrame(schema=STOPS_SCHEMA)}
    notices = validate_pathway_dangling_generic_node(feed, CTX)
    assert notices == []


def test_pathways_absent_yields_no_notice() -> None:
    """No pathways key → all generic nodes have empty neighbor sets, no notices."""
    feed = {"stops": BASE_STOPS}
    notices = validate_pathway_dangling_generic_node(feed, CTX)
    assert notices == []


def test_pathways_empty_yields_no_notice() -> None:
    """Empty pathways DataFrame → all neighbor sets remain empty, no notices."""
    feed = {
        "stops": BASE_STOPS,
        "pathways": pl.DataFrame(schema=PATHWAYS_SCHEMA),
    }
    notices = validate_pathway_dangling_generic_node(feed, CTX)
    assert notices == []


def test_stop_name_and_parent_station_null_passed_through() -> None:
    """Null stop_name and parent_station are passed through as None in notice fields."""
    stops = pl.DataFrame(
        {
            "csvRowNumber": [5],
            "stop_id": ["gn1"],
            "stop_name": [None],
            "location_type": [3],
            "parent_station": [None],
        },
        schema=STOPS_SCHEMA,
    )
    feed = {
        "stops": stops,
        "pathways": make_pathways(("stop11", "gn1")),
    }
    notices = validate_pathway_dangling_generic_node(feed, CTX)
    assert len(notices) == 1
    assert notices[0].fields["stopName"] is None
    assert notices[0].fields["parentStation"] is None


def test_multiple_generic_nodes_mixed() -> None:
    """Only the dangling subset of multiple GENERIC_NODE stops is reported."""
    stops = pl.DataFrame(
        {
            "csvRowNumber": [1, 2, 3, 4],
            "stop_id": ["sta", "gn_dangling", "gn_passthrough", "gn_isolated"],
            "stop_name": ["Station", "Dangling", "PassThrough", "Isolated"],
            "location_type": [1, 3, 3, 3],
            "parent_station": [None, None, None, None],
        },
        schema=STOPS_SCHEMA,
    )
    pathways = make_pathways(
        ("sta", "gn_dangling"),
        ("sta", "gn_passthrough"),
        ("gn_passthrough", "stop_other"),
    )
    feed = {"stops": stops, "pathways": pathways}
    notices = validate_pathway_dangling_generic_node(feed, CTX)
    assert len(notices) == 1
    assert notices[0].fields["stopId"] == "gn_dangling"


def test_output_order_matches_stops_row_order() -> None:
    """Notices are emitted in DataFrame row order, not sorted by csvRowNumber."""
    stops = pl.DataFrame(
        {
            "csvRowNumber": [30, 20, 10],
            "stop_id": ["gn30", "gn20", "gn10"],
            "stop_name": ["A", "B", "C"],
            "location_type": [3, 3, 3],
            "parent_station": [None, None, None],
        },
        schema=STOPS_SCHEMA,
    )
    pathways = make_pathways(
        ("neighbor_a", "gn30"),
        ("neighbor_b", "gn20"),
        ("neighbor_c", "gn10"),
    )
    feed = {"stops": stops, "pathways": pathways}
    notices = validate_pathway_dangling_generic_node(feed, CTX)
    assert len(notices) == 3
    assert [n.fields["csvRowNumber"] for n in notices] == [30, 20, 10]


def test_dangling_via_reverse_edge_only() -> None:
    """to_stop_id direction contributes to the GENERIC_NODE's neighbor set."""
    stops = pl.DataFrame(
        {
            "csvRowNumber": [5, 6],
            "stop_id": ["gn", "other"],
            "stop_name": ["Node", "Other"],
            "location_type": [3, 0],
            "parent_station": [None, None],
        },
        schema=STOPS_SCHEMA,
    )
    feed = {
        "stops": stops,
        "pathways": make_pathways(("other", "gn")),
    }
    notices = validate_pathway_dangling_generic_node(feed, CTX)
    assert len(notices) == 1
    assert notices[0].fields["stopId"] == "gn"


def test_pathway_referencing_non_generic_node_stop() -> None:
    """Pathway edges between non-GENERIC_NODE stops cause no notices."""
    stops = pl.DataFrame(
        {
            "csvRowNumber": [1, 2],
            "stop_id": ["stop_a", "stop_b"],
            "stop_name": ["A", "B"],
            "location_type": [0, 0],
            "parent_station": [None, None],
        },
        schema=STOPS_SCHEMA,
    )
    feed = {
        "stops": stops,
        "pathways": make_pathways(("stop_a", "stop_b")),
    }
    notices = validate_pathway_dangling_generic_node(feed, CTX)
    assert notices == []


def test_pathway_with_unknown_stop_id_still_counts_as_neighbor() -> None:
    """FK-dangling pathway endpoint still contributes to neighbor set."""
    stops = pl.DataFrame(
        {
            "csvRowNumber": [1],
            "stop_id": ["gn"],
            "stop_name": ["Node"],
            "location_type": [3],
            "parent_station": [None],
        },
        schema=STOPS_SCHEMA,
    )
    feed = {
        "stops": stops,
        "pathways": make_pathways(("gn", "nonexistent_stop")),
    }
    notices = validate_pathway_dangling_generic_node(feed, CTX)
    assert len(notices) == 1
    assert notices[0].fields["stopId"] == "gn"
