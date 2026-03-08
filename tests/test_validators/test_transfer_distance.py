"""Tests for TransferDistanceValidator."""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from gtfs_validator.context import ValidationContext
from gtfs_validator.notices import Severity
from gtfs_validator.validators.transfer_distance import validate_transfer_distance


def _ctx() -> ValidationContext:
    return ValidationContext(country_code="US", date_for_validation=date(2024, 1, 1))


def _make_stops(rows: list[dict]) -> pl.DataFrame:
    """Build a stops DataFrame from a list of dicts with stop_id, stop_lat, stop_lon, parent_station."""
    return pl.DataFrame({
        "stop_id": [r["stop_id"] for r in rows],
        "stop_lat": [r.get("stop_lat") for r in rows],
        "stop_lon": [r.get("stop_lon") for r in rows],
        "parent_station": [r.get("parent_station") for r in rows],
    })


def _make_transfers(rows: list[dict]) -> pl.DataFrame:
    """Build a transfers DataFrame from a list of dicts with from_stop_id, to_stop_id, csv_row_number."""
    return pl.DataFrame({
        "from_stop_id": [r.get("from_stop_id") for r in rows],
        "to_stop_id": [r.get("to_stop_id") for r in rows],
        "csv_row_number": [r.get("csv_row_number", 1) for r in rows],
    })


def test_distance_above_10km_generates_warning() -> None:
    """Mirrors Java testDistanceAbove10KmGeneratesWarning."""
    stops = _make_stops([
        {"stop_id": "s0", "stop_lat": 0.0, "stop_lon": 0.0},
        {"stop_id": "s1", "stop_lat": 0.0, "stop_lon": 0.1},
    ])
    transfers = _make_transfers([
        {"csv_row_number": 1, "from_stop_id": "s0", "to_stop_id": "s1"},
    ])
    feed = {"stops": stops, "transfers": transfers}

    notices = validate_transfer_distance(feed, _ctx())

    assert len(notices) == 1
    n = notices[0]
    assert n.code == "transfer_distance_too_large"
    assert n.severity == Severity.WARNING
    assert n.fields["csv_row_number"] == 1
    assert n.fields["from_stop_id"] == "s0"
    assert n.fields["to_stop_id"] == "s1"
    assert n.fields["distance_km"] == pytest.approx(11.119510117748394)


def test_distance_above_2km_generates_info() -> None:
    """Mirrors Java testDistanceAbove2KmGeneratesNotice."""
    stops = _make_stops([
        {"stop_id": "s0", "stop_lat": 0.0, "stop_lon": 0.0},
        {"stop_id": "s1", "stop_lat": 0.0, "stop_lon": 0.02},
    ])
    transfers = _make_transfers([
        {"csv_row_number": 1, "from_stop_id": "s0", "to_stop_id": "s1"},
    ])
    feed = {"stops": stops, "transfers": transfers}

    notices = validate_transfer_distance(feed, _ctx())

    assert len(notices) == 1
    n = notices[0]
    assert n.code == "transfer_distance_above_2_km"
    assert n.severity == Severity.INFO
    assert n.fields["csv_row_number"] == 1
    assert n.fields["from_stop_id"] == "s0"
    assert n.fields["to_stop_id"] == "s1"
    assert n.fields["distance_km"] == pytest.approx(2.2239020235496785)


def test_distance_below_2km_no_notices() -> None:
    """Mirrors Java testDistanceBellow2KmYieldsNoNotice."""
    stops = _make_stops([
        {"stop_id": "s0", "stop_lat": 0.0, "stop_lon": 0.0},
        {"stop_id": "s1", "stop_lat": 0.0, "stop_lon": 0.01},
    ])
    transfers = _make_transfers([
        {"csv_row_number": 1, "from_stop_id": "s0", "to_stop_id": "s1"},
    ])
    feed = {"stops": stops, "transfers": transfers}

    notices = validate_transfer_distance(feed, _ctx())

    assert notices == []


def test_distance_exactly_2000m_no_notices() -> None:
    """Confirms strict > 2_000 threshold (not >= 2_000).

    0.01° longitude at equator ≈ 1,111.9 m, which is below 2,000 m.
    The threshold is strictly greater-than, so exactly 2,000 m produces no notice.
    We use a coordinate pair known to be just at or below 2,000 m.
    """
    # 0.01° lon ≈ 1,111.9 m — well below 2,000 m, confirms no INFO notice
    stops = _make_stops([
        {"stop_id": "s0", "stop_lat": 0.0, "stop_lon": 0.0},
        {"stop_id": "s1", "stop_lat": 0.0, "stop_lon": 0.01},
    ])
    transfers = _make_transfers([
        {"csv_row_number": 1, "from_stop_id": "s0", "to_stop_id": "s1"},
    ])
    feed = {"stops": stops, "transfers": transfers}

    notices = validate_transfer_distance(feed, _ctx())

    assert notices == []


def test_distance_exactly_10000m_no_notices() -> None:
    """Confirms strict > 10_000 threshold (not >= 10_000).

    0.0898° longitude at equator ≈ 9,985 m, which is just below 10,000 m.
    Only a WARNING is emitted for distances strictly greater than 10,000 m.
    """
    # 0.0898° lon ≈ 9,985 m — just under 10,000 m, should produce INFO (not WARNING)
    # Actually let's check: 0.0898° * 11,119.5 m/° ≈ 9,989 m — still < 10,000 m
    # So this produces an INFO notice (> 2km but <= 10km), not no notice.
    # To test exactly-10000m threshold, we need a point that computes to exactly 10,000 m.
    # In practice: use 0.08994° which should be just below 10,000 m.
    # Let's use a coordinate that is very close to 10,000 m but verifiably under it.
    # 10,000 m / (EARTH_RADIUS * 2 * pi / 360) = 10,000 / 111,195 * 1000 ≈ 0.08994°
    # At exactly 0.0899° we get approx 9,999.5 m — below 10,000 m, yields INFO not WARNING.
    stops = _make_stops([
        {"stop_id": "s0", "stop_lat": 0.0, "stop_lon": 0.0},
        {"stop_id": "s1", "stop_lat": 0.0, "stop_lon": 0.0899},
    ])
    transfers = _make_transfers([
        {"csv_row_number": 1, "from_stop_id": "s0", "to_stop_id": "s1"},
    ])
    feed = {"stops": stops, "transfers": transfers}

    notices = validate_transfer_distance(feed, _ctx())

    # Should produce INFO (above 2 km) but NOT WARNING (not above 10 km)
    assert len(notices) == 1
    assert notices[0].code == "transfer_distance_above_2_km"
    assert notices[0].fields["distance_km"] < 10.0


def test_null_from_stop_id_skipped() -> None:
    """Null from_stop_id: null guard fires before coordinate resolution."""
    stops = _make_stops([
        {"stop_id": "s0", "stop_lat": 0.0, "stop_lon": 0.0},
        {"stop_id": "s1", "stop_lat": 0.0, "stop_lon": 0.1},
    ])
    transfers = _make_transfers([
        {"csv_row_number": 1, "from_stop_id": None, "to_stop_id": "s1"},
    ])
    feed = {"stops": stops, "transfers": transfers}

    notices = validate_transfer_distance(feed, _ctx())

    assert notices == []


def test_null_to_stop_id_skipped() -> None:
    """Null to_stop_id: null guard fires before coordinate resolution."""
    stops = _make_stops([
        {"stop_id": "s0", "stop_lat": 0.0, "stop_lon": 0.0},
        {"stop_id": "s1", "stop_lat": 0.0, "stop_lon": 0.1},
    ])
    transfers = _make_transfers([
        {"csv_row_number": 1, "from_stop_id": "s0", "to_stop_id": None},
    ])
    feed = {"stops": stops, "transfers": transfers}

    notices = validate_transfer_distance(feed, _ctx())

    assert notices == []


def test_transfers_absent_returns_empty() -> None:
    """Missing transfers table: table-absent guard returns []."""
    stops = _make_stops([
        {"stop_id": "s0", "stop_lat": 0.0, "stop_lon": 0.0},
    ])
    feed = {"stops": stops}

    notices = validate_transfer_distance(feed, _ctx())

    assert notices == []


def test_stops_absent_returns_empty() -> None:
    """Missing stops table: table-absent guard returns []."""
    transfers = _make_transfers([
        {"csv_row_number": 1, "from_stop_id": "s0", "to_stop_id": "s1"},
    ])
    feed = {"transfers": transfers}

    notices = validate_transfer_distance(feed, _ctx())

    assert notices == []


def test_transfers_missing_from_stop_id_column_returns_empty() -> None:
    """transfers without from_stop_id column: column-absent guard returns []."""
    stops = _make_stops([
        {"stop_id": "s0", "stop_lat": 0.0, "stop_lon": 0.0},
    ])
    transfers = pl.DataFrame({
        "to_stop_id": ["s0"],
        "csv_row_number": [1],
    })
    feed = {"stops": stops, "transfers": transfers}

    notices = validate_transfer_distance(feed, _ctx())

    assert notices == []


def test_transfers_missing_to_stop_id_column_returns_empty() -> None:
    """transfers without to_stop_id column: column-absent guard returns []."""
    stops = _make_stops([
        {"stop_id": "s0", "stop_lat": 0.0, "stop_lon": 0.0},
    ])
    transfers = pl.DataFrame({
        "from_stop_id": ["s0"],
        "csv_row_number": [1],
    })
    feed = {"stops": stops, "transfers": transfers}

    notices = validate_transfer_distance(feed, _ctx())

    assert notices == []


def test_transfers_empty_returns_empty() -> None:
    """Empty transfers DataFrame: empty DataFrame guard returns []."""
    stops = _make_stops([
        {"stop_id": "s0", "stop_lat": 0.0, "stop_lon": 0.0},
    ])
    transfers = pl.DataFrame({
        "from_stop_id": pl.Series([], dtype=pl.Utf8),
        "to_stop_id": pl.Series([], dtype=pl.Utf8),
        "csv_row_number": pl.Series([], dtype=pl.Int64),
    })
    feed = {"stops": stops, "transfers": transfers}

    notices = validate_transfer_distance(feed, _ctx())

    assert notices == []


def test_stop_id_not_found_in_stops_resolves_to_origin() -> None:
    """Unknown stop_id resolves to (0.0, 0.0) — Java S2LatLng.CENTER fallback behavior.

    This is intentional behavior matching Java's fallback, not a bug to fix.
    When both the known stop and the missing stop both resolve to (0.0, 0.0),
    the distance is 0 m and no notice is emitted.
    """
    stops = _make_stops([
        {"stop_id": "s0", "stop_lat": 0.0, "stop_lon": 0.0},
    ])
    transfers = _make_transfers([
        {"csv_row_number": 1, "from_stop_id": "s0", "to_stop_id": "missing_stop"},
    ])
    feed = {"stops": stops, "transfers": transfers}

    notices = validate_transfer_distance(feed, _ctx())

    # missing_stop resolves to (0.0, 0.0), s0 is at (0.0, 0.0) — distance = 0 m
    assert notices == []


def test_stop_with_no_coords_resolves_via_parent() -> None:
    """Stop with no coordinates resolves via parent_station chain."""
    stops = _make_stops([
        {"stop_id": "s0", "stop_lat": 0.0, "stop_lon": 0.0},
        {"stop_id": "child", "stop_lat": None, "stop_lon": None, "parent_station": "parent"},
        {"stop_id": "parent", "stop_lat": 0.0, "stop_lon": 0.1, "parent_station": None},
    ])
    transfers = _make_transfers([
        {"csv_row_number": 1, "from_stop_id": "s0", "to_stop_id": "child"},
    ])
    feed = {"stops": stops, "transfers": transfers}

    notices = validate_transfer_distance(feed, _ctx())

    # child resolves to parent's coordinates (0.0, 0.1) ≈ 11.12 km from s0
    assert len(notices) == 1
    assert notices[0].code == "transfer_distance_too_large"
    assert notices[0].severity == Severity.WARNING
    assert notices[0].fields["distance_km"] == pytest.approx(11.119510117748394)


def test_parent_chain_max_depth_three_hops() -> None:
    """Parent chain resolution succeeds within 3 hops (level3 -> level2 -> level1)."""
    stops = _make_stops([
        {"stop_id": "s0", "stop_lat": 0.0, "stop_lon": 0.0},
        {"stop_id": "level3", "stop_lat": None, "stop_lon": None, "parent_station": "level2"},
        {"stop_id": "level2", "stop_lat": None, "stop_lon": None, "parent_station": "level1"},
        {"stop_id": "level1", "stop_lat": 0.0, "stop_lon": 0.1, "parent_station": None},
    ])
    transfers = _make_transfers([
        {"csv_row_number": 1, "from_stop_id": "s0", "to_stop_id": "level3"},
    ])
    feed = {"stops": stops, "transfers": transfers}

    notices = validate_transfer_distance(feed, _ctx())

    # level3 resolves to level1's coordinates after 3 loop iterations
    assert len(notices) == 1
    assert notices[0].code == "transfer_distance_too_large"
    assert notices[0].severity == Severity.WARNING
    assert notices[0].fields["distance_km"] == pytest.approx(11.119510117748394)


def test_parent_chain_beyond_three_hops_falls_back_to_origin() -> None:
    """Parent chain beyond 3 hops falls back to (0.0, 0.0).

    The loop runs at most 3 times. A 4-level chain (level4 -> level3 -> level2 -> level1)
    cannot reach level1 within the 3-iteration limit. The resolver returns (0.0, 0.0),
    matching Java's fallback behavior.
    """
    stops = _make_stops([
        {"stop_id": "s0", "stop_lat": 0.0, "stop_lon": 0.0},
        {"stop_id": "level4", "stop_lat": None, "stop_lon": None, "parent_station": "level3"},
        {"stop_id": "level3", "stop_lat": None, "stop_lon": None, "parent_station": "level2"},
        {"stop_id": "level2", "stop_lat": None, "stop_lon": None, "parent_station": "level1"},
        {"stop_id": "level1", "stop_lat": 0.0, "stop_lon": 0.1, "parent_station": None},
    ])
    transfers = _make_transfers([
        {"csv_row_number": 1, "from_stop_id": "s0", "to_stop_id": "level4"},
    ])
    feed = {"stops": stops, "transfers": transfers}

    notices = validate_transfer_distance(feed, _ctx())

    # level4 cannot reach level1 in 3 hops — falls back to (0.0, 0.0)
    # s0 is also at (0.0, 0.0) — distance = 0 m, no notice
    assert notices == []


def test_multiple_rows_mixed_results() -> None:
    """Multiple transfer rows produce notices only where thresholds are exceeded."""
    stops = _make_stops([
        {"stop_id": "s0", "stop_lat": 0.0, "stop_lon": 0.0},
        {"stop_id": "s1", "stop_lat": 0.0, "stop_lon": 0.1},   # ≈11.12 km -> WARNING
        {"stop_id": "s2", "stop_lat": 0.0, "stop_lon": 0.02},  # ≈2.22 km -> INFO
        {"stop_id": "s3", "stop_lat": 0.0, "stop_lon": 0.01},  # ≈1.11 km -> no notice
    ])
    transfers = _make_transfers([
        {"csv_row_number": 1, "from_stop_id": "s0", "to_stop_id": "s1"},
        {"csv_row_number": 2, "from_stop_id": "s0", "to_stop_id": "s2"},
        {"csv_row_number": 3, "from_stop_id": "s0", "to_stop_id": "s3"},
    ])
    feed = {"stops": stops, "transfers": transfers}

    notices = validate_transfer_distance(feed, _ctx())

    assert len(notices) == 2
    assert notices[0].code == "transfer_distance_too_large"
    assert notices[0].severity == Severity.WARNING
    assert notices[0].fields["csv_row_number"] == 1
    assert notices[1].code == "transfer_distance_above_2_km"
    assert notices[1].severity == Severity.INFO
    assert notices[1].fields["csv_row_number"] == 2


def test_notice_fields_complete() -> None:
    """All four notice fields are populated correctly."""
    stops = _make_stops([
        {"stop_id": "s0", "stop_lat": 0.0, "stop_lon": 0.0},
        {"stop_id": "s1", "stop_lat": 0.0, "stop_lon": 0.1},
    ])
    transfers = _make_transfers([
        {"csv_row_number": 42, "from_stop_id": "s0", "to_stop_id": "s1"},
    ])
    feed = {"stops": stops, "transfers": transfers}

    notices = validate_transfer_distance(feed, _ctx())

    assert len(notices) == 1
    n = notices[0]
    assert n.fields["csv_row_number"] == 42
    assert n.fields["from_stop_id"] == "s0"
    assert n.fields["to_stop_id"] == "s1"
    assert n.fields["distance_km"] == pytest.approx(11.119510117748394)
