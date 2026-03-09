"""Tests for StopTimesGeographyIdPresenceValidator."""

from datetime import date

import polars as pl

from gtfs_validator.context import ValidationContext
from gtfs_validator.notices import Severity
from gtfs_validator.validators.stop_times_geography_id_presence import (
    validate_stop_times_geography_id_presence,
)

CTX = ValidationContext(country_code="US", date_for_validation=date(2026, 3, 8))


def make_stop_times(
    rows: list[dict],
    columns: list[str] | None = None,
) -> pl.DataFrame:
    """Build a stop_times DataFrame from a list of row dicts.

    ``columns`` controls which geography columns are included in the schema.
    Defaults to ["stop_id", "location_group_id", "location_id"].
    Each row dict may supply any of these plus ``csv_row_number``.
    """
    if columns is None:
        columns = ["stop_id", "location_group_id", "location_id"]

    geo_cols = ["stop_id", "location_group_id", "location_id"]
    included_geo = [c for c in geo_cols if c in columns]

    schema: dict[str, pl.PolarsDataType] = {"csv_row_number": pl.Int64}
    for col in included_geo:
        schema[col] = pl.Utf8

    data: dict[str, list] = {col: [] for col in schema}
    for r in rows:
        data["csv_row_number"].append(r.get("csv_row_number", 2))
        for col in included_geo:
            data[col].append(r.get(col))

    return pl.DataFrame(data, schema=schema)


# ---------------------------------------------------------------------------
# Guard clause tests
# ---------------------------------------------------------------------------


def test_no_geography_columns_returns_empty() -> None:
    """stop_times has only csv_row_number — all three geography columns absent."""
    feed = {
        "stop_times": pl.DataFrame(
            {"csv_row_number": [2]},
            schema={"csv_row_number": pl.Int64},
        )
    }
    assert validate_stop_times_geography_id_presence(feed, CTX) == []


def test_empty_stop_times_returns_empty() -> None:
    """stop_times is an empty DataFrame — guard clause fires."""
    feed = {"stop_times": make_stop_times([])}
    assert validate_stop_times_geography_id_presence(feed, CTX) == []


def test_stop_times_absent_returns_empty() -> None:
    """feed has no stop_times key — guard clause fires."""
    assert validate_stop_times_geography_id_presence({}, CTX) == []


# ---------------------------------------------------------------------------
# Valid single-column cases
# ---------------------------------------------------------------------------


def test_stop_id_only_valid() -> None:
    """One row with stop_id set, other geo columns absent as columns."""
    feed = {
        "stop_times": make_stop_times(
            [{"csv_row_number": 2, "stop_id": "S1"}],
            columns=["stop_id"],
        )
    }
    assert validate_stop_times_geography_id_presence(feed, CTX) == []


def test_location_group_id_only_valid() -> None:
    """One row with location_group_id set, other geo columns absent as columns."""
    feed = {
        "stop_times": make_stop_times(
            [{"csv_row_number": 2, "location_group_id": "G1"}],
            columns=["location_group_id"],
        )
    }
    assert validate_stop_times_geography_id_presence(feed, CTX) == []


def test_location_id_only_valid() -> None:
    """One row with location_id set, other geo columns absent as columns."""
    feed = {
        "stop_times": make_stop_times(
            [{"csv_row_number": 2, "location_id": "L1"}],
            columns=["location_id"],
        )
    }
    assert validate_stop_times_geography_id_presence(feed, CTX) == []


# ---------------------------------------------------------------------------
# missing_required_field notices
# ---------------------------------------------------------------------------


def test_all_three_absent_emits_missing_required_field() -> None:
    """All three geography columns present as columns but all values are None."""
    feed = {
        "stop_times": make_stop_times(
            [{"csv_row_number": 2, "stop_id": None, "location_group_id": None, "location_id": None}],
        )
    }
    notices = validate_stop_times_geography_id_presence(feed, CTX)
    assert len(notices) == 1
    n = notices[0]
    assert n.code == "missing_required_field"
    assert n.severity == Severity.ERROR
    assert n.fields["filename"] == "stop_times.txt"
    assert n.fields["csv_row_number"] == 2
    assert n.fields["field_name"] == "stop_id"


def test_empty_string_treated_as_absent() -> None:
    """Empty strings count as absent — should emit missing_required_field."""
    feed = {
        "stop_times": make_stop_times(
            [{"csv_row_number": 2, "stop_id": "", "location_group_id": "", "location_id": ""}],
        )
    }
    notices = validate_stop_times_geography_id_presence(feed, CTX)
    assert len(notices) == 1
    assert notices[0].code == "missing_required_field"


def test_only_stop_id_column_present_null_value_emits_missing() -> None:
    """Only stop_id column exists, its value is null — should emit missing."""
    feed = {
        "stop_times": make_stop_times(
            [{"csv_row_number": 2, "stop_id": None}],
            columns=["stop_id"],
        )
    }
    notices = validate_stop_times_geography_id_presence(feed, CTX)
    assert len(notices) == 1
    n = notices[0]
    assert n.code == "missing_required_field"
    assert n.fields["field_name"] == "stop_id"


# ---------------------------------------------------------------------------
# forbidden_geography_id notices
# ---------------------------------------------------------------------------


def test_stop_id_and_location_group_id_both_set_emits_forbidden() -> None:
    """stop_id + location_group_id both set — forbidden."""
    feed = {
        "stop_times": make_stop_times(
            [{"csv_row_number": 2, "stop_id": "S1", "location_group_id": "G1"}],
            columns=["stop_id", "location_group_id"],
        )
    }
    notices = validate_stop_times_geography_id_presence(feed, CTX)
    assert len(notices) == 1
    n = notices[0]
    assert n.code == "forbidden_geography_id"
    assert n.severity == Severity.ERROR
    assert n.fields["csv_row_number"] == 2
    assert n.fields["stop_id"] == "S1"
    assert n.fields["location_group_id"] == "G1"
    assert n.fields["location_id"] is None


def test_stop_id_and_location_id_both_set_emits_forbidden() -> None:
    """stop_id + location_id both set — forbidden."""
    feed = {
        "stop_times": make_stop_times(
            [{"csv_row_number": 2, "stop_id": "S1", "location_id": "L1"}],
            columns=["stop_id", "location_id"],
        )
    }
    notices = validate_stop_times_geography_id_presence(feed, CTX)
    assert len(notices) == 1
    n = notices[0]
    assert n.code == "forbidden_geography_id"
    assert n.fields["stop_id"] == "S1"
    assert n.fields["location_id"] == "L1"
    assert n.fields["location_group_id"] is None


def test_location_group_id_and_location_id_both_set_emits_forbidden() -> None:
    """location_group_id + location_id both set — forbidden."""
    feed = {
        "stop_times": make_stop_times(
            [{"csv_row_number": 2, "location_group_id": "G1", "location_id": "L1"}],
            columns=["location_group_id", "location_id"],
        )
    }
    notices = validate_stop_times_geography_id_presence(feed, CTX)
    assert len(notices) == 1
    n = notices[0]
    assert n.code == "forbidden_geography_id"
    assert n.fields["stop_id"] is None
    assert n.fields["location_group_id"] == "G1"
    assert n.fields["location_id"] == "L1"


def test_all_three_set_emits_forbidden() -> None:
    """All three geography IDs set on the same row — forbidden."""
    feed = {
        "stop_times": make_stop_times(
            [{"csv_row_number": 2, "stop_id": "S1", "location_group_id": "G1", "location_id": "L1"}],
        )
    }
    notices = validate_stop_times_geography_id_presence(feed, CTX)
    assert len(notices) == 1
    n = notices[0]
    assert n.code == "forbidden_geography_id"
    assert n.fields["stop_id"] == "S1"
    assert n.fields["location_group_id"] == "G1"
    assert n.fields["location_id"] == "L1"


# ---------------------------------------------------------------------------
# Multi-row tests
# ---------------------------------------------------------------------------


def test_multiple_rows_mixed_validity() -> None:
    """Three rows with mixed validity: valid, missing, forbidden."""
    feed = {
        "stop_times": make_stop_times(
            [
                # Row A: valid (stop_id only set)
                {"csv_row_number": 2, "stop_id": "S1", "location_group_id": None, "location_id": None},
                # Row B: missing (all null)
                {"csv_row_number": 3, "stop_id": None, "location_group_id": None, "location_id": None},
                # Row C: forbidden (stop_id + location_group_id)
                {"csv_row_number": 4, "stop_id": "S2", "location_group_id": "G1", "location_id": None},
            ],
        )
    }
    notices = validate_stop_times_geography_id_presence(feed, CTX)
    assert len(notices) == 2

    missing_notices = [n for n in notices if n.code == "missing_required_field"]
    forbidden_notices = [n for n in notices if n.code == "forbidden_geography_id"]

    assert len(missing_notices) == 1
    assert missing_notices[0].fields["csv_row_number"] == 3

    assert len(forbidden_notices) == 1
    assert forbidden_notices[0].fields["csv_row_number"] == 4


def test_column_present_but_all_values_valid() -> None:
    """All three geography columns present; each row has exactly one geo ID set."""
    rows = [
        {"csv_row_number": 2, "stop_id": "S1", "location_group_id": None, "location_id": None},
        {"csv_row_number": 3, "stop_id": None, "location_group_id": "G1", "location_id": None},
        {"csv_row_number": 4, "stop_id": None, "location_group_id": None, "location_id": "L1"},
        {"csv_row_number": 5, "stop_id": "S2", "location_group_id": None, "location_id": None},
        {"csv_row_number": 6, "stop_id": None, "location_group_id": "G2", "location_id": None},
        {"csv_row_number": 7, "stop_id": None, "location_group_id": None, "location_id": "L2"},
        {"csv_row_number": 8, "stop_id": "S3", "location_group_id": None, "location_id": None},
        {"csv_row_number": 9, "stop_id": None, "location_group_id": "G3", "location_id": None},
        {"csv_row_number": 10, "stop_id": None, "location_group_id": None, "location_id": "L3"},
        {"csv_row_number": 11, "stop_id": "S4", "location_group_id": None, "location_id": None},
    ]
    feed = {"stop_times": make_stop_times(rows)}
    assert validate_stop_times_geography_id_presence(feed, CTX) == []
