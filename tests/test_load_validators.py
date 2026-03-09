"""Tests for schema-driven load-time checks."""

from __future__ import annotations

import polars as pl
import pytest

from gtfs_validator.load_validators import run_load_validators
from gtfs_validator.loading import TableStatus
from gtfs_validator.notices import Severity


def _notices_by_code(container):
    result = {}
    for key, ns in container.grouped_notices().items():
        for n in ns:
            result.setdefault(n.code, []).append(n)
    return result


class TestPrimaryKey:
    def test_duplicate_single_pk(self):
        feed = {
            "agency.txt": pl.DataFrame({
                "agency_id": ["A1", "A1", "A2"],
                "agency_name": ["X", "Y", "Z"],
                "agency_url": ["http://a", "http://b", "http://c"],
                "agency_timezone": ["America/New_York"] * 3,
            }),
        }
        statuses = {"agency.txt": TableStatus.PARSABLE}
        notices = run_load_validators(feed, statuses)
        by_code = _notices_by_code(notices)
        assert "duplicate_key" in by_code

    def test_no_duplicate(self):
        feed = {
            "agency.txt": pl.DataFrame({
                "agency_id": ["A1", "A2"],
                "agency_name": ["X", "Y"],
                "agency_url": ["http://a", "http://b"],
                "agency_timezone": ["America/New_York"] * 2,
            }),
        }
        statuses = {"agency.txt": TableStatus.PARSABLE}
        notices = run_load_validators(feed, statuses)
        by_code = _notices_by_code(notices)
        assert "duplicate_key" not in by_code


class TestForeignKey:
    def test_fk_violation(self):
        feed = {
            "routes.txt": pl.DataFrame({
                "route_id": ["R1"],
                "agency_id": ["NONEXISTENT"],
                "route_type": ["3"],
            }),
            "agency.txt": pl.DataFrame({
                "agency_id": ["A1"],
                "agency_name": ["Test"],
                "agency_url": ["http://a"],
                "agency_timezone": ["America/New_York"],
            }),
        }
        statuses = {
            "routes.txt": TableStatus.PARSABLE,
            "agency.txt": TableStatus.PARSABLE,
        }
        notices = run_load_validators(feed, statuses)
        by_code = _notices_by_code(notices)
        assert "foreign_key_violation" in by_code

    def test_fk_null_ignored(self):
        feed = {
            "routes.txt": pl.DataFrame({
                "route_id": ["R1"],
                "agency_id": [None],
                "route_type": ["3"],
            }),
            "agency.txt": pl.DataFrame({
                "agency_id": ["A1"],
                "agency_name": ["Test"],
                "agency_url": ["http://a"],
                "agency_timezone": ["America/New_York"],
            }),
        }
        statuses = {
            "routes.txt": TableStatus.PARSABLE,
            "agency.txt": TableStatus.PARSABLE,
        }
        notices = run_load_validators(feed, statuses)
        by_code = _notices_by_code(notices)
        assert "foreign_key_violation" not in by_code

    def test_fk_skipped_when_target_missing(self):
        feed = {
            "routes.txt": pl.DataFrame({
                "route_id": ["R1"],
                "agency_id": ["NONEXISTENT"],
                "route_type": ["3"],
            }),
            "agency.txt": pl.DataFrame({
                "agency_id": pl.Series([], dtype=pl.Utf8),
            }),
        }
        statuses = {
            "routes.txt": TableStatus.PARSABLE,
            "agency.txt": TableStatus.MISSING_FILE,
        }
        notices = run_load_validators(feed, statuses)
        by_code = _notices_by_code(notices)
        assert "foreign_key_violation" not in by_code


class TestEndRange:
    def test_out_of_order_dates(self):
        feed = {
            "calendar.txt": pl.DataFrame({
                "service_id": ["SVC1"],
                "monday": ["1"],
                "tuesday": ["1"],
                "wednesday": ["1"],
                "thursday": ["1"],
                "friday": ["1"],
                "saturday": ["0"],
                "sunday": ["0"],
                "start_date": ["20250101"],
                "end_date": ["20240101"],  # end before start
            }),
        }
        statuses = {"calendar.txt": TableStatus.PARSABLE}
        notices = run_load_validators(feed, statuses)
        by_code = _notices_by_code(notices)
        assert "start_and_end_range_out_of_order" in by_code


class TestMixedCase:
    def test_all_caps_warning(self):
        feed = {
            "agency.txt": pl.DataFrame({
                "agency_id": ["A1"],
                "agency_name": ["ALL CAPS AGENCY"],
                "agency_url": ["http://a"],
                "agency_timezone": ["America/New_York"],
            }),
        }
        statuses = {"agency.txt": TableStatus.PARSABLE}
        notices = run_load_validators(feed, statuses)
        by_code = _notices_by_code(notices)
        assert "mixed_case_recommended_field" in by_code
