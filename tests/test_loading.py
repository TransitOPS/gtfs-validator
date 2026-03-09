"""Tests for CSV loading and schema enforcement."""

from __future__ import annotations

import io
from pathlib import Path

import polars as pl
import pytest

from gtfs_validator.input import DirectoryInput
from gtfs_validator.loading import TableStatus, load_feed
from gtfs_validator.notices import Severity


class TestLoadFeed:
    def test_load_small_feed(self, small_feed_dir: Path):
        with DirectoryInput(small_feed_dir) as inp:
            feed, statuses, notices = load_feed(inp)

        assert statuses["agency.txt"] == TableStatus.PARSABLE
        assert statuses["stops.txt"] == TableStatus.PARSABLE
        assert statuses["routes.txt"] == TableStatus.PARSABLE
        assert statuses["trips.txt"] == TableStatus.PARSABLE
        assert statuses["stop_times.txt"] == TableStatus.PARSABLE
        assert statuses["calendar.txt"] == TableStatus.PARSABLE

        assert feed["agency.txt"].height == 1
        assert feed["stops.txt"].height == 2
        assert feed["routes.txt"].height == 1
        assert feed["trips.txt"].height == 1
        assert feed["stop_times.txt"].height == 2
        assert feed["calendar.txt"].height == 1

    def test_missing_required_file(self, tmp_path: Path):
        """A feed with no files should emit missing_required_file notices."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        with DirectoryInput(empty_dir) as inp:
            feed, statuses, notices = load_feed(inp)

        assert statuses["agency.txt"] == TableStatus.MISSING_FILE
        error_codes = [
            n.code for key, ns in notices.grouped_notices().items() for n in ns
            if n.severity == Severity.ERROR
        ]
        assert "missing_required_file" in error_codes

    def test_empty_file(self, tmp_path: Path):
        feed_dir = tmp_path / "feed"
        feed_dir.mkdir()
        # agency.txt with header only, no data rows.
        (feed_dir / "agency.txt").write_text(
            "agency_id,agency_name,agency_url,agency_timezone\n"
        )
        with DirectoryInput(feed_dir) as inp:
            feed, statuses, notices = load_feed(inp)

        assert statuses["agency.txt"] == TableStatus.EMPTY_FILE

    def test_unknown_file_notice(self, tmp_path: Path):
        feed_dir = tmp_path / "feed"
        feed_dir.mkdir()
        (feed_dir / "mystery.txt").write_text("col1,col2\na,b\n")
        with DirectoryInput(feed_dir) as inp:
            feed, statuses, notices = load_feed(inp)

        codes = [n.code for key, ns in notices.grouped_notices().items() for n in ns]
        assert "unknown_file" in codes

    def test_header_validation(self, tmp_path: Path):
        feed_dir = tmp_path / "feed"
        feed_dir.mkdir()
        # agency.txt missing required columns.
        (feed_dir / "agency.txt").write_text(
            "agency_id,extra_col\n"
            "A1,foo\n"
        )
        with DirectoryInput(feed_dir) as inp:
            feed, statuses, notices = load_feed(inp)

        codes = [n.code for key, ns in notices.grouped_notices().items() for n in ns]
        assert "missing_required_column" in codes
        assert "unknown_column" in codes

    def test_whitespace_trimming(self, tmp_path: Path):
        feed_dir = tmp_path / "feed"
        feed_dir.mkdir()
        (feed_dir / "agency.txt").write_text(
            "agency_id,agency_name,agency_url,agency_timezone\n"
            " A1 , Test Agency ,http://example.com,America/New_York\n"
        )
        with DirectoryInput(feed_dir) as inp:
            feed, statuses, notices = load_feed(inp)

        df = feed["agency.txt"]
        assert df["agency_id"][0] == "A1"  # trimmed
        assert df["agency_name"][0] == "Test Agency"  # trimmed

        codes = [n.code for key, ns in notices.grouped_notices().items() for n in ns]
        assert "leading_or_trailing_whitespaces" in codes
