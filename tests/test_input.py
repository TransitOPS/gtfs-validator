"""Tests for GTFS input handling."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest

from gtfs_validator.input import DirectoryInput, ZipInput, open_input


class TestDirectoryInput:
    def test_filenames(self, small_feed_dir: Path):
        with DirectoryInput(small_feed_dir) as inp:
            names = inp.filenames()
        assert "agency.txt" in names
        assert "stops.txt" in names

    def test_open_file(self, small_feed_dir: Path):
        with DirectoryInput(small_feed_dir) as inp:
            data = inp.open_file("agency.txt").read()
        assert b"agency_id" in data


class TestZipInput:
    def test_filenames(self, small_feed_zip: Path):
        with ZipInput(small_feed_zip) as inp:
            names = inp.filenames()
        assert "agency.txt" in names

    def test_filters_macosx(self, tmp_path: Path):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("agency.txt", "agency_id\nA1\n")
            zf.writestr("__MACOSX/agency.txt", "junk")
            zf.writestr(".DS_Store", "junk")
        buf.seek(0)
        notices: list = []
        with ZipInput(buf, notices=notices) as inp:
            assert "__MACOSX/agency.txt" not in inp.filenames()
            assert ".DS_Store" not in inp.filenames()
            assert "agency.txt" in inp.filenames()

    def test_subdirectory_notice(self, tmp_path: Path):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("feed/agency.txt", "agency_id\nA1\n")
        buf.seek(0)
        notices: list = []
        with ZipInput(buf, notices=notices) as inp:
            assert "agency.txt" in inp.filenames()
        assert any(n.code == "subdirectory_transit_feed" for n in notices)

    def test_zip_slip_rejected(self, tmp_path: Path):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("../evil.txt", "bad")
            zf.writestr("agency.txt", "agency_id\nA1\n")
        buf.seek(0)
        with ZipInput(buf) as inp:
            assert "../evil.txt" not in inp.filenames()
            assert "evil.txt" not in inp.filenames()


class TestOpenInput:
    def test_open_directory(self, small_feed_dir: Path):
        inp, notices = open_input(str(small_feed_dir))
        assert "agency.txt" in inp.filenames()
        inp.close()

    def test_open_zip(self, small_feed_zip: Path):
        inp, notices = open_input(str(small_feed_zip))
        assert "agency.txt" in inp.filenames()
        inp.close()
