"""Tests for report generation."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import polars as pl
import pytest

from gtfs_validator.config import ValidationConfig
from gtfs_validator.notices import Notice, NoticeContainer, Severity
from gtfs_validator.report import generate_reports


def test_report_json_structure(tmp_output: Path):
    config = ValidationConfig(
        gtfs_source="/test/feed.zip",
        output_directory=tmp_output,
        pretty_json=True,
        date_for_validation=date(2024, 6, 15),
    )
    feed = {
        "agency.txt": pl.DataFrame({
            "agency_id": ["A1"],
            "agency_name": ["Test Agency"],
            "agency_url": ["http://example.com"],
            "agency_phone": [None],
            "agency_email": [None],
        }),
        "stops.txt": pl.DataFrame({"stop_id": ["S1", "S2"]}),
        "routes.txt": pl.DataFrame({"route_id": ["R1"]}),
        "trips.txt": pl.DataFrame({
            "trip_id": ["T1"],
            "block_id": ["B1"],
        }),
        "shapes.txt": pl.DataFrame({"shape_id": ["SH1"]}),
    }
    notices = NoticeContainer()
    notices.add(Notice("test_notice", Severity.WARNING, {"detail": "x"}))
    features = ["Shapes"]

    generate_reports(config, feed, notices, features, 1.234)

    report_path = tmp_output / "report.json"
    assert report_path.exists()
    report = json.loads(report_path.read_text())

    assert "summary" in report
    assert "notices" in report
    assert report["summary"]["gtfsInput"] == "file:///test/feed.zip"
    assert report["summary"]["countryCode"] == "ZZ"
    assert report["summary"]["gtfsFeatures"] == ["Shapes"]
    assert report["summary"]["counts"]["Stops"] == 2
    assert report["summary"]["counts"]["Blocks"] == 1
    assert len(report["notices"]) == 1
    assert report["notices"][0]["code"] == "test_notice"
    assert report["notices"][0]["severity"] == "WARNING"


def test_system_errors_json(tmp_output: Path):
    config = ValidationConfig(
        gtfs_source="/test/feed.zip",
        output_directory=tmp_output,
    )
    notices = NoticeContainer()
    from gtfs_validator.notices import SystemError
    notices.add_system_error(SystemError(
        code="runtime_exception_in_validator",
        fields={"validatorClassName": "TestValidator", "exception": "ValueError"},
    ))

    generate_reports(config, {}, notices, [], 0.1)

    errors_path = tmp_output / "system_errors.json"
    assert errors_path.exists()
    errors = json.loads(errors_path.read_text())
    assert len(errors["notices"]) == 1
    assert errors["notices"][0]["code"] == "runtime_exception_in_validator"


def test_html_generated(tmp_output: Path):
    config = ValidationConfig(
        gtfs_source="/test/feed.zip",
        output_directory=tmp_output,
    )
    generate_reports(config, {}, NoticeContainer(), [], 0.5)
    assert (tmp_output / "report.html").exists()


def test_notice_grouping_and_capping(tmp_output: Path):
    config = ValidationConfig(
        gtfs_source="/test",
        output_directory=tmp_output,
        pretty_json=True,
    )
    notices = NoticeContainer()
    # Add more than MAX_EXPORTS_PER_TYPE notices.
    for i in range(1500):
        notices.add(Notice("many_notices", Severity.INFO, {"i": i}))

    generate_reports(config, {}, notices, [], 0.1)

    report = json.loads((tmp_output / "report.json").read_text())
    group = report["notices"][0]
    assert group["totalNotices"] == 1500
    assert len(group["sampleNotices"]) == 1000  # capped at MAX_EXPORTS_PER_TYPE
