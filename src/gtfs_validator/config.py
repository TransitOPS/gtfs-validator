"""Immutable validation configuration and run-result enum."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path


class RunResult(enum.IntEnum):
    """Maps to process exit codes."""

    SUCCESS = 0
    SYSTEM_ERRORS = 1
    EXCEPTION = 2


@dataclass(frozen=True)
class ValidationConfig:
    """Immutable configuration built from CLI arguments.

    Constructed once, passed through the entire pipeline.
    """

    gtfs_source: str
    output_directory: Path
    storage_directory: Path | None = None
    validation_report_name: str = "report.json"
    html_report_name: str = "report.html"
    system_errors_report_name: str = "system_errors.json"
    num_threads: int = 1
    country_code: str = "ZZ"
    date_for_validation: date = field(default_factory=date.today)
    pretty_json: bool = False
