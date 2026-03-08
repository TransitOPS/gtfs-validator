"""Validator: route name checks for routes.txt."""

from __future__ import annotations

import re

import polars as pl

from gtfs_validator.context import ValidationContext
from gtfs_validator.notices import Notice, Severity

MAX_SHORT_NAME_LENGTH = 12
LONG_NAME_CONTAINS_SHORT_NAME_RE = re.compile(r"^\s?[\s\-\(\)].*")


def validate_route_name(
    feed: dict[str, pl.DataFrame],
    ctx: ValidationContext,
) -> list[Notice]:
    """Check route name validity for each row in routes.txt."""
    if "routes" not in feed or feed["routes"].is_empty():
        return []

    routes = feed["routes"]
    notices: list[Notice] = []

    for row in routes.iter_rows(named=True):
        short = row["route_short_name"]   # str | None
        long_ = row["route_long_name"]    # str | None
        desc = row["route_desc"]          # str | None
        route_id = row["route_id"]
        csv_row = row["csv_row_number"]

        # Check 1: both names missing
        if short is None and long_ is None:
            notices.append(Notice(
                code="route_both_short_and_long_name_missing",
                severity=Severity.ERROR,
                fields={"route_id": route_id, "csv_row_number": csv_row},
            ))

        # Check 2: short name too long
        if short is not None and len(short) > MAX_SHORT_NAME_LENGTH:
            notices.append(Notice(
                code="route_short_name_too_long",
                severity=Severity.WARNING,
                fields={
                    "route_id": route_id,
                    "csv_row_number": csv_row,
                    "route_short_name": short,
                },
            ))

        # Check 3: long name contains short name as prefix
        if short is not None and long_ is not None:
            if long_.lower().startswith(short.lower()):
                remainder = long_[len(short):]
                if remainder == "" or LONG_NAME_CONTAINS_SHORT_NAME_RE.match(remainder):
                    notices.append(Notice(
                        code="route_long_name_contains_short_name",
                        severity=Severity.WARNING,
                        fields={
                            "route_id": route_id,
                            "csv_row_number": csv_row,
                            "route_short_name": short,
                            "route_long_name": long_,
                        },
                    ))

        # Check 4: description duplicates a name
        if desc is not None:
            # Sub-check 4a: desc vs. short name (takes precedence)
            if short is not None and desc.lower() == short.lower():
                notices.append(Notice(
                    code="same_name_and_description_for_route",
                    severity=Severity.WARNING,
                    fields={
                        "csv_row_number": csv_row,
                        "route_id": route_id,
                        "route_desc": desc,
                        "specified_field": "route_short_name",
                    },
                ))
                continue  # suppress sub-check 4b for this row

            # Sub-check 4b: desc vs. long name (only if 4a did not fire)
            if long_ is not None and desc.lower() == long_.lower():
                notices.append(Notice(
                    code="same_name_and_description_for_route",
                    severity=Severity.WARNING,
                    fields={
                        "csv_row_number": csv_row,
                        "route_id": route_id,
                        "route_desc": desc,
                        "specified_field": "route_long_name",
                    },
                ))

    return notices
