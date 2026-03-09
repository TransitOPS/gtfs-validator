"""Tests for runner dispatch and safe_validate."""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from gtfs_validator.context import ValidationContext
from gtfs_validator.loading import TableStatus
from gtfs_validator.notices import Notice, Severity
from gtfs_validator.runner import safe_validate, should_skip
from gtfs_validator.validators import ValidatorEntry


def _good_validator(feed, ctx):
    return [Notice("test_ok", Severity.INFO, {})]


def _bad_validator(feed, ctx):
    raise ValueError("boom")


def test_should_skip_missing_dependency():
    entry = ValidatorEntry(name="test", fn=_good_validator, requires=["stops.txt"])
    assert should_skip(entry, {"stops.txt": TableStatus.MISSING_FILE})
    assert should_skip(entry, {"stops.txt": TableStatus.UNPARSABLE_ROWS})
    assert not should_skip(entry, {"stops.txt": TableStatus.PARSABLE})
    assert not should_skip(entry, {"stops.txt": TableStatus.EMPTY_FILE})


def test_safe_validate_success():
    entry = ValidatorEntry(name="good", fn=_good_validator, requires=[])
    ctx = ValidationContext(country_code="ZZ", date_for_validation=date.today())
    notices, errors, elapsed = safe_validate(entry, {}, ctx)
    assert len(notices) == 1
    assert len(errors) == 0
    assert elapsed >= 0


def test_safe_validate_exception():
    entry = ValidatorEntry(name="bad", fn=_bad_validator, requires=[])
    ctx = ValidationContext(country_code="ZZ", date_for_validation=date.today())
    notices, errors, elapsed = safe_validate(entry, {}, ctx)
    assert len(notices) == 0
    assert len(errors) == 1
    assert errors[0].code == "runtime_exception_in_validator"
    assert errors[0].fields["exception"] == "ValueError"
