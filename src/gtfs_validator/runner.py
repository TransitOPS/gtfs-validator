"""Orchestration pipeline — runs the full validation sequence."""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from gtfs_validator.config import RunResult, ValidationConfig
from gtfs_validator.context import ValidationContext
from gtfs_validator.features import detect_features
from gtfs_validator.input import open_input
from gtfs_validator.load_validators import run_load_validators
from gtfs_validator.loading import TableStatus, load_feed
from gtfs_validator.notices import Notice, NoticeContainer, Severity, SystemError
from gtfs_validator.report import generate_reports
from gtfs_validator.validators import VALIDATOR_REGISTRY, ValidatorEntry

logger = logging.getLogger(__name__)


def run(config: ValidationConfig) -> RunResult:
    """Execute the full validation pipeline.

    Stages (strict order):
    1. Open input
    2. Load tables (parallel)
    3. Load-time validation (schema-driven)
    4. Multi-file validators (parallel)
    5. Feature detection
    6. Report generation
    7. Cleanup
    """
    notices = NoticeContainer()
    start_time = time.monotonic()

    # Stage 1: Open input.
    try:
        gtfs_input, input_notices = open_input(
            config.gtfs_source, config.storage_directory,
        )
        notices.add_all(input_notices)
    except Exception as exc:
        logger.error("Failed to open input: %s", exc)
        return RunResult.EXCEPTION

    try:
        # Stage 2: Load tables.
        feed, table_statuses, load_notices = load_feed(
            gtfs_input, config.num_threads,
        )
        notices.merge(load_notices)

        # Stage 3: Load-time validation.
        load_val_notices = run_load_validators(feed, table_statuses)
        notices.merge(load_val_notices)

        # Stage 4: Multi-file validators.
        ctx = ValidationContext(
            country_code=config.country_code,
            date_for_validation=config.date_for_validation,
        )
        _run_validators(feed, table_statuses, ctx, notices, config.num_threads)

        # Stage 5: Feature detection.
        features = detect_features(feed)

        # Stage 6: Report generation.
        elapsed = time.monotonic() - start_time
        generate_reports(config, feed, notices, features, elapsed)

    finally:
        # Stage 7: Cleanup.
        try:
            gtfs_input.close()
        except Exception as exc:
            notices.add_system_error(SystemError(
                code="io_error",
                fields={"message": str(exc)},
            ))

    if notices.has_system_errors():
        return RunResult.SYSTEM_ERRORS
    return RunResult.SUCCESS


# ---------------------------------------------------------------------------
# Validator dispatch helpers
# ---------------------------------------------------------------------------


def should_skip(
    entry: ValidatorEntry,
    table_statuses: dict[str, TableStatus],
) -> bool:
    """Check if a validator should be skipped due to missing dependencies."""
    for required_file in entry.requires:
        status = table_statuses.get(required_file, TableStatus.MISSING_FILE)
        if status in (TableStatus.UNPARSABLE_ROWS, TableStatus.MISSING_FILE):
            return True
    return False


def safe_validate(
    entry: ValidatorEntry,
    feed: dict[str, object],
    ctx: ValidationContext,
) -> tuple[list[Notice], list[SystemError]]:
    """Execute a validator, catching any runtime exceptions."""
    try:
        result = entry.fn(feed, ctx)  # type: ignore[arg-type]
        return result, []
    except Exception as exc:
        return [], [SystemError(
            code="runtime_exception_in_validator",
            fields={
                "validatorClassName": entry.name,
                "exception": type(exc).__name__,
                "message": str(exc),
            },
        )]


def _run_validators(
    feed: dict[str, object],
    table_statuses: dict[str, TableStatus],
    ctx: ValidationContext,
    notices: NoticeContainer,
    num_threads: int,
) -> None:
    """Run all registered multi-file validators."""
    entries = [e for e in VALIDATOR_REGISTRY if not should_skip(e, table_statuses)]

    if not entries:
        return

    if num_threads <= 1:
        for entry in entries:
            result_notices, result_errors = safe_validate(entry, feed, ctx)
            notices.add_all(result_notices)
            for err in result_errors:
                notices.add_system_error(err)
    else:
        with ThreadPoolExecutor(max_workers=num_threads) as pool:
            futures = {
                pool.submit(safe_validate, e, feed, ctx): e
                for e in entries
            }
            for fut in as_completed(futures):
                result_notices, result_errors = fut.result()
                notices.add_all(result_notices)
                for err in result_errors:
                    notices.add_system_error(err)
