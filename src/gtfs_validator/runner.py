"""Orchestration pipeline — runs the full validation sequence."""

from __future__ import annotations

import dataclasses
import logging
import pathlib
import time
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed

import polars as pl

from gtfs_validator.config import RunResult, ValidationConfig
from gtfs_validator.context import ValidationContext, build_stop_location_cache
from gtfs_validator.features import detect_features
from gtfs_validator.input import open_input
from gtfs_validator.load_validators import run_load_validators
from gtfs_validator.loading import TableStatus, load_feed
from gtfs_validator.notices import Notice, NoticeContainer, SystemError
from gtfs_validator.report import generate_reports
from gtfs_validator.validators import VALIDATOR_REGISTRY, ValidatorEntry

logger = logging.getLogger(__name__)


def _load_timezone_country_map() -> dict[str, str]:
    """Build a timezone -> ISO 3166-1 alpha-2 country code map from zone.tab.

    Falls back to an empty dict if the file is not available on the system.
    """
    candidates = [
        pathlib.Path("/usr/share/zoneinfo/zone1970.tab"),
        pathlib.Path("/usr/share/zoneinfo/zone.tab"),
    ]
    for path in candidates:
        if path.exists():
            mapping: dict[str, str] = {}
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.startswith("#") or not line.strip():
                    continue
                parts = line.split("\t")
                if len(parts) >= 3:
                    country = parts[0].strip()
                    tz_name = parts[2].strip()
                    # zone1970.tab may list multiple countries; take the first.
                    if tz_name not in mapping:
                        mapping[tz_name] = country
            return mapping
    return {}


# Loaded once at import time; the system file rarely changes.
_TZ_TO_COUNTRY: dict[str, str] = _load_timezone_country_map()


def infer_country_code(feed: dict[str, pl.DataFrame]) -> str:
    """Infer ISO 3166-1 alpha-2 country code from agency_timezone in agency.txt.

    Returns "ZZ" if the timezone cannot be mapped to a country.
    """
    ag = feed.get("agency.txt")
    if ag is None or ag.height == 0 or "agency_timezone" not in ag.columns:
        return "ZZ"
    tz_value = ag["agency_timezone"][0]
    if tz_value is None:
        return "ZZ"
    return _TZ_TO_COUNTRY.get(str(tz_value), "ZZ")


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
        t2 = time.monotonic()
        feed, table_statuses, load_notices = load_feed(
            gtfs_input, config.num_threads,
        )
        notices.merge(load_notices)
        logger.info("Stage 2 (load tables):        %.3fs", time.monotonic() - t2)

        # Stage 3: Load-time validation (per-table checks in parallel, then FK).
        t3 = time.monotonic()
        load_val_notices = run_load_validators(feed, table_statuses, config.num_threads)
        notices.merge(load_val_notices)
        logger.info("Stage 3 (load validators):    %.3fs", time.monotonic() - t3)

        # Resolve effective country code: infer from agency_timezone when not
        # explicitly provided (i.e., when the placeholder "ZZ" is still set).
        effective_country_code = config.country_code
        if effective_country_code == "ZZ":
            effective_country_code = infer_country_code(feed)
        effective_config = (
            config
            if effective_country_code == config.country_code
            else dataclasses.replace(config, country_code=effective_country_code)
        )

        # Stage 4: Multi-file validators.
        stop_location_cache = None
        stops_df = feed.get("stops")
        if isinstance(stops_df, pl.DataFrame) and not stops_df.is_empty():
            stop_location_cache = build_stop_location_cache(stops_df)

        ctx = ValidationContext(
            country_code=effective_country_code,
            date_for_validation=config.date_for_validation,
            stop_location_cache=stop_location_cache,
        )
        t4 = time.monotonic()
        _run_validators(feed, table_statuses, ctx, notices, config.num_threads)
        logger.info("Stage 4 (validators total):   %.3fs", time.monotonic() - t4)

        # Stage 5: Feature detection.
        t5 = time.monotonic()
        features = detect_features(feed)
        logger.info("Stage 5 (feature detection):  %.3fs", time.monotonic() - t5)

        # Stage 6: Report generation.
        t6 = time.monotonic()
        elapsed = time.monotonic() - start_time
        generate_reports(effective_config, feed, notices, features, elapsed)
        logger.info("Stage 6 (report generation):  %.3fs", time.monotonic() - t6)

    finally:
        # Stage 7: Cleanup.
        try:
            gtfs_input.close()
        except Exception as exc:
            notices.add_system_error(SystemError(
                code="io_error",
                fields={"message": str(exc)},
            ))

    logger.info("Total elapsed:                %.3fs", time.monotonic() - start_time)
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
    feed: Mapping[str, object],
    ctx: ValidationContext,
) -> tuple[list[Notice], list[SystemError], float]:
    """Execute a validator, catching any runtime exceptions.

    Returns notices, system errors, and elapsed seconds.
    """
    t0 = time.monotonic()
    try:
        result = entry.fn(feed, ctx)  # type: ignore[arg-type]
        return result, [], time.monotonic() - t0
    except Exception as exc:
        return [], [SystemError(
            code="runtime_exception_in_validator",
            fields={
                "validatorClassName": entry.name,
                "exception": type(exc).__name__,
                "message": str(exc),
            },
        )], time.monotonic() - t0


def _run_validators(
    feed: Mapping[str, object],
    table_statuses: dict[str, TableStatus],
    ctx: ValidationContext,
    notices: NoticeContainer,
    num_threads: int,
) -> None:
    """Run all registered multi-file validators."""
    entries = [e for e in VALIDATOR_REGISTRY if not should_skip(e, table_statuses)]

    if not entries:
        return

    timings: list[tuple[str, float]] = []

    if num_threads <= 1:
        for entry in entries:
            result_notices, result_errors, elapsed = safe_validate(entry, feed, ctx)
            timings.append((entry.name, elapsed))
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
                entry = futures[fut]
                result_notices, result_errors, elapsed = fut.result()
                timings.append((entry.name, elapsed))
                notices.add_all(result_notices)
                for err in result_errors:
                    notices.add_system_error(err)

    _log_validator_timings(timings)


def _log_validator_timings(timings: list[tuple[str, float]]) -> None:
    """Log per-validator elapsed times sorted slowest-first."""
    if not timings:
        return
    timings_sorted = sorted(timings, key=lambda t: t[1], reverse=True)
    total = sum(t for _, t in timings)
    lines = ["Validator timings (slowest first):"]
    for name, elapsed in timings_sorted:
        lines.append(f"  {elapsed:7.3f}s  {name}")
    lines.append(f"  {'total':->7}   {total:.3f}s  ({len(timings)} validators)")
    logger.info("\n".join(lines))
