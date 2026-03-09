"""Profile the stop_time_travel_speed validator on a GTFS input."""

from __future__ import annotations

import argparse
import cProfile
import pstats
import time
from datetime import date

import polars as pl

from gtfs_validator.context import ValidationContext, build_stop_location_cache
from gtfs_validator.input import open_input
from gtfs_validator.loading import load_feed
from gtfs_validator.validators.stop_time_travel_speed import (
    validate_stop_time_travel_speed,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run and profile stop_time_travel_speed on a GTFS feed.",
    )
    parser.add_argument("source", help="Path or URL passed to open_input.")
    parser.add_argument(
        "--storage-directory",
        default="storage",
        help="Temporary storage directory for extracted input.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="Thread count for feed loading.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=20,
        help="Number of top cumulative-time profile entries to print.",
    )
    args = parser.parse_args()

    gtfs_input, _ = open_input(args.source, args.storage_directory)
    try:
        feed, _, _ = load_feed(gtfs_input, args.threads)
    finally:
        gtfs_input.close()

    stop_location_cache = None
    stops_df = feed.get("stops")
    if isinstance(stops_df, pl.DataFrame) and not stops_df.is_empty():
        stop_location_cache = build_stop_location_cache(stops_df)

    ctx = ValidationContext(
        country_code="ZZ",
        date_for_validation=date.today(),
        stop_location_cache=stop_location_cache,
    )

    profiler = cProfile.Profile()
    t0 = time.perf_counter()
    profiler.enable()
    notices = validate_stop_time_travel_speed(feed, ctx)
    profiler.disable()
    elapsed = time.perf_counter() - t0

    print(f"elapsed_seconds={elapsed:.3f}")
    print(f"notice_count={len(notices)}")
    stats = pstats.Stats(profiler).sort_stats("cumtime")
    stats.print_stats(args.top)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
