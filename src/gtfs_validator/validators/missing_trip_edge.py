"""Validator: MissingTripEdgeValidator.

Checks that the first and last stop time of every trip each define both
arrival_time and departure_time, unless the row uses a continuous
pickup/drop-off window (start_pickup_drop_off_window /
end_pickup_drop_off_window).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from gtfs_validator.notices import Notice, Severity

if TYPE_CHECKING:
    from gtfs_validator.context import ValidationContext

_REQUIRED_COLS = {
    "trip_id",
    "stop_sequence",
    "arrival_time",
    "departure_time",
    "_row_number",
}
_WINDOW_COLS = {
    "start_pickup_drop_off_window",
    "end_pickup_drop_off_window",
}


def validate_missing_trip_edge(
    feed: dict[str, pl.DataFrame],
    ctx: ValidationContext,
) -> list[Notice]:
    """Emit an error for each trip edge missing arrival_time or departure_time."""
    stop_times = feed.get("stop_times")
    if stop_times is None or stop_times.is_empty():
        return []

    available = set(stop_times.columns)
    if not _REQUIRED_COLS.issubset(available):
        return []

    select_cols = list(_REQUIRED_COLS | (_WINDOW_COLS & available))
    df = stop_times.select(select_cols).sort("stop_sequence")

    # Ensure window columns exist (fill with null if absent from the file)
    for col in _WINDOW_COLS:
        if col not in df.columns:
            df = df.with_columns(pl.lit(None).cast(pl.Utf8).alias(col))

    notices: list[Notice] = []

    for (trip_id,), group in df.group_by(["trip_id"]):
        first_row = group.row(0, named=True)
        last_row = group.row(-1, named=True)

        for edge in [first_row, last_row]:
            # Exemption: either window field non-null means skip all time checks
            if (
                edge["start_pickup_drop_off_window"] is not None
                or edge["end_pickup_drop_off_window"] is not None
            ):
                continue

            if edge["arrival_time"] is None:
                notices.append(
                    Notice(
                        code="missing_trip_edge",
                        severity=Severity.ERROR,
                        fields={
                            "csv_row_number": edge["_row_number"],
                            "stop_sequence": edge["stop_sequence"],
                            "trip_id": edge["trip_id"],
                            "specified_field": "arrival_time",
                        },
                    )
                )

            if edge["departure_time"] is None:
                notices.append(
                    Notice(
                        code="missing_trip_edge",
                        severity=Severity.ERROR,
                        fields={
                            "csv_row_number": edge["_row_number"],
                            "stop_sequence": edge["stop_sequence"],
                            "trip_id": edge["trip_id"],
                            "specified_field": "departure_time",
                        },
                    )
                )

    return notices
