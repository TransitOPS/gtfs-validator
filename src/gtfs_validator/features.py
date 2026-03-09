"""Feature detection for GTFS feeds.

Two detection methods:
1. File-based — table exists and has >= 1 row.
2. Field-based — at least one row has a non-null, non-default value.
"""

from __future__ import annotations

import polars as pl


# ---------------------------------------------------------------------------
# File-based features: (feature_name, filename, group)
# ---------------------------------------------------------------------------

FILE_BASED_FEATURES: list[tuple[str, str, str]] = [
    ("Pathway Connections", "pathways.txt", "Pathways"),
    ("Levels", "levels.txt", "Pathways"),
    ("Transfers", "transfers.txt", "Base Add-ons"),
    ("Shapes", "shapes.txt", "Base Add-ons"),
    ("Frequencies", "frequencies.txt", "Base Add-ons"),
    ("Feed Information", "feed_info.txt", "Base Add-ons"),
    ("Attributions", "attributions.txt", "Base Add-ons"),
    ("Translations", "translations.txt", "Base Add-ons"),
    ("Fares V1", "fare_attributes.txt", "Fares"),
    ("Fare Products", "fare_products.txt", "Fares"),
    ("Fare Transfers", "fare_transfer_rules.txt", "Fares"),
    ("Booking Rules", "booking_rules.txt", "Flexible Services"),
]

# ---------------------------------------------------------------------------
# Field-based features: (feature_name, checks, group)
#   checks is a list of (filename, column) pairs — feature is present if
#   ANY of them has a non-null value.
# ---------------------------------------------------------------------------

_FieldCheck = tuple[str, str]  # (filename, column)

FIELD_BASED_FEATURES: list[tuple[str, list[_FieldCheck], str]] = [
    (
        "Route Colors",
        [("routes.txt", "route_color"), ("routes.txt", "route_text_color")],
        "Base Add-ons",
    ),
    (
        "Headsigns",
        [("trips.txt", "trip_headsign"), ("stop_times.txt", "stop_headsign")],
        "Base Add-ons",
    ),
    (
        "Stops Wheelchair Accessibility",
        [("stops.txt", "wheelchair_boarding")],
        "Accessibility",
    ),
    (
        "Trips Wheelchair Accessibility",
        [("trips.txt", "wheelchair_accessible")],
        "Accessibility",
    ),
    (
        "Text-to-Speech",
        [("stops.txt", "tts_stop_name")],
        "Accessibility",
    ),
    (
        "Bike Allowed",
        [("trips.txt", "bikes_allowed")],
        "Accessibility",
    ),
    (
        "Cars Allowed",
        [("trips.txt", "cars_allowed")],
        "Base Add-ons",
    ),
    (
        "Location Types",
        [("stops.txt", "location_type")],
        "Base Add-ons",
    ),
    (
        "Stop Access",
        [("stops.txt", "stop_access")],
        "Base Add-ons",
    ),
    (
        "In-station Traversal Time",
        [("pathways.txt", "traversal_time")],
        "Pathways",
    ),
    (
        "Pathway Signs",
        [("pathways.txt", "signposted_as"), ("pathways.txt", "reversed_signposted_as")],
        "Pathways",
    ),
    (
        "Pathway Details",
        [
            ("pathways.txt", "max_slope"),
            ("pathways.txt", "min_width"),
            ("pathways.txt", "length"),
            ("pathways.txt", "stair_count"),
        ],
        "Pathways",
    ),
    (
        "Continuous Stops",
        [
            ("routes.txt", "continuous_pickup"),
            ("routes.txt", "continuous_drop_off"),
            ("stop_times.txt", "continuous_pickup"),
            ("stop_times.txt", "continuous_drop_off"),
        ],
        "Flexible Services",
    ),
    (
        "Fare Media",
        [("fare_products.txt", "fare_media_id")],
        "Fares",
    ),
    (
        "Rider Categories",
        [("fare_products.txt", "rider_category_id")],
        "Fares",
    ),
    (
        "Route-Based Fares",
        [("fare_leg_rules.txt", "network_id")],
        "Fares",
    ),
    (
        "Time-Based Fares",
        [
            ("fare_leg_rules.txt", "from_timeframe_group_id"),
            ("fare_leg_rules.txt", "to_timeframe_group_id"),
        ],
        "Fares",
    ),
    (
        "Zone-Based Fares",
        [
            ("fare_leg_rules.txt", "from_area_id"),
            ("fare_leg_rules.txt", "to_area_id"),
        ],
        "Fares",
    ),
]


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


def _has_non_default_values(df: pl.DataFrame, column: str) -> bool:
    if column not in df.columns:
        return False
    return bool(df.select(pl.col(column).is_not_null().any()).item())


def detect_features(feed: dict[str, pl.DataFrame]) -> list[str]:
    """Return list of detected feature names (only present features)."""
    detected: list[str] = []

    # File-based detection.
    for feature_name, filename, _group in FILE_BASED_FEATURES:
        df = feed.get(filename)
        if df is not None and df.height > 0:
            detected.append(feature_name)

    # Field-based detection.
    for feature_name, checks, _group in FIELD_BASED_FEATURES:
        found = False
        for filename, column in checks:
            df = feed.get(filename)
            if df is not None and df.height > 0 and _has_non_default_values(df, column):
                found = True
                break
        if found:
            detected.append(feature_name)

    return detected
