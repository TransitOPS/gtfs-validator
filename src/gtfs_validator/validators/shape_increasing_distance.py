"""Validator: ShapeIncreasingDistanceValidator.

Checks that shape_dist_traveled values are non-decreasing along each shape,
and handles equal-distance cases by examining GPS coordinates.
"""

from __future__ import annotations

import math

import polars as pl

from gtfs_validator.context import ValidationContext
from gtfs_validator.notices import Notice, Severity

EARTH_RADIUS_METERS: float = 6_371_010.0
DISTANCE_THRESHOLD_METERS: float = 1.11


def _geodesic_distance_meters(
    lat1: float, lon1: float, lat2: float, lon2: float
) -> float:
    """Great-circle distance in metres using the haversine formula (matches S2Earth)."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a)) * EARTH_RADIUS_METERS


def validate_shape_increasing_distance(
    feed: dict[str, pl.DataFrame],
    ctx: ValidationContext,
) -> list[Notice]:
    """Validate that shape_dist_traveled is non-decreasing along each shape."""
    if "shapes" not in feed or feed["shapes"].is_empty():
        return []

    df = feed["shapes"]
    notices: list[Notice] = []

    for (shape_id,), group in df.group_by("shape_id"):
        pts = (
            group
            .sort("shape_pt_sequence")
            .select([
                "csv_row_number",
                "shape_id",
                "shape_pt_lat",
                "shape_pt_lon",
                "shape_pt_sequence",
                "shape_dist_traveled",
            ])
            .to_dicts()
        )

        for i in range(1, len(pts)):
            prev = pts[i - 1]
            curr = pts[i]

            # Skip pair if either shape_dist_traveled is null
            if prev["shape_dist_traveled"] is None or curr["shape_dist_traveled"] is None:
                continue

            prev_dist = prev["shape_dist_traveled"]
            curr_dist = curr["shape_dist_traveled"]

            # Decreasing distance → ERROR
            if prev_dist > curr_dist:
                notices.append(Notice(
                    code="decreasing_shape_distance",
                    severity=Severity.ERROR,
                    fields={
                        "shape_id": curr["shape_id"],
                        "csv_row_number": curr["csv_row_number"],
                        "shape_dist_traveled": curr_dist,
                        "shape_pt_sequence": curr["shape_pt_sequence"],
                        "prev_csv_row_number": prev["csv_row_number"],
                        "prev_shape_dist_traveled": prev_dist,
                        "prev_shape_pt_sequence": prev["shape_pt_sequence"],
                    },
                ))
                continue

            # Equal distance — examine coordinates
            if prev_dist == curr_dist:
                same_lat = curr["shape_pt_lat"] == prev["shape_pt_lat"]
                same_lon = curr["shape_pt_lon"] == prev["shape_pt_lon"]

                if same_lat and same_lon:
                    # Same coordinates → WARNING (duplicated point)
                    notices.append(Notice(
                        code="equal_shape_distance_same_coordinates",
                        severity=Severity.WARNING,
                        fields={
                            "shape_id": curr["shape_id"],
                            "csv_row_number": curr["csv_row_number"],
                            "shape_dist_traveled": curr_dist,
                            "shape_pt_sequence": curr["shape_pt_sequence"],
                            "prev_csv_row_number": prev["csv_row_number"],
                            "prev_shape_dist_traveled": prev_dist,
                            "prev_shape_pt_sequence": prev["shape_pt_sequence"],
                        },
                    ))
                else:
                    # Different coordinates — compute geodesic distance
                    distance_m = _geodesic_distance_meters(
                        prev["shape_pt_lat"], prev["shape_pt_lon"],
                        curr["shape_pt_lat"], curr["shape_pt_lon"],
                    )

                    if distance_m >= DISTANCE_THRESHOLD_METERS:
                        # >= 1.11 m → ERROR
                        notices.append(Notice(
                            code="equal_shape_distance_diff_coordinates",
                            severity=Severity.ERROR,
                            fields={
                                "shape_id": curr["shape_id"],
                                "csv_row_number": curr["csv_row_number"],
                                "shape_dist_traveled": curr_dist,
                                "shape_pt_sequence": curr["shape_pt_sequence"],
                                "prev_csv_row_number": prev["csv_row_number"],
                                "prev_shape_dist_traveled": prev_dist,
                                "prev_shape_pt_sequence": prev["shape_pt_sequence"],
                                "actual_distance_between_shape_points": distance_m,
                            },
                        ))
                    elif distance_m > 0.0:
                        # 0 < distance < 1.11 m → WARNING
                        notices.append(Notice(
                            code="equal_shape_distance_diff_coordinates_distance_below_threshold",
                            severity=Severity.WARNING,
                            fields={
                                "shape_id": curr["shape_id"],
                                "csv_row_number": curr["csv_row_number"],
                                "shape_dist_traveled": curr_dist,
                                "shape_pt_sequence": curr["shape_pt_sequence"],
                                "prev_csv_row_number": prev["csv_row_number"],
                                "prev_shape_dist_traveled": prev_dist,
                                "prev_shape_pt_sequence": prev["shape_pt_sequence"],
                                "actual_distance_between_shape_points": distance_m,
                            },
                        ))
                    # distance == 0.0 exactly → no notice

            # prev_dist < curr_dist → increasing, normal case, no action

    return notices
