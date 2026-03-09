"""GeoJSON geometry validator using Shapely for topological validation."""

from __future__ import annotations

import polars as pl
from shapely.errors import GEOSException
from shapely.geometry import MultiPolygon, Polygon
from shapely.validation import explain_validity

from gtfs_validator.context import ValidationContext
from gtfs_validator.notices import Notice, Severity


def _validate_polygon(
    rings: list[list[list[float]]],
    feature_id: str,
    feature_index: int,
    geometry_type: str,
) -> tuple[object | None, list[Notice]]:
    """Construct and validate a single Polygon from GeoJSON coordinate rings.

    Returns (Polygon, []) on success or (None, [notice]) on failure.
    """
    try:
        shell = rings[0]
        holes = rings[1:] if len(rings) > 1 else None
        polygon = Polygon(shell, holes)
    except (GEOSException, ValueError) as exc:
        return (
            None,
            [
                Notice(
                    code="invalid_geometry",
                    severity=Severity.ERROR,
                    fields={
                        "feature_id": feature_id,
                        "feature_index": feature_index,
                        "geometry_type": geometry_type,
                        "message": str(exc),
                    },
                )
            ],
        )
    if not polygon.is_valid:
        return (
            None,
            [
                Notice(
                    code="invalid_geometry",
                    severity=Severity.ERROR,
                    fields={
                        "feature_id": feature_id,
                        "feature_index": feature_index,
                        "geometry_type": geometry_type,
                        "message": explain_validity(polygon),
                    },
                )
            ],
        )
    return (polygon, [])


def _validate_multi_polygon(
    polygons_coords: list[list[list[list[float]]]],
    feature_id: str,
    feature_index: int,
) -> tuple[object | None, list[Notice]]:
    """Construct and validate a MultiPolygon from GeoJSON coordinate arrays.

    Returns (MultiPolygon, []) on success or (None, [notices]) on failure.
    """
    sub_polygons = []
    for p, poly_rings in enumerate(polygons_coords):
        polygon, notices = _validate_polygon(
            poly_rings, feature_id, p, "MultiPolygon"
        )
        if notices:
            return (None, notices)  # short-circuit on first invalid sub-polygon
        sub_polygons.append(polygon)
    try:
        multi = MultiPolygon(sub_polygons)
    except (GEOSException, ValueError) as exc:
        return (
            None,
            [
                Notice(
                    code="invalid_geometry",
                    severity=Severity.ERROR,
                    fields={
                        "feature_id": feature_id,
                        "feature_index": feature_index,
                        "geometry_type": "MultiPolygon",
                        "message": str(exc),
                    },
                )
            ],
        )
    if not multi.is_valid:
        return (
            None,
            [
                Notice(
                    code="invalid_geometry",
                    severity=Severity.ERROR,
                    fields={
                        "feature_id": feature_id,
                        "feature_index": feature_index,
                        "geometry_type": "MultiPolygon",
                        "message": explain_validity(multi),
                    },
                )
            ],
        )
    return (multi, [])


def validate_geojson_geometry(
    feed: dict[str, pl.DataFrame],
    ctx: ValidationContext,
) -> list[Notice]:
    """Validate GeoJSON polygon and multi-polygon geometries using Shapely."""
    if "locations_geojson" not in feed or not feed["locations_geojson"]:
        return []

    notices: list[Notice] = []
    for feature in feed["locations_geojson"]:
        feature_id = feature["id"]
        feature_index = feature["index"]
        geom_type = feature["geometry_type"]
        coords = feature["coordinates"]

        if geom_type == "Polygon":
            _, feat_notices = _validate_polygon(
                coords, feature_id, feature_index, "Polygon"
            )
        elif geom_type == "MultiPolygon":
            _, feat_notices = _validate_multi_polygon(
                coords, feature_id, feature_index
            )
        else:
            continue  # unsupported types handled elsewhere
        notices.extend(feat_notices)

    return notices
