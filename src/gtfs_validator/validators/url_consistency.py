"""Validator: check that URLs are not duplicated across agency, routes, and stops."""

from __future__ import annotations

import polars as pl

from gtfs_validator.context import ValidationContext
from gtfs_validator.notices import Notice, Severity


def validate_url_consistency(
    feed: dict[str, pl.DataFrame],
    ctx: ValidationContext,
) -> list[Notice]:
    """Check that URLs are not duplicated across agency.txt, routes.txt, and stops.txt.
    
    Emits WARNING notices when:
    - routes.route_url matches agency.agency_url
    - stops.stop_url matches agency.agency_url
    - stops.stop_url matches routes.route_url
    
    URL comparison is case-insensitive.
    """
    notices: list[Notice] = []
    
    # Build maps of URLs to records (case-insensitive)
    agency_by_url: dict[str, list[dict]] = {}
    route_by_url: dict[str, list[dict]] = {}
    
    # Index agencies by URL
    if "agency" in feed and not feed["agency"].is_empty():
        agency_df = feed["agency"]
        if "agency_url" in agency_df.columns:
            for row in agency_df.select("agency_url", "agency_name", "csv_row_number").iter_rows(named=True):
                if row["agency_url"] is not None:
                    url_lower = row["agency_url"].lower()
                    if url_lower not in agency_by_url:
                        agency_by_url[url_lower] = []
                    agency_by_url[url_lower].append({
                        "agency_name": row["agency_name"],
                        "csv_row_number": row["csv_row_number"],
                    })
    
    # Index routes by URL
    if "routes" in feed and not feed["routes"].is_empty():
        routes_df = feed["routes"]
        if "route_url" in routes_df.columns:
            for row in routes_df.select("route_url", "route_id", "csv_row_number").iter_rows(named=True):
                if row["route_url"] is not None:
                    url_lower = row["route_url"].lower()
                    if url_lower not in route_by_url:
                        route_by_url[url_lower] = []
                    route_by_url[url_lower].append({
                        "route_id": row["route_id"],
                        "csv_row_number": row["csv_row_number"],
                        "route_url": row["route_url"],
                    })
    
    # Check routes against agencies
    if "routes" in feed and not feed["routes"].is_empty():
        routes_df = feed["routes"]
        if "route_url" in routes_df.columns:
            for row in routes_df.select("route_url", "route_id", "csv_row_number").iter_rows(named=True):
                if row["route_url"] is not None:
                    url_lower = row["route_url"].lower()
                    if url_lower in agency_by_url:
                        for agency in agency_by_url[url_lower]:
                            notices.append(
                                Notice(
                                    code="same_route_and_agency_url",
                                    severity=Severity.WARNING,
                                    fields={
                                        "route_csv_row_number": row["csv_row_number"],
                                        "route_id": row["route_id"],
                                        "agency_name": agency["agency_name"],
                                        "route_url": row["route_url"],
                                        "agency_csv_row_number": agency["csv_row_number"],
                                    },
                                )
                            )
    
    # Check stops against agencies and routes
    if "stops" in feed and not feed["stops"].is_empty():
        stops_df = feed["stops"]
        if "stop_url" in stops_df.columns:
            for row in stops_df.select("stop_url", "stop_id", "csv_row_number").iter_rows(named=True):
                if row["stop_url"] is not None:
                    url_lower = row["stop_url"].lower()
                    
                    # Check against agencies
                    if url_lower in agency_by_url:
                        for agency in agency_by_url[url_lower]:
                            notices.append(
                                Notice(
                                    code="same_stop_and_agency_url",
                                    severity=Severity.WARNING,
                                    fields={
                                        "stop_csv_row_number": row["csv_row_number"],
                                        "stop_id": row["stop_id"],
                                        "agency_name": agency["agency_name"],
                                        "stop_url": row["stop_url"],
                                        "agency_csv_row_number": agency["csv_row_number"],
                                    },
                                )
                            )
                    
                    # Check against routes
                    if url_lower in route_by_url:
                        for route in route_by_url[url_lower]:
                            notices.append(
                                Notice(
                                    code="same_stop_and_route_url",
                                    severity=Severity.WARNING,
                                    fields={
                                        "stop_csv_row_number": row["csv_row_number"],
                                        "stop_id": row["stop_id"],
                                        "stop_url": row["stop_url"],
                                        "route_id": route["route_id"],
                                        "route_csv_row_number": route["csv_row_number"],
                                    },
                                )
                            )
    
    return notices