"""Explicit validator registry for cross-table business-logic validators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    import polars as pl

    from gtfs_validator.context import ValidationContext
    from gtfs_validator.notices import Notice

ValidatorFn = Callable[
    ["dict[str, pl.DataFrame]", "ValidationContext"],
    "list[Notice]",
]


@dataclass(frozen=True)
class ValidatorEntry:
    """A registered cross-table validator with its dependency metadata."""

    name: str
    fn: ValidatorFn
    requires: list[str]


# Validators are added here as they are implemented.
# Order matters — validators run in registry order.

from gtfs_validator.validators.agency_consistency import validate_agency_consistency
from gtfs_validator.validators.attribution import validate_attribution_without_role
from gtfs_validator.validators.bikes_allowance import validate_bikes_allowance
from gtfs_validator.validators.block_trips_overlapping import validate_block_trips_overlapping
from gtfs_validator.validators.booking_rules_entity import validate_booking_rules_entity
from gtfs_validator.validators.continuous_pickup_drop_off import validate_continuous_pickup_drop_off
from gtfs_validator.validators.date_trips import validate_date_trips
from gtfs_validator.validators.duplicate_fare_media import validate_duplicate_fare_media
from gtfs_validator.validators.duplicate_route_name import validate_duplicate_route_name
from gtfs_validator.validators.expired_calendar import validate_expired_calendar
from gtfs_validator.validators.fare_attribute_agency_id import validate_fare_attribute_agency_id
from gtfs_validator.validators.fare_leg_join_rule import validate_fare_leg_join_rule
from gtfs_validator.validators.fare_media_name import validate_fare_media_name
from gtfs_validator.validators.fare_transfer_rule_duration_limit_type import (
    validate_fare_transfer_rule_duration_limit_type,
)
from gtfs_validator.validators.fare_transfer_rule_transfer_count import (
    validate_fare_transfer_rule_transfer_count,
)
from gtfs_validator.validators.fare_product_default_rider_categories import (
    validate_fare_product_default_rider_categories,
)
from gtfs_validator.validators.fare_leg_rule_network_id_foreign_key import (
    validate_fare_leg_rule_network_id_foreign_key,
)
from gtfs_validator.validators.feed_contact import validate_feed_contact
from gtfs_validator.validators.inconsistent_route_type_for_block_id import (
    validate_inconsistent_route_type_for_block_id,
)
from gtfs_validator.validators.inconsistent_route_type_for_in_seat_transfer import (
    validate_inconsistent_route_type_for_in_seat_transfer,
)
from gtfs_validator.validators.location_has_stop_times import (
    validate_location_has_stop_times,
)
from gtfs_validator.validators.location_type_single_entity import (
    validate_location_type_single_entity,
)
from gtfs_validator.validators.location_id_foreign_key import (
    validate_location_id_foreign_key,
)
from gtfs_validator.validators.matching_feed_and_agency_lang import (
    validate_matching_feed_and_agency_lang,
)
from gtfs_validator.validators.missing_calendar_and_calendar_date import (
    validate_missing_calendar_and_calendar_date,
)
from gtfs_validator.validators.missing_feed_info import validate_missing_feed_info
from gtfs_validator.validators.missing_stops_file import validate_missing_stops_file
from gtfs_validator.validators.missing_level_id import validate_missing_level_id
from gtfs_validator.validators.geojson_geometry import validate_geojson_geometry
from gtfs_validator.validators.feed_expiration_date import validate_feed_expiration_date
from gtfs_validator.validators.feed_service_date import validate_feed_service_date
from gtfs_validator.validators.feed_valid_today import validate_feed_valid_today
from gtfs_validator.validators.pathways import validate_bidirectional_exit_gate
from gtfs_validator.validators.trip_service_id_foreign_key_validator import (
    validate_trip_service_id_foreign_key,
)
from gtfs_validator.validators.trip_usage import validate_trip_usage

VALIDATOR_REGISTRY: list[ValidatorEntry] = [
    ValidatorEntry(
        name="agency_consistency",
        fn=validate_agency_consistency,
        requires=["agency"],
    ),
    ValidatorEntry(
        name="attribution_without_role",
        fn=validate_attribution_without_role,
        requires=["attributions"],
    ),
    ValidatorEntry(
        name="bidirectional_exit_gate",
        fn=validate_bidirectional_exit_gate,
        requires=["pathways"],
    ),
    ValidatorEntry(
        name="block_trips_overlapping",
        fn=validate_block_trips_overlapping,
        requires=["trips", "stop_times"],
    ),
    ValidatorEntry(
        name="bikes_allowance",
        fn=validate_bikes_allowance,
        requires=["routes", "trips"],
    ),
    ValidatorEntry(
        name="booking_rules_entity",
        fn=validate_booking_rules_entity,
        requires=["booking_rules"],
    ),
    ValidatorEntry(
        name="continuous_pickup_drop_off",
        fn=validate_continuous_pickup_drop_off,
        requires=["routes", "trips", "stop_times"],
    ),
    ValidatorEntry(
        name="date_trips",
        fn=validate_date_trips,
        requires=["trips"],
    ),
    ValidatorEntry(
        name="trip_usage",
        fn=validate_trip_usage,
        requires=["trips", "stop_times"],
    ),
    ValidatorEntry(
        name="duplicate_fare_media",
        fn=validate_duplicate_fare_media,
        requires=["fare_media"],
    ),
    ValidatorEntry(
        name="duplicate_route_name",
        fn=validate_duplicate_route_name,
        requires=["routes"],
    ),
    ValidatorEntry(
        name="expired_calendar",
        fn=validate_expired_calendar,
        requires=["calendar", "calendar_dates"],
    ),
    ValidatorEntry(
        name="fare_attribute_agency_id",
        fn=validate_fare_attribute_agency_id,
        requires=["agency", "fare_attributes"],
    ),
    ValidatorEntry(
        name="fare_leg_join_rule",
        fn=validate_fare_leg_join_rule,
        requires=["fare_leg_join_rules"],
    ),
    ValidatorEntry(
        name="fare_leg_rule_network_id_foreign_key",
        fn=validate_fare_leg_rule_network_id_foreign_key,
        requires=["fare_leg_rules"],
    ),
    ValidatorEntry(
        name="fare_media_name",
        fn=validate_fare_media_name,
        requires=["fare_media"],
    ),
    ValidatorEntry(
        name="fare_product_default_rider_categories",
        fn=validate_fare_product_default_rider_categories,
        requires=["fare_products", "rider_categories"],
    ),
    ValidatorEntry(
        name="fare_transfer_rule_duration_limit_type",
        fn=validate_fare_transfer_rule_duration_limit_type,
        requires=["fare_transfer_rules"],
    ),
    ValidatorEntry(
        name="fare_transfer_rule_transfer_count",
        fn=validate_fare_transfer_rule_transfer_count,
        requires=["fare_transfer_rules"],
    ),
    ValidatorEntry(
        name="feed_contact",
        fn=validate_feed_contact,
        requires=["feed_info"],
    ),
    ValidatorEntry(
        name="feed_expiration_date",
        fn=validate_feed_expiration_date,
        requires=["feed_info"],
    ),
    ValidatorEntry(
        name="feed_service_date",
        fn=validate_feed_service_date,
        requires=["feed_info"],
    ),
    ValidatorEntry(
        name="feed_valid_today",
        fn=validate_feed_valid_today,
        requires=["feed_info"],
    ),
    ValidatorEntry(
        name="geojson_geometry",
        fn=validate_geojson_geometry,
        requires=["locations_geojson"],
    ),
    ValidatorEntry(
        name="trip_service_id_foreign_key",
        fn=validate_trip_service_id_foreign_key,
        requires=["trips"],
    ),
    ValidatorEntry(
        name="inconsistent_route_type_for_block_id",
        fn=validate_inconsistent_route_type_for_block_id,
        requires=["trips", "routes"],
    ),
    ValidatorEntry(
        name="inconsistent_route_type_for_in_seat_transfer",
        fn=validate_inconsistent_route_type_for_in_seat_transfer,
        requires=["transfers", "routes"],
    ),
    ValidatorEntry(
        name="location_has_stop_times",
        fn=validate_location_has_stop_times,
        requires=["stops", "stop_times"],
    ),
    ValidatorEntry(
        name="location_id_foreign_key",
        fn=validate_location_id_foreign_key,
        requires=["stop_times", "locations_geojson"],
    ),
    ValidatorEntry(
        name="location_type_single_entity",
        fn=validate_location_type_single_entity,
        requires=["stops"],
    ),
    ValidatorEntry(
        name="matching_feed_and_agency_lang",
        fn=validate_matching_feed_and_agency_lang,
        requires=["feed_info", "agency"],
    ),
    ValidatorEntry(
        name="missing_calendar_and_calendar_date",
        fn=validate_missing_calendar_and_calendar_date,
        requires=[],
    ),
    ValidatorEntry(
        name="missing_feed_info",
        fn=validate_missing_feed_info,
        requires=[],
    ),
    ValidatorEntry(
        name="missing_stops_file",
        fn=validate_missing_stops_file,
        requires=[],
    ),
    ValidatorEntry(
        name="missing_level_id",
        fn=validate_missing_level_id,
        requires=["pathways", "stops"],
    ),
]
