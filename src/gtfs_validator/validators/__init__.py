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
from gtfs_validator.validators.pathways import validate_bidirectional_exit_gate

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
        name="duplicate_fare_media",
        fn=validate_duplicate_fare_media,
        requires=["fare_media"],
    ),
]
