"""Table schema registry — pure data, no behavior.

Each GTFS table is described by a ``TableDefinition`` containing column
definitions, primary/foreign key metadata, and constraint annotations.  The
loader and load-time validators consume these definitions to drive parsing
and schema-level checks.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Field type enum — determines Polars dtype and parse/validation logic
# ---------------------------------------------------------------------------


class FieldType(enum.Enum):
    ID = "id"
    TEXT = "text"
    URL = "url"
    EMAIL = "email"
    PHONE = "phone"
    INTEGER = "integer"
    FLOAT = "float"
    DECIMAL = "decimal"
    LATITUDE = "latitude"
    LONGITUDE = "longitude"
    COLOR = "color"
    DATE = "date"
    TIME = "time"
    TIMEZONE = "timezone"
    LANGUAGE = "language"
    CURRENCY = "currency"
    ENUM = "enum"


class Presence(enum.Enum):
    REQUIRED = "required"
    RECOMMENDED = "recommended"
    OPTIONAL = "optional"
    CONDITIONALLY_REQUIRED = "conditionally_required"


class NumericConstraint(enum.Enum):
    NON_NEGATIVE = "non_negative"
    POSITIVE = "positive"
    NON_ZERO = "non_zero"


class TableRequirement(enum.Enum):
    REQUIRED = "required"
    RECOMMENDED = "recommended"
    OPTIONAL = "optional"
    CONDITIONALLY_REQUIRED = "conditionally_required"


# ---------------------------------------------------------------------------
# Column and table definitions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ForeignKeyRef:
    table: str   # e.g. "stops.txt"
    field: str   # e.g. "stop_id"


@dataclass(frozen=True)
class EndRangeRef:
    field: str         # target column name
    allow_equal: bool


@dataclass(frozen=True)
class CurrencyAmountRef:
    currency_field: str


@dataclass(frozen=True)
class ColumnDefinition:
    name: str
    field_type: FieldType
    presence: Presence = Presence.OPTIONAL
    primary_key: bool = False
    foreign_key: ForeignKeyRef | None = None
    numeric_constraint: NumericConstraint | None = None
    mixed_case: bool = False
    default_value: str | None = None
    end_range: EndRangeRef | None = None
    currency_amount: CurrencyAmountRef | None = None
    enum_values: set[int] | None = None


@dataclass(frozen=True)
class TableDefinition:
    filename: str
    requirement: TableRequirement
    columns: list[ColumnDefinition]
    single_row: bool = False

    def column_names(self) -> list[str]:
        return [c.name for c in self.columns]

    def column_map(self) -> dict[str, ColumnDefinition]:
        return {c.name: c for c in self.columns}

    def primary_key_columns(self) -> list[str]:
        return [c.name for c in self.columns if c.primary_key]


# ---------------------------------------------------------------------------
# Helper shortcuts
# ---------------------------------------------------------------------------

_R = Presence.REQUIRED
_REC = Presence.RECOMMENDED
_O = Presence.OPTIONAL
_CR = Presence.CONDITIONALLY_REQUIRED

_ID = FieldType.ID
_TXT = FieldType.TEXT
_URL = FieldType.URL
_EMAIL = FieldType.EMAIL
_PHONE = FieldType.PHONE
_INT = FieldType.INTEGER
_FLT = FieldType.FLOAT
_DEC = FieldType.DECIMAL
_LAT = FieldType.LATITUDE
_LON = FieldType.LONGITUDE
_COLOR = FieldType.COLOR
_DATE = FieldType.DATE
_TIME = FieldType.TIME
_TZ = FieldType.TIMEZONE
_LANG = FieldType.LANGUAGE
_CUR = FieldType.CURRENCY
_ENUM = FieldType.ENUM

_NN = NumericConstraint.NON_NEGATIVE
_POS = NumericConstraint.POSITIVE
_NZ = NumericConstraint.NON_ZERO

_FK = ForeignKeyRef
_ER = EndRangeRef
_CA = CurrencyAmountRef

_TR = TableRequirement.REQUIRED
_TO = TableRequirement.OPTIONAL
_TCR = TableRequirement.CONDITIONALLY_REQUIRED


def _col(
    name: str,
    ft: FieldType = _TXT,
    pr: Presence = _O,
    *,
    pk: bool = False,
    fk: ForeignKeyRef | None = None,
    nc: NumericConstraint | None = None,
    mc: bool = False,
    dv: str | None = None,
    er: EndRangeRef | None = None,
    ca: CurrencyAmountRef | None = None,
    ev: set[int] | None = None,
) -> ColumnDefinition:
    return ColumnDefinition(
        name=name,
        field_type=ft,
        presence=pr,
        primary_key=pk,
        foreign_key=fk,
        numeric_constraint=nc,
        mixed_case=mc,
        default_value=dv,
        end_range=er,
        currency_amount=ca,
        enum_values=ev,
    )


# ---------------------------------------------------------------------------
# GTFS table definitions (32 tables)
# ---------------------------------------------------------------------------

AGENCY = TableDefinition(
    filename="agency.txt",
    requirement=_TR,
    columns=[
        _col("agency_id", _ID, _CR, pk=True),
        _col("agency_name", _TXT, _R, mc=True),
        _col("agency_url", _URL, _R),
        _col("agency_timezone", _TZ, _R),
        _col("agency_lang", _LANG),
        _col("agency_phone", _PHONE),
        _col("agency_fare_url", _URL),
        _col("agency_email", _EMAIL),
        _col("cemv_support", _ENUM),
    ],
)

STOPS = TableDefinition(
    filename="stops.txt",
    requirement=_TCR,
    columns=[
        _col("stop_id", _ID, _R, pk=True),
        _col("stop_code", _TXT),
        _col("stop_name", _TXT, _CR, mc=True),
        _col("tts_stop_name", _TXT),
        _col("stop_desc", _TXT),
        _col("stop_lat", _LAT, _CR),
        _col("stop_lon", _LON, _CR),
        _col("zone_id", _ID, _CR),
        _col("stop_url", _URL),
        _col("location_type", _ENUM),
        _col("parent_station", _ID, _CR, fk=_FK("stops.txt", "stop_id")),
        _col("stop_timezone", _TZ),
        _col("wheelchair_boarding", _ENUM),
        _col("level_id", _ID, fk=_FK("levels.txt", "level_id")),
        _col("platform_code", _TXT),
        _col("stop_access", _ENUM),
    ],
)

ROUTES = TableDefinition(
    filename="routes.txt",
    requirement=_TR,
    columns=[
        _col("route_id", _ID, _R, pk=True),
        _col("agency_id", _ID, _CR, fk=_FK("agency.txt", "agency_id")),
        _col("route_short_name", _TXT, _CR, mc=True),
        _col("route_long_name", _TXT, _CR, mc=True),
        _col("route_desc", _TXT, mc=True),
        _col("route_type", _ENUM, _R),
        _col("route_url", _URL),
        _col("route_color", _COLOR, dv="FFFFFF"),
        _col("route_text_color", _COLOR, dv="000000"),
        _col("route_sort_order", _INT, nc=_NN),
        _col("continuous_pickup", _ENUM, dv="1"),
        _col("continuous_drop_off", _ENUM, dv="1"),
        _col("network_id", _ID, _CR),
        _col("cemv_support", _ENUM),
    ],
)

TRIPS = TableDefinition(
    filename="trips.txt",
    requirement=_TR,
    columns=[
        _col("trip_id", _ID, _R, pk=True),
        _col("route_id", _ID, _R, fk=_FK("routes.txt", "route_id")),
        _col("service_id", _ID, _R),
        _col("trip_headsign", _TXT, mc=True),
        _col("trip_short_name", _TXT, mc=True),
        _col("direction_id", _ENUM),
        _col("block_id", _ID),
        _col("shape_id", _ID, _CR, fk=_FK("shapes.txt", "shape_id")),
        _col("wheelchair_accessible", _ENUM),
        _col("bikes_allowed", _ENUM),
        _col("cars_allowed", _ENUM),
    ],
)

STOP_TIMES = TableDefinition(
    filename="stop_times.txt",
    requirement=_TR,
    columns=[
        _col("trip_id", _ID, _R, pk=True, fk=_FK("trips.txt", "trip_id")),
        _col("arrival_time", _TIME, _CR, er=_ER("departure_time", allow_equal=True)),
        _col("departure_time", _TIME, _CR),
        _col("stop_id", _ID, _CR, fk=_FK("stops.txt", "stop_id")),
        _col("location_group_id", _ID, _CR, fk=_FK("location_groups.txt", "location_group_id")),
        _col("location_id", _ID, _CR),
        _col("stop_sequence", _INT, _R, pk=True, nc=_NN),
        _col("stop_headsign", _TXT),
        _col("start_pickup_drop_off_window", _TIME),
        _col("end_pickup_drop_off_window", _TIME),
        _col("pickup_type", _ENUM),
        _col("drop_off_type", _ENUM),
        _col("continuous_pickup", _ENUM, dv="1"),
        _col("continuous_drop_off", _ENUM, dv="1"),
        _col("shape_dist_traveled", _FLT, nc=_NN),
        _col("timepoint", _ENUM, dv="1"),
        _col("pickup_booking_rule_id", _ID, fk=_FK("booking_rules.txt", "booking_rule_id")),
        _col("drop_off_booking_rule_id", _ID, fk=_FK("booking_rules.txt", "booking_rule_id")),
    ],
)

CALENDAR = TableDefinition(
    filename="calendar.txt",
    requirement=_TCR,
    columns=[
        _col("service_id", _ID, _R, pk=True),
        _col("monday", _ENUM, _R),
        _col("tuesday", _ENUM, _R),
        _col("wednesday", _ENUM, _R),
        _col("thursday", _ENUM, _R),
        _col("friday", _ENUM, _R),
        _col("saturday", _ENUM, _R),
        _col("sunday", _ENUM, _R),
        _col("start_date", _DATE, _R, er=_ER("end_date", allow_equal=True)),
        _col("end_date", _DATE, _R),
    ],
)

CALENDAR_DATES = TableDefinition(
    filename="calendar_dates.txt",
    requirement=_TCR,
    columns=[
        _col("service_id", _ID, _R, pk=True),
        _col("date", _DATE, _R, pk=True),
        _col("exception_type", _ENUM, _R),
    ],
)

FARE_ATTRIBUTES = TableDefinition(
    filename="fare_attributes.txt",
    requirement=_TO,
    columns=[
        _col("fare_id", _ID, _R, pk=True),
        _col("price", _DEC, _R, nc=_NN),
        _col("currency_type", _CUR, _R),
        _col("payment_method", _ENUM, _R),
        _col("transfers", _ENUM),
        _col("agency_id", _ID, _CR, fk=_FK("agency.txt", "agency_id")),
        _col("transfer_duration", _INT, nc=_NN),
    ],
)

FARE_RULES = TableDefinition(
    filename="fare_rules.txt",
    requirement=_TO,
    columns=[
        _col("fare_id", _ID, _R, pk=True, fk=_FK("fare_attributes.txt", "fare_id")),
        _col("route_id", _ID, pk=True, fk=_FK("routes.txt", "route_id")),
        _col("origin_id", _ID, pk=True, fk=_FK("stops.txt", "zone_id")),
        _col("destination_id", _ID, pk=True, fk=_FK("stops.txt", "zone_id")),
        _col("contains_id", _ID, pk=True, fk=_FK("stops.txt", "zone_id")),
    ],
)

TIMEFRAMES = TableDefinition(
    filename="timeframes.txt",
    requirement=_TO,
    columns=[
        _col("timeframe_group_id", _ID, pk=True),
        _col("start_time", _TIME, _CR, pk=True, er=_ER("end_time", allow_equal=False)),
        _col("end_time", _TIME, _CR, pk=True),
        _col("service_id", _TXT, _R, pk=True),
    ],
)

FARE_MEDIA = TableDefinition(
    filename="fare_media.txt",
    requirement=_TO,
    columns=[
        _col("fare_media_id", _ID, _R, pk=True),
        _col("fare_media_name", _TXT),
        _col("fare_media_type", _ENUM, _R),
    ],
)

FARE_PRODUCTS = TableDefinition(
    filename="fare_products.txt",
    requirement=_TO,
    columns=[
        _col("fare_product_id", _ID, _R, pk=True),
        _col("fare_product_name", _TXT),
        _col("amount", _DEC, _R, nc=_NN, ca=_CA("currency")),
        _col("currency", _CUR, _R),
        _col("fare_media_id", _ID, pk=True, fk=_FK("fare_media.txt", "fare_media_id")),
        _col("rider_category_id", _ID, pk=True, fk=_FK("rider_categories.txt", "rider_category_id")),
    ],
)

FARE_LEG_RULES = TableDefinition(
    filename="fare_leg_rules.txt",
    requirement=_TO,
    columns=[
        _col("leg_group_id", _ID),
        _col("network_id", _ID, pk=True),
        _col("from_area_id", _ID, pk=True, fk=_FK("areas.txt", "area_id")),
        _col("to_area_id", _ID, pk=True, fk=_FK("areas.txt", "area_id")),
        _col("from_timeframe_group_id", _ID, pk=True, fk=_FK("timeframes.txt", "timeframe_group_id")),
        _col("to_timeframe_group_id", _ID, pk=True, fk=_FK("timeframes.txt", "timeframe_group_id")),
        _col("fare_product_id", _ID, _R, pk=True, fk=_FK("fare_products.txt", "fare_product_id")),
        _col("rule_priority", _INT, nc=_NN),
    ],
)

FARE_LEG_JOIN_RULES = TableDefinition(
    filename="fare_leg_join_rules.txt",
    requirement=_TO,
    columns=[
        _col("from_network_id", _ID, _R),
        _col("to_network_id", _ID, _R),
        _col("from_stop_id", _ID, _CR, fk=_FK("stops.txt", "stop_id")),
        _col("to_stop_id", _ID, _CR, fk=_FK("stops.txt", "stop_id")),
    ],
)

FARE_TRANSFER_RULES = TableDefinition(
    filename="fare_transfer_rules.txt",
    requirement=_TO,
    columns=[
        _col("from_leg_group_id", _ID, pk=True, fk=_FK("fare_leg_rules.txt", "leg_group_id")),
        _col("to_leg_group_id", _ID, pk=True, fk=_FK("fare_leg_rules.txt", "leg_group_id")),
        _col("duration_limit", _INT, pk=True, nc=_POS),
        _col("duration_limit_type", _ENUM, _CR),
        _col("fare_transfer_type", _ENUM, _R),
        _col("transfer_count", _INT, pk=True, pr=_CR),
        _col("fare_product_id", _ID, pk=True, fk=_FK("fare_products.txt", "fare_product_id")),
    ],
)

AREAS = TableDefinition(
    filename="areas.txt",
    requirement=_TO,
    columns=[
        _col("area_id", _ID, _R, pk=True),
        _col("area_name", _TXT),
    ],
)

STOP_AREAS = TableDefinition(
    filename="stop_areas.txt",
    requirement=_TO,
    columns=[
        _col("area_id", _ID, _R, pk=True, fk=_FK("areas.txt", "area_id")),
        _col("stop_id", _ID, _R, pk=True, fk=_FK("stops.txt", "stop_id")),
    ],
)

RIDER_CATEGORIES = TableDefinition(
    filename="rider_categories.txt",
    requirement=_TO,
    columns=[
        _col("rider_category_id", _ID, _R, pk=True),
        _col("rider_category_name", _TXT, _R),
        _col("is_default_fare_category", _ENUM, dv="0"),
        _col("eligibility_url", _URL),
    ],
)

NETWORKS = TableDefinition(
    filename="networks.txt",
    requirement=_TO,
    columns=[
        _col("network_id", _ID, _R, pk=True),
        _col("network_name", _TXT, mc=True),
    ],
)

ROUTE_NETWORKS = TableDefinition(
    filename="route_networks.txt",
    requirement=_TO,
    columns=[
        _col("route_id", _ID, _R, pk=True, fk=_FK("routes.txt", "route_id")),
        _col("network_id", _ID, _R, fk=_FK("networks.txt", "network_id")),
    ],
)

SHAPES = TableDefinition(
    filename="shapes.txt",
    requirement=_TO,
    columns=[
        _col("shape_id", _ID, _R, pk=True),
        _col("shape_pt_lat", _LAT, _R),
        _col("shape_pt_lon", _LON, _R),
        _col("shape_pt_sequence", _INT, _R, pk=True, nc=_NN),
        _col("shape_dist_traveled", _FLT, nc=_NN),
    ],
)

FREQUENCIES = TableDefinition(
    filename="frequencies.txt",
    requirement=_TO,
    columns=[
        _col("trip_id", _ID, _R, pk=True, fk=_FK("trips.txt", "trip_id")),
        _col("start_time", _TIME, _R, pk=True, er=_ER("end_time", allow_equal=False)),
        _col("end_time", _TIME, _R),
        _col("headway_secs", _INT, _R, nc=_POS),
        _col("exact_times", _ENUM),
    ],
)

TRANSFERS = TableDefinition(
    filename="transfers.txt",
    requirement=_TO,
    columns=[
        _col("from_stop_id", _ID, _CR, pk=True, fk=_FK("stops.txt", "stop_id")),
        _col("to_stop_id", _ID, _CR, pk=True, fk=_FK("stops.txt", "stop_id")),
        _col("transfer_type", _ENUM),
        _col("min_transfer_time", _INT, nc=_NN),
        _col("from_trip_id", _ID, _CR, pk=True, fk=_FK("trips.txt", "trip_id")),
        _col("to_trip_id", _ID, _CR, pk=True, fk=_FK("trips.txt", "trip_id")),
        _col("from_route_id", _ID, pk=True, fk=_FK("routes.txt", "route_id")),
        _col("to_route_id", _ID, pk=True, fk=_FK("routes.txt", "route_id")),
    ],
)

PATHWAYS = TableDefinition(
    filename="pathways.txt",
    requirement=_TO,
    columns=[
        _col("pathway_id", _ID, _R, pk=True),
        _col("from_stop_id", _ID, _R, fk=_FK("stops.txt", "stop_id")),
        _col("to_stop_id", _ID, _R, fk=_FK("stops.txt", "stop_id")),
        _col("pathway_mode", _ENUM, _R),
        _col("is_bidirectional", _ENUM, _R),
        _col("length", _FLT, nc=_NN),
        _col("traversal_time", _INT, nc=_POS),
        _col("stair_count", _INT, nc=_NZ),
        _col("max_slope", _FLT),
        _col("min_width", _FLT, nc=_POS),
        _col("signposted_as", _TXT, mc=True),
        _col("reversed_signposted_as", _TXT, mc=True),
    ],
)

LEVELS = TableDefinition(
    filename="levels.txt",
    requirement=_TO,
    columns=[
        _col("level_id", _ID, _R, pk=True),
        _col("level_index", _FLT, _R),
        _col("level_name", _TXT, mc=True),
    ],
)

TRANSLATIONS = TableDefinition(
    filename="translations.txt",
    requirement=_TO,
    columns=[
        _col("table_name", _TXT, _R, pk=True),
        _col("field_name", _TXT, _R, pk=True),
        _col("language", _LANG, _R, pk=True),
        _col("translation", _TXT, _R),
        _col("record_id", _TXT, _CR, pk=True),
        _col("record_sub_id", _TXT, _CR, pk=True),
        _col("field_value", _TXT, _CR, pk=True),
    ],
)

FEED_INFO = TableDefinition(
    filename="feed_info.txt",
    requirement=_TCR,
    single_row=True,
    columns=[
        _col("feed_publisher_name", _TXT, _R),
        _col("feed_publisher_url", _URL, _R),
        _col("feed_lang", _LANG, _R),
        _col("default_lang", _LANG),
        _col("feed_start_date", _DATE, _REC, er=_ER("feed_end_date", allow_equal=True)),
        _col("feed_end_date", _DATE, _REC),
        _col("feed_version", _TXT, _REC),
        _col("feed_contact_email", _EMAIL),
        _col("feed_contact_url", _URL),
    ],
)

ATTRIBUTIONS = TableDefinition(
    filename="attributions.txt",
    requirement=_TO,
    columns=[
        _col("attribution_id", _ID, pk=True),
        _col("agency_id", _ID, fk=_FK("agency.txt", "agency_id")),
        _col("route_id", _ID, fk=_FK("routes.txt", "route_id")),
        _col("trip_id", _ID, fk=_FK("trips.txt", "trip_id")),
        _col("organization_name", _TXT, _R),
        _col("is_producer", _ENUM),
        _col("is_operator", _ENUM),
        _col("is_authority", _ENUM),
        _col("attribution_url", _URL),
        _col("attribution_email", _EMAIL),
        _col("attribution_phone", _PHONE),
    ],
)

BOOKING_RULES = TableDefinition(
    filename="booking_rules.txt",
    requirement=_TO,
    columns=[
        _col("booking_rule_id", _ID, _R, pk=True),
        _col("booking_type", _ENUM, _R),
        _col("prior_notice_duration_min", _INT, _CR),
        _col("prior_notice_duration_max", _INT, _CR),
        _col("prior_notice_start_day", _INT, _CR),
        _col("prior_notice_start_time", _TIME, _CR),
        _col("prior_notice_last_day", _INT, _CR),
        _col("prior_notice_last_time", _TIME, _CR),
        _col("prior_notice_service_id", _TXT, _CR),
        _col("message", _TXT, mc=True),
        _col("pickup_message", _TXT, mc=True),
        _col("drop_off_message", _TXT, mc=True),
        _col("phone_number", _PHONE),
        _col("info_url", _URL),
        _col("booking_url", _URL),
    ],
)

LOCATION_GROUPS = TableDefinition(
    filename="location_groups.txt",
    requirement=_TO,
    columns=[
        _col("location_group_id", _ID, _R, pk=True),
        _col("location_group_name", _TXT, mc=True),
    ],
)

LOCATION_GROUP_STOPS = TableDefinition(
    filename="location_group_stops.txt",
    requirement=_TO,
    columns=[
        _col("location_group_id", _ID, _R),
        _col("stop_id", _ID, _R),
    ],
)


# ---------------------------------------------------------------------------
# Registry: filename -> TableDefinition
# ---------------------------------------------------------------------------

ALL_TABLES: list[TableDefinition] = [
    AGENCY,
    STOPS,
    ROUTES,
    TRIPS,
    STOP_TIMES,
    CALENDAR,
    CALENDAR_DATES,
    FARE_ATTRIBUTES,
    FARE_RULES,
    TIMEFRAMES,
    FARE_MEDIA,
    FARE_PRODUCTS,
    FARE_LEG_RULES,
    FARE_LEG_JOIN_RULES,
    FARE_TRANSFER_RULES,
    AREAS,
    STOP_AREAS,
    RIDER_CATEGORIES,
    NETWORKS,
    ROUTE_NETWORKS,
    SHAPES,
    FREQUENCIES,
    TRANSFERS,
    PATHWAYS,
    LEVELS,
    TRANSLATIONS,
    FEED_INFO,
    ATTRIBUTIONS,
    BOOKING_RULES,
    LOCATION_GROUPS,
    LOCATION_GROUP_STOPS,
]

TABLE_BY_FILENAME: dict[str, TableDefinition] = {t.filename: t for t in ALL_TABLES}
