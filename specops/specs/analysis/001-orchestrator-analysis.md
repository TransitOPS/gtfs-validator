# 001 — Orchestrator Analysis

Extracted from the Java GTFS Validator codebase. This document captures every behavioral contract the Python/Polars orchestrator must satisfy to produce identical output.

---

## 1. Pipeline Stages

The Java validator executes a strict sequential pipeline. The Python orchestrator must replicate these stages in order:

```
CLI parse → Config build → Input open → Table load (parallel) →
Single-entity validate (during load) → Single-file validate (after each table) →
Multi-file validate (parallel) → Feature detection → Report generation → Cleanup
```

### Stage 1: CLI Argument Parsing

**Source:** `cli/src/main/java/.../cli/Arguments.java`

| Flag | Long form | Type | Required | Default | Notes |
|------|-----------|------|----------|---------|-------|
| `-i` | `--input` | string | conditional | — | Path to GTFS ZIP or directory |
| `-u` | `--url` | string | conditional | — | URL to GTFS archive |
| `-o` | `--output_base` | string | yes | — | Output directory for reports |
| `-s` | `--storage_directory` | string | no | `null` | Disk path to save downloaded ZIP |
| `-t` | `--threads` | int | no | `1` | Parallel execution thread count |
| `-c` | `--country_code` | string | no | `null` | ISO 3166-1 alpha-2 country code |
| `-d` | `--date` | string | no | `null` | ISO_LOCAL_DATE override (YYYY-MM-DD) |
| `-v` | `--validation_report_name` | string | no | `null` | Custom JSON report filename |
| `-r` | `--html_report_name` | string | no | `null` | Custom HTML report filename |
| `-e` | `--system_errors_report_name` | string | no | `null` | Custom system errors filename |
| `-p` | `--pretty` | bool | no | `false` | Pretty-print JSON output |
| `-n` | `--export_notices_schema` | bool | no | `false` | Export notice schema and exit |
| `-h` | `--help` | bool | no | `false` | Print help |
| `-svu` | `--skip_validator_update` | bool | no | `false` | Skip version update check |

**Validation rules:**
- Exactly one of `--input` or `--url` must be provided (not both, not neither)
- `--storage_directory` is only valid with `--url`
- Exception: `--export_notices_schema` does not require `--input` or `--url`

**Exit codes:**
- `0` — validation completed successfully
- `1` — validation completed but system errors occurred
- `2` — exception during validation (fatal)

### Stage 2: Configuration

**Source:** `main/src/main/java/.../runner/ValidationRunnerConfig.java`

Immutable config object built from CLI arguments:

| Property | Type | Default |
|----------|------|---------|
| `gtfsSource` | URI | from `--input` (file://) or `--url` (http/https) |
| `outputDirectory` | Path | from `--output_base` |
| `storageDirectory` | Optional\<Path\> | from `--storage_directory` |
| `validationReportFileName` | string | `"report.json"` |
| `htmlReportFileName` | string | `"report.html"` |
| `systemErrorsReportFileName` | string | `"system_errors.json"` |
| `numThreads` | int | `1` |
| `countryCode` | CountryCode | `ZZ` (unknown) |
| `dateForValidation` | LocalDate | `LocalDate.now()` |
| `prettyJson` | bool | `false` |
| `skipValidatorUpdate` | bool | `false` |

### Stage 3: Input Handling

**Source:** `core/src/main/java/.../input/GtfsInput.java`, `GtfsZipFileInput.java`, `GtfsUnarchivedInput.java`

Three input modes:

1. **Local file/directory** (`--input`):
   - If path is a directory → `GtfsUnarchivedInput`: streams directory for regular files, builds filename set
   - If path is a ZIP file → `GtfsZipFileInput`: opens via Apache Commons ZipFile

2. **URL with disk storage** (`--url` + `--storage_directory`):
   - Downloads ZIP via HTTP GET to disk path
   - Opens downloaded file as `GtfsZipFileInput`

3. **URL in-memory** (`--url` without `--storage_directory`):
   - Downloads ZIP into `ByteArrayOutputStream`
   - Opens via `SeekableInMemoryByteChannel` as `GtfsZipFileInput`

**Input validation during open:**
- Checks for GTFS files in subdirectories (invalid structure) → notice emitted
- Filters out macOS `.DS_Store` and `__MACOSX/` entries from ZIP
- Filters out directory entries (paths ending with `/`)
- ZIP entry names that include a directory prefix (e.g., `feed/stops.txt`) are stripped to just the filename

**Abstract contract (`GtfsInput`):**
- `getFilenames() -> ImmutableSet<String>` — all CSV filenames in the feed
- `getFile(filename) -> InputStream` — byte stream for a single file
- `close()` — release resources

### Stage 4: Table Loading (Parallel)

**Source:** `core/src/main/java/.../table/GtfsFeedLoader.java`, `CsvFileLoader.java`

The feed loader executes in two phases:

**Phase 1 — Load each table in parallel:**
- Uses `ExecutorService` with configurable thread count
- For each filename in the input:
  - Look up `GtfsFileDescriptor` by filename
  - If no descriptor found → emit `UnknownFileNotice` (INFO)
  - If descriptor found → submit `CsvFileLoader.load()` as callable
- For each known table NOT in input:
  - Call `loadMissingFile()` → creates empty container
  - If table is required → emit `MissingRequiredFileNotice` (ERROR)
  - If table is recommended → emit `MissingRecommendedFileNotice` (WARNING)

**CsvFileLoader per-table process:**
1. Parse CSV with Univocity parser (configurable `maxCharsPerColumn`)
2. If parse fails → `CsvParsingFailedNotice` (ERROR), return invalid container
3. If file empty → `EmptyFileNotice` (ERROR for required, WARNING for recommended)
4. Validate header:
   - Required columns missing → `MissingRequiredColumnNotice` (ERROR)
   - Recommended columns missing → `MissingRecommendedColumnNotice` (WARNING)
   - Unknown columns → `UnknownColumnNotice` (INFO)
   - Duplicate columns → `DuplicateColumnNotice` (ERROR)
5. For each row:
   - Validate row length matches header → `InvalidRowLengthNotice` (ERROR)
   - Parse each field via `RowParser` (type conversion + constraint checks)
   - Run single-entity validators on parsed entity
   - Build entity and add to list
6. Create `GtfsTableContainer` with header + entities
7. Run single-file validators
8. Log field cache statistics

**Phase 2 — Multi-file validation (parallel):**
- Create `GtfsFeedContainer` from all loaded table containers
- Create multi-file validators via `DefaultValidatorProvider`
- Execute each multi-file validator as a callable in thread pool
- Each wrapped with `ValidatorUtil.safeValidate()` to catch exceptions
- Merge all notice containers

### Stage 5: Type Conversion (RowParser)

**Source:** `core/src/main/java/.../parsing/RowParser.java`

Field parsing rules — each produces a typed value or `null` + optional notice:

| Parse method | Input format | Output type | Validation | Notice on failure |
|---|---|---|---|---|
| `asString` | any | String | checks required/recommended presence, invalid chars | `MissingRequiredValueNotice`, `MissingRecommendedValueNotice` |
| `asId` | any | String | field validator | `InvalidIdNotice` |
| `asUrl` | URL string | String | URL format | `InvalidUrlNotice` |
| `asEmail` | email string | String | email format | `InvalidEmailNotice` |
| `asPhoneNumber` | phone string | String | country-specific | `InvalidPhoneNumberNotice` |
| `asInteger` | integer string | int | `NumberBounds` (POSITIVE, NON_NEGATIVE, NON_ZERO) | `InvalidIntegerNotice`, `NumberOutOfRangeNotice` |
| `asFloat` | decimal string | double | `NumberBounds` | `InvalidFloatNotice`, `NumberOutOfRangeNotice` |
| `asDecimal` | decimal string | BigDecimal | — | `InvalidDecimalNotice` |
| `asLatitude` | decimal string | double | range [-90, 90] | `InvalidFloatNotice`, `PointNearOriginNotice` (if ~0,0) |
| `asLongitude` | decimal string | double | range [-180, 180] | `InvalidFloatNotice`, `PointNearOriginNotice` |
| `asColor` | hex string | GtfsColor | 6-char hex (no `#`) | `InvalidColorNotice` |
| `asDate` | YYYYMMDD | GtfsDate | parseable date | `InvalidDateNotice` |
| `asTime` | H:MM:SS or HH:MM:SS | GtfsTime | allows hours > 23 | `InvalidTimeNotice` |
| `asLanguageCode` | BCP 47 tag | Locale | IETF language tag | `InvalidLanguageCodeNotice` |
| `asCurrencyCode` | ISO 4217 | Currency | 3-letter code | `InvalidCurrencyNotice` |
| `asTimezone` | IANA tz | ZoneId | known timezone | `InvalidTimezoneNotice` |
| `asEnum` | integer | enum | valid enum value | `UnexpectedEnumValueNotice` |

**Additional RowParser behaviors:**
- Empty string treated as missing value
- Whitespace-only values treated as missing
- Leading/trailing whitespace is trimmed (with `LeadingOrTrailingWhitespacesNotice` WARNING)
- Row numbers capped at 1 billion (safety check)
- Row length mismatch (more or fewer columns than header) → `InvalidRowLengthNotice`

### Stage 6: Annotation-Driven Load-Time Validation

These checks are generated by the Java annotation processor and executed during loading. In the Python orchestrator, they become schema-driven logic:

| Annotation | Check | Notice |
|---|---|---|
| `@Required` (on field) | Column header must exist; value must be non-empty in every row | `MissingRequiredColumnNotice`, `MissingRequiredValueNotice` |
| `@Recommended` | Column header should exist; value should be non-empty | `MissingRecommendedColumnNotice`, `MissingRecommendedValueNotice` |
| `@PrimaryKey` | Values must be unique (single or composite key) | `DuplicateKeyNotice` |
| `@ForeignKey(table, field)` | Referenced value must exist in target table | `ForeignKeyViolationNotice` |
| `@NonNegative` | Numeric value >= 0 | `NumberOutOfRangeNotice` |
| `@Positive` | Numeric value > 0 | `NumberOutOfRangeNotice` |
| `@NonZero` | Numeric value != 0 | `NumberOutOfRangeNotice` |
| `@EndRange(field, allowEqual)` | Start field <= end field (or < if `!allowEqual`) | `StartAndEndRangeOutOfOrderNotice` |
| `@MixedCase` | String should not be ALL CAPS | `MixedCaseRecommendedFieldNotice` |
| `@CurrencyAmount(currencyField)` | Decimal format matches currency's minor unit count | `InvalidCurrencyAmountNotice` |
| `@DefaultValue(value)` | If field missing, use this default (not a validation — a data transform) | — |
| `@FieldType(LATITUDE)` | Range [-90, 90] | `InvalidFloatNotice` |
| `@FieldType(LONGITUDE)` | Range [-180, 180] | `InvalidFloatNotice` |
| `@FieldType(COLOR)` | 6-char hex without `#` | `InvalidColorNotice` |
| `@FieldType(URL)` | Valid URL format | `InvalidUrlNotice` |
| `@FieldType(EMAIL)` | Valid email format | `InvalidEmailNotice` |
| `@FieldType(PHONE_NUMBER)` | Country-specific phone validation | `InvalidPhoneNumberNotice` |
| `@FieldType(TIMEZONE)` | IANA timezone identifier | `InvalidTimezoneNotice` |

### Stage 7: Validator Categories

**Source:** `core/src/main/java/.../validator/ValidatorLoader.java`, `DefaultValidatorProvider.java`

Three validator categories, determined by constructor parameter analysis:

1. **Single-entity validators** (`SingleEntityValidator<T extends GtfsEntity>`):
   - Run on each parsed entity during CSV loading (inside the row loop)
   - Signature: `validate(T entity, NoticeContainer noticeContainer)`
   - Optional: `shouldCallValidate(ColumnInspector header) -> bool` to skip if columns missing
   - Mapped by entity type

2. **Single-file validators** (`FileValidator` with one table container dependency):
   - Run after entire table is loaded
   - Constructor receives a single `GtfsTableContainer` subclass
   - Signature: `validate(NoticeContainer noticeContainer)`
   - Optional: `shouldCallValidate() -> bool`

3. **Multi-file validators** (`FileValidator` with multiple table dependencies or `GtfsFeedContainer`):
   - Run after all tables loaded, executed in parallel
   - Constructor receives multiple table containers or `GtfsFeedContainer`
   - Same validate signature as single-file

**Dependency injection:**
- Constructor parameters determine dependencies
- `@Inject`-annotated constructor preferred over default
- Special injectable types: `CountryCode`, `CurrentDateTime`, `DateForValidation`
- If any dependency table failed to parse → validator skipped (tracked in `skippedValidators` multimap)

**Skipped validator reasons:**
- `SINGLE_ENTITY_VALIDATORS_WITH_ERROR` — entity validators skipped due to parse errors
- `SINGLE_FILE_VALIDATORS_WITH_ERROR` — file validators skipped due to parse errors
- `MULTI_FILE_VALIDATORS_WITH_ERROR` — multi-file validators skipped due to dependency parse errors
- `VALIDATORS_NO_NEED_TO_RUN` — `shouldCallValidate()` returned false

**Safe execution:** `ValidatorUtil.safeValidate()` wraps every validator call in try-catch. Runtime exceptions → `RuntimeExceptionInValidatorError` system error. Validator crashes never stop the pipeline.

### Stage 8: Feature Detection

**Source:** `main/src/main/java/.../reportsummary/model/FeedMetadata.java`, `FeatureMetadata.java`

39 features in 6 groups. Two detection approaches:

**File-based detection** (table exists and has >= 1 row):

| Feature | File | Group |
|---|---|---|
| Pathway Connections | pathways.txt | Pathways |
| Pathway Signs | pathways.txt | Pathways |
| Pathway Details | pathways.txt | Pathways |
| Levels | levels.txt | Pathways |
| Transfers | transfers.txt | Base Add-ons |
| Shapes | shapes.txt | Base Add-ons |
| Frequencies | frequencies.txt | Base Add-ons |
| Feed Information | feed_info.txt | Base Add-ons |
| Attributions | attributions.txt | Base Add-ons |
| Translations | translations.txt | Base Add-ons |
| Fares V1 | fare_attributes.txt | Fares |
| Fare Products | fare_products.txt | Fares |
| Fare Transfers | fare_transfer_rules.txt | Fares |
| Booking Rules | booking_rules.txt | Flexible Services |

**Field-based detection** (at least one entity with non-default field value):

| Feature | Table(s) | Field condition | Group |
|---|---|---|---|
| Route Colors | routes.txt | `route_color` or `route_text_color` present | Base Add-ons |
| Headsigns | trips.txt or stop_times.txt | `trip_headsign` or `stop_headsign` present | Base Add-ons |
| Stops Wheelchair Accessibility | stops.txt | `wheelchair_boarding` present | Accessibility |
| Trips Wheelchair Accessibility | trips.txt | `wheelchair_accessible` present | Accessibility |
| Text-to-Speech | stops.txt | `tts_stop_name` present | Accessibility |
| Bike Allowed | trips.txt | `bikes_allowed` present | Accessibility |
| Cars Allowed | trips.txt | `cars_allowed` present | Base Add-ons |
| Location Types | stops.txt | `location_type` present | Base Add-ons |
| Stop Access | stops.txt | `stop_access` present | Base Add-ons |
| Traversal Time | pathways.txt | `traversal_time` present | Pathways |
| Pathway Signs (field) | pathways.txt | `signposted_as` or `reversed_signposted_as` | Pathways |
| Pathway Details (field) | pathways.txt | `max_slope` or `min_width` or `length` or `stair_count` | Pathways |
| Continuous Stops | routes.txt or stop_times.txt | `continuous_pickup` or `continuous_dropoff` | Flexible Services |
| Fare Media | fare_media.txt + fare_products.txt | `fare_media_id` in fare_products | Fares |
| Rider Categories | rider_categories.txt + fare_products.txt | `rider_category_id` in fare_products | Fares |
| Route-Based Fares | routes.txt + fare_leg_rules.txt (+ networks.txt) | `network_id` linkage | Fares |
| Time-Based Fares | timeframes.txt + fare_leg_rules.txt | `from_timeframe_id` or `to_timeframe_id` | Fares |
| Zone-Based Fares | areas.txt + fare_leg_rules.txt | `from_area_id` or `to_area_id` | Fares |
| Zone-Based Demand Responsive | stop_times.txt | `location_id` without `stop_id` | Flexible Services |
| Fixed-Stops Demand Responsive | location_groups.txt + stop_times.txt | `location_group_id` without `stop_id` | Flexible Services |
| Predefined Routes with Deviation | stop_times.txt | `trip_id` + `location_id` + `stop_id` + `arrival_time` + `departure_time` all present | Flexible Services |

**Detection method:** `FeedMetadata.loadSpecFeatures(GtfsFeedContainer)` calls two sub-methods:
1. `loadSpecFeaturesBasedOnFilePresence()` — checks entity count > 0
2. `loadSpecFeaturesBasedOnFieldPresence()` — streams entities, applies field predicates

Results stored in `LinkedHashMap<FeatureMetadata, Boolean>`. Only features where value is `true` are included in the report's `gtfsFeatures` list.

### Stage 9: Report Generation

**Source:** `main/src/main/java/.../runner/ValidationRunner.java` (exportReport method)

Three output files produced:

1. **`report.json`** — validation report (JSON)
2. **`report.html`** — human-readable HTML report
3. **`system_errors.json`** — internal system errors (JSON)

Output directory created if it doesn't exist. Filenames can be overridden via CLI flags.

---

## 2. Notice System

### Notice Hierarchy

```
Notice (abstract)
├── ValidationNotice (abstract) — user-facing validation findings
│   ├── annotated with @GtfsValidationNotice(severity=...)
│   └── fields are public instance variables, serialized by GSON
└── SystemError (abstract) — internal errors, not exported to users
    └── always severity ERROR
```

### Severity Levels

| Level | Ordinal | Meaning |
|---|---|---|
| `INFO` | 0 | Unknown files/fields, non-impactful observations |
| `WARNING` | 1 | Quality issues, GTFS spec "should" recommendations |
| `ERROR` | 2 | Spec violations, RFC 2119 "must" requirements |

### NoticeContainer Capacity Limits

**Source:** `core/src/main/java/.../notice/NoticeContainer.java`

| Limit | Value | Purpose |
|---|---|---|
| `MAX_VALIDATION_NOTICES_TYPE_AND_SEVERITY` | 100,000 | Max notices stored per (code + severity) pair |
| `MAX_TOTAL_VALIDATION_NOTICES` | 10,000,000 | Total notices across all types (OOM prevention) |
| `MAX_EXPORTS_PER_NOTICE_TYPE_AND_SEVERITY` | 1,000 | Max sample notices exported per type in report.json |

**Behavior:**
- Notices exceeding per-type limit are silently dropped (count still tracked)
- Notices exceeding total limit are silently dropped
- System errors have no capacity limit (expected to be rare)
- `NoticeContainer` is NOT thread-safe — each thread gets its own, merged after completion via `addAll()`

### Notice Serialization

**Notice → JSON field mapping:**
- Each notice's public fields are serialized via GSON reflection
- Special type serializers:
  - `GtfsColor` → HTML color string (e.g., `"FFFFFF"`)
  - `GtfsDate` → `"YYYYMMDD"` string
  - `GtfsTime` → `"HH:MM:SS"` string
  - `S2LatLng` → `[lat, lng]` array

**Notice code derivation:**
- Class name → snake_case (e.g., `ForeignKeyViolationNotice` → `foreign_key_violation`)
- Suffix `Notice` or `Error` is stripped

### Notice Grouping for Export

Notices are grouped by `code + severity.ordinal()` (mapping key).

For each group:
- `code`: snake_case notice identifier
- `severity`: `"ERROR"` | `"WARNING"` | `"INFO"`
- `totalNotices`: count of all notices in this group (not just exported samples)
- `sampleNotices`: array of up to 1,000 serialized notice objects

---

## 3. Report JSON Format

### report.json Structure

```json
{
  "summary": {
    "validatorVersion": "6.0.0",
    "validatedAt": "2024-01-15T10:30:00.000Z",
    "gtfsInput": "/path/to/feed.zip",
    "threads": 4,
    "outputDirectory": "/path/to/output",
    "systemErrorsReportName": "system_errors.json",
    "validationReportName": "report.json",
    "htmlReportName": "report.html",
    "countryCode": "US",
    "dateForValidation": "2024-01-15",
    "feedInfo": {
      "publisherName": "Example Transit",
      "publisherUrl": "https://example.com",
      "feedLanguage": "en",
      "feedStartDate": "20240101",
      "feedEndDate": "20240630"
    },
    "agencies": [
      {
        "name": "Example Transit Agency",
        "url": "https://example.com",
        "phone": "555-0100",
        "email": "transit@example.com"
      }
    ],
    "files": ["agency.txt", "stops.txt", "routes.txt", "trips.txt", "stop_times.txt"],
    "counts": {
      "Shapes": 150,
      "Stops": 2400,
      "Routes": 45,
      "Trips": 3200,
      "Agencies": 1,
      "Blocks": 800
    },
    "gtfsFeatures": [
      "Shapes",
      "Route Colors",
      "Headsigns",
      "Wheelchair Accessibility"
    ],
    "validationTimeSeconds": 12.345,
    "memoryUsageRecords": [...]
  },
  "notices": [
    {
      "code": "foreign_key_violation",
      "severity": "ERROR",
      "totalNotices": 42,
      "sampleNotices": [
        {
          "filename": "stop_times.txt",
          "csvRowNumber": 15,
          "fieldName": "trip_id",
          "fieldValue": "TRIP_999",
          "foreignFilename": "trips.txt",
          "foreignFieldName": "trip_id"
        }
      ]
    }
  ]
}
```

### system_errors.json Structure

```json
{
  "notices": [
    {
      "code": "runtime_exception_in_validator",
      "severity": "ERROR",
      "totalNotices": 1,
      "sampleNotices": [
        {
          "validatorClassName": "ShapeToStopMatchingValidator",
          "exception": "java.lang.NullPointerException",
          "message": "..."
        }
      ]
    }
  ]
}
```

### FeedMetadata Entity Counts

**Source:** `FeedMetadata.java`

The `counts` map is populated from:
- `Shapes` → `shapes.txt` entity count
- `Stops` → `stops.txt` entity count
- `Routes` → `routes.txt` entity count
- `Trips` → `trips.txt` entity count
- `Agencies` → `agency.txt` entity count
- `Blocks` → distinct `block_id` values from `trips.txt`

### Service Window Detection

**Source:** `FeedMetadata.loadServiceWindow()`

Determines feed date range from:
1. `calendar.txt` → min `start_date`, max `end_date`
2. `calendar_dates.txt` → min/max `date` values
3. Combined → overall min start, max end

Three cases: calendar only, calendar_dates only, or both.

---

## 4. Table Schema Registry

32 GTFS tables defined as schema interfaces:

| # | Filename | Required | Single row | Notes |
|---|---|---|---|---|
| 1 | agency.txt | yes | no | agencies |
| 2 | stops.txt | conditional | no | stops, stations, entrances, nodes |
| 3 | routes.txt | yes | no | transit routes |
| 4 | trips.txt | yes | no | trips on routes |
| 5 | stop_times.txt | yes | no | stop arrival/departure times (high-volume) |
| 6 | calendar.txt | conditional | no | weekly service patterns |
| 7 | calendar_dates.txt | conditional | no | service exceptions |
| 8 | fare_attributes.txt | no | no | fare v1 pricing |
| 9 | fare_rules.txt | no | no | fare v1 zone rules |
| 10 | timeframes.txt | no | no | fare v2 time periods |
| 11 | fare_media.txt | no | no | fare v2 payment media |
| 12 | fare_products.txt | no | no | fare v2 products |
| 13 | fare_leg_rules.txt | no | no | fare v2 leg pricing |
| 14 | fare_leg_join_rules.txt | no | no | fare v2 leg joins |
| 15 | fare_transfer_rules.txt | no | no | fare v2 transfer pricing |
| 16 | areas.txt | no | no | geographic areas |
| 17 | stop_areas.txt | no | no | stop-to-area mapping |
| 18 | rider_categories.txt | no | no | fare v2 rider types |
| 19 | networks.txt | no | no | route networks |
| 20 | route_networks.txt | no | no | route-to-network mapping |
| 21 | shapes.txt | conditional | no | geographic shapes |
| 22 | frequencies.txt | no | no | headway-based service |
| 23 | transfers.txt | no | no | transfer rules |
| 24 | pathways.txt | no | no | station pathways |
| 25 | levels.txt | no | no | station levels |
| 26 | translations.txt | no | no | text translations |
| 27 | feed_info.txt | conditional | **yes** | feed metadata (single row) |
| 28 | attributions.txt | no | no | dataset attributions |
| 29 | booking_rules.txt | no | no | demand-responsive booking |
| 30 | location_groups.txt | no | no | location groupings |
| 31 | location_group_stops.txt | no | no | stop-to-location-group mapping |
| 32 | locations.geojson | no | — | GeoJSON locations (not CSV) |

### Schema Annotation Summary

Each table schema interface declares fields with annotations that drive load-time validation:

**Field-level annotations and their orchestrator behavior:**

- `@Required` → column header must exist, every row must have non-empty value
- `@Recommended` → column header should exist, emit warning if missing
- `@ConditionallyRequired` → custom validator handles the logic (not orchestrator)
- `@PrimaryKey` → deduplicate check after loading; composite keys use multiple `@PrimaryKey` fields
- `@PrimaryKey(isSequenceUsedForSorting=true)` → entities sorted by this field within key groups
- `@ForeignKey(table="x.txt", field="y")` → post-load cross-table existence check
- `@Index` → build lookup index (e.g., `byRouteId()`) for validator use
- `@NonNegative`, `@Positive`, `@NonZero` → numeric bounds checked during field parsing
- `@EndRange(field="x", allowEqual=bool)` → start/end range validated after both fields parsed
- `@MixedCase` → warning if value is all uppercase
- `@CurrencyAmount(currencyField="x")` → decimal precision matches currency
- `@DefaultValue("x")` → substitute default when field is empty
- `@CachedField` → memory optimization (intern string values)
- `@FieldType(...)` → determines parse method and format validation

### Polars Type Mapping

For the Python orchestrator, Java types map to Polars dtypes:

| Java type | FieldType annotation | Polars dtype | Notes |
|---|---|---|---|
| String | ID, TEXT | `pl.Utf8` | |
| String | URL, EMAIL, PHONE_NUMBER | `pl.Utf8` | validated as string |
| String | COLOR | `pl.Utf8` | 6-char hex |
| int | INTEGER | `pl.Int32` | or `pl.Int64` for large values |
| int | ENUM | `pl.Int32` | enum integer codes |
| double | FLOAT | `pl.Float64` | |
| BigDecimal | DECIMAL | `pl.Float64` | or `pl.Decimal` |
| GtfsDate | DATE | `pl.Utf8` | keep as YYYYMMDD string for GTFS compat |
| GtfsTime | TIME | `pl.Utf8` | keep as H:MM:SS string (hours > 23) |
| ZoneId | TIMEZONE | `pl.Utf8` | |
| Locale | LANGUAGE_CODE | `pl.Utf8` | |
| Currency | CURRENCY_CODE | `pl.Utf8` | |
| double | LATITUDE | `pl.Float64` | range [-90, 90] |
| double | LONGITUDE | `pl.Float64` | range [-180, 180] |
| boolean | — | `pl.Boolean` | |

**Important:** `GtfsTime` values allow hours > 23 (e.g., `25:30:00` for next-day service). Must NOT use Polars time/duration types — store as string or total seconds integer.

**Important:** `GtfsDate` uses `YYYYMMDD` format (no separators). Can be stored as `pl.Date` if converted, but raw string comparison is simpler for the GTFS domain.

---

## 5. Feed Container Contract

**Source:** `core/src/main/java/.../table/GtfsFeedContainer.java`, `GtfsTableContainer.java`

### GtfsFeedContainer

The top-level container holding all loaded tables:

- `getTableForFilename(String) -> GtfsEntityContainer` — case-insensitive filename lookup
- `getTable(Class<T>) -> T` — lookup by container class (Java-specific; Python uses dict)
- `getTables() -> Collection<GtfsEntityContainer>` — all containers
- `isParsedSuccessfully() -> bool` — true only if ALL tables parsed successfully
- `tableTotalsText() -> String` — formatted entity count summary

### GtfsTableContainer

Per-table container:

- `getEntities() -> List<T>` — all parsed entities
- `entityCount() -> int` — number of entities
- `getHeader() -> CsvHeader` — column headers from CSV
- `hasColumn(String) -> bool` — checks if column exists in CSV
- `getKeyColumnNames() -> List<String>` — primary key column names
- `isParsedSuccessfully() -> bool` — table status check
- `tableStatus` enum: `MISSING_FILE`, `EMPTY_FILE`, `UNPARSABLE_ROWS`, `PARSABLE_HEADERS_AND_ROWS`
- Indexed lookups generated from `@PrimaryKey` and `@Index` annotations

**Python equivalent:** `dict[str, pl.DataFrame]` where key is filename (e.g., `"stops.txt"`). Table status tracked separately. Column presence checked via DataFrame schema. Primary key and index lookups are Polars filter/join operations.

---

## 6. Threading Model

**Source:** `GtfsFeedLoader.java`

- `ExecutorService` with fixed thread pool (size = `--threads` flag, default 1)
- Phase 1 (table loading): each table loaded as independent callable
- Phase 2 (multi-file validators): each validator executed as independent callable
- `NoticeContainer` per thread, merged after completion via `addAll()`
- Thread pool shared between phases (created once, shut down after both phases)

**Python equivalent:** Polars handles intra-query parallelism natively. For inter-table loading, use `ThreadPoolExecutor` or Polars' built-in CSV reading (which is already parallel). For multi-file validators, use `concurrent.futures.ProcessPoolExecutor` or keep sequential if Polars operations are already parallel.

---

## 7. Error Handling

### System Errors vs Validation Notices

- **Validation notices**: user-facing, capacity-limited, exported in `report.json`
- **System errors**: internal errors (IO failures, validator crashes), exported in `system_errors.json`, no capacity limit

### Specific Error Handling Patterns

1. **CSV parse failure**: `CsvParsingFailedNotice` → table container created with `UNPARSABLE_ROWS` status → all validators depending on this table are skipped
2. **IO error on input close**: `IOError` system error → logged but doesn't stop pipeline
3. **Validator runtime exception**: caught by `ValidatorUtil.safeValidate()` → `RuntimeExceptionInValidatorError` → other validators continue
4. **Download failure**: exception propagated to `ValidationRunner.run()` → returns `EXCEPTION` status
5. **Missing required file**: `MissingRequiredFileNotice` (ERROR) → empty container created → dependent validators may skip or handle empty input

---

## 8. Orchestrator-Owned vs Validator-Owned Checks

The line between orchestrator responsibility and validator responsibility:

### Orchestrator owns (schema-driven, during loading):
- File presence/absence detection
- CSV parsing and header validation
- Required/recommended field presence
- Type conversion and format validation
- Primary key uniqueness
- Foreign key existence (cross-table)
- Numeric bounds (@NonNegative, @Positive, @NonZero)
- Start/end range ordering (@EndRange)
- Mixed case checks (@MixedCase)
- Currency amount precision (@CurrencyAmount)
- Default value substitution (@DefaultValue)
- Field format validation (URL, email, phone, color, lat/lon, timezone, language, currency)

### Validators own (business logic, post-loading):
- Cross-table semantic checks (e.g., route agency consistency)
- Temporal logic (e.g., feed expiration, service gaps)
- Spatial logic (e.g., shape-to-stop matching)
- Graph logic (e.g., pathway loops, reachability)
- Domain-specific rules (e.g., block overlap detection, color contrast)
- Conditional requirement logic (e.g., fields required only when another field has specific value)

---

## 9. Version Checking

**Source:** `ValidationRunner.java`, `VersionResolver.java`

- On startup, checks for newer validator version via `VersionResolver`
- Logs update notification if newer version available
- Can be skipped with `--skip_validator_update`
- Version info included in report summary

**Python equivalent:** Optional — can be deferred or replaced with PyPI version check.

---

## 10. HTML Report

**Source:** `HtmlReportGenerator.java`

Uses Thymeleaf template engine with:
- `metadata` — FeedMetadata (counts, agencies, service window, features)
- `summary` — ReportSummary (notices grouped by severity, then by code)
- `config` — validation configuration
- `date` — validation date
- `is_different_date` — whether CLI date differs from today
- `uniqueFieldsByCode` — discovered fields per notice type

**Python equivalent:** Jinja2 template with equivalent data model. The HTML template itself can be ported or redesigned.

---

## 11. Notice Schema Export

**Source:** `NoticeSchemaGenerator.java`

When `--export_notices_schema` flag is set:
1. Discovers all `Notice` subclasses via ClassGraph
2. For each notice:
   - Extracts severity from `@GtfsValidationNotice` annotation
   - Loads documentation from resource files
   - Introspects fields via reflection
   - Maps field types: int→INTEGER, double→NUMBER, bool→BOOLEAN, enum→ENUM, String→STRING, S2LatLng→ARRAY[NUMBER,2]
   - Captures deprecation info and replacement notices
   - Builds references (files, best practices, spec sections, URLs)
3. Writes `notice_schema.json` to output directory
4. Optionally exits after export (no validation run)

**Python equivalent:** Build schema from notice dataclass definitions using `dataclasses.fields()` or similar introspection.
