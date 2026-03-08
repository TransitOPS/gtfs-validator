# 001 — Orchestrator Implementation Spec

Derived from [001-orchestrator-analysis.md](../analysis/001-orchestrator-analysis.md). Defines the Python/Polars orchestrator that replaces the Java validation pipeline.

---

## 1. Module Layout

```
src/gtfs_validator/
├── cli.py                 # CLI entry point (click/typer)
├── config.py              # Immutable ValidationConfig dataclass
├── input.py               # GTFS input abstraction (ZIP/directory/URL)
├── loading.py             # Polars-based CSV loading + schema enforcement
├── load_validators.py     # Schema-driven load-time checks
├── schemas.py             # Table schema registry (pure data)
├── notices.py             # Notice dataclass + NoticeContainer
├── context.py             # ValidationContext for validators
├── validators/            # Cross-table business-logic validators
│   └── __init__.py        # Explicit validator registry
├── features.py            # Feature detection (~39 checks)
├── report.py              # JSON + HTML report generation
└── runner.py              # Orchestration pipeline
```

---

## 2. Pipeline Stages

The orchestrator executes a strict sequential pipeline matching the Java validator's stage order:

```
CLI parse → Config build → Input open → Table load (parallel) →
Load-time validate (during load) → Multi-file validate (parallel) →
Feature detection → Report generation → Cleanup
```

Each stage is a function called from `runner.run()`. Stages never skip silently — failures are captured as system errors or validation notices and the pipeline continues.

---

## 3. CLI (`cli.py`)

### Entry Point

Use `click` (preferred for its maturity) or `typer`. Single command, no subcommands.

```python
@click.command()
@click.option("-i", "--input", "input_path", type=click.Path(), default=None)
@click.option("-u", "--url", type=str, default=None)
@click.option("-o", "--output_base", type=click.Path(), required=True)
@click.option("-s", "--storage_directory", type=click.Path(), default=None)
@click.option("-t", "--threads", type=int, default=1)
@click.option("-c", "--country_code", type=str, default=None)
@click.option("-d", "--date", "date_for_validation", type=str, default=None)
@click.option("-v", "--validation_report_name", type=str, default=None)
@click.option("-r", "--html_report_name", type=str, default=None)
@click.option("-e", "--system_errors_report_name", type=str, default=None)
@click.option("-p", "--pretty", is_flag=True, default=False)
@click.option("-n", "--export_notices_schema", is_flag=True, default=False)
def main(...) -> None: ...
```

### Argument Validation

- Exactly one of `--input` or `--url` must be provided (not both, not neither)
- Exception: `--export_notices_schema` bypasses the input requirement — exports schema and exits
- `--storage_directory` is only valid when `--url` is provided; error if used with `--input`
- `--date` must be ISO 8601 (`YYYY-MM-DD`) if provided; parse with `datetime.date.fromisoformat()`
- `--country_code` must be ISO 3166-1 alpha-2 (2 uppercase letters) if provided; default `"ZZ"`
- `--threads` must be >= 1

### Exit Codes

| Code | Meaning |
|------|---------|
| `0` | Validation completed successfully |
| `1` | Validation completed but system errors occurred |
| `2` | Fatal exception during validation |

Implementation: `runner.run()` returns a `RunResult` enum. `cli.py` maps it to `sys.exit()`.

```python
class RunResult(enum.IntEnum):
    SUCCESS = 0
    SYSTEM_ERRORS = 1
    EXCEPTION = 2
```

### Dropped Java Flags

| Flag | Reason |
|------|--------|
| `--skip_validator_update` | No version-check mechanism needed in Python; users manage versions via `pip`/`uv` |
| `--help` | Provided automatically by click/typer |

---

## 4. Configuration (`config.py`)

Immutable dataclass built from CLI arguments. Constructed once, passed through the pipeline.

```python
from dataclasses import dataclass
from datetime import date
from pathlib import Path

@dataclass(frozen=True)
class ValidationConfig:
    gtfs_source: str                          # file path or URL
    output_directory: Path
    storage_directory: Path | None = None
    validation_report_name: str = "report.json"
    html_report_name: str = "report.html"
    system_errors_report_name: str = "system_errors.json"
    num_threads: int = 1
    country_code: str = "ZZ"
    date_for_validation: date = field(default_factory=date.today)
    pretty_json: bool = False
```

### Construction Rules

- `gtfs_source` stores the raw `--input` path or `--url` string
- `output_directory` is created (with `parents=True, exist_ok=True`) at report-generation time, not at config construction
- Report filenames default to `"report.json"`, `"report.html"`, `"system_errors.json"` — overridden by CLI flags if provided
- `date_for_validation` defaults to `date.today()` at invocation time

---

## 5. Input Handling (`input.py`)

### Abstract Contract

```python
from typing import Protocol

class GtfsInput(Protocol):
    def filenames(self) -> set[str]: ...
    def open_file(self, filename: str) -> IO[bytes]: ...
    def close(self) -> None: ...
```

Use as a context manager (`__enter__`/`__exit__`) for resource cleanup.

### Three Input Modes

**1. Local directory** (`--input` pointing to a directory):
- Walk directory for regular files (non-recursive — GTFS feeds are flat)
- Build filename set from file basenames
- `open_file()` returns `open(path, "rb")`

**2. Local or downloaded ZIP** (`--input` pointing to a `.zip`, or `--url` + `--storage_directory`):
- Open with `zipfile.ZipFile`
- Filter entries: skip directories (names ending with `/`), skip `__MACOSX/` prefixes, skip `.DS_Store`
- Strip directory prefixes from entry names (e.g., `feed/stops.txt` -> `stops.txt`)
- Detect GTFS files in subdirectories → emit `SubdirectoryTransitFeedNotice` (INFO)
- `open_file()` returns `zipfile.open(entry_name)`

**3. URL in-memory** (`--url` without `--storage_directory`):
- Download with `urllib.request.urlopen()` or `httpx` (prefer `httpx` for timeouts, redirects, and modern TLS)
- Read entire response into `io.BytesIO`
- Open as `zipfile.ZipFile` from the `BytesIO` object
- Same ZIP filtering rules as mode 2

**4. URL with disk storage** (`--url` + `--storage_directory`):
- Download to `storage_directory / <filename>` on disk
- Open the on-disk file as `zipfile.ZipFile`
- Same ZIP filtering rules as mode 2

### Security

- Validate ZIP entry paths to prevent zip-slip (reject entries with `..` or absolute paths)
- Bound in-memory download size (configurable, e.g., 2 GB default) to prevent OOM
- Treat all input as untrusted

---

## 6. Table Loading (`loading.py`)

### Schema-Driven CSV Parsing

For each filename in the GTFS input:
1. Look up the table definition in the schema registry (`schemas.py`)
2. If no schema found → emit `UnknownFileNotice` (INFO), skip parsing
3. If schema found → parse CSV with Polars

For each known table NOT in the input:
- Create an empty `pl.DataFrame` with the correct schema
- If table is required → emit `MissingRequiredFileNotice` (ERROR)
- If table is recommended → emit `MissingRecommendedFileNotice` (WARNING)

### Polars CSV Reading

```python
def load_table(
    gtfs_input: GtfsInput,
    table_def: TableDefinition,
) -> tuple[pl.DataFrame, TableStatus, list[Notice]]:
    ...
```

Reading strategy:
- Read all columns as `pl.Utf8` initially (preserves raw values for validation)
- Then cast to target dtypes column by column, emitting type-conversion notices for failures
- This two-phase approach allows validating the raw string values (leading/trailing whitespace, empty strings) before conversion

```python
# Phase 1: Read raw
raw_df = pl.read_csv(
    source=byte_stream,
    has_header=True,
    dtypes={col: pl.Utf8 for col in expected_columns},
    ignore_errors=False,  # fail on malformed CSV
    truncate_ragged_lines=False,
    encoding="utf8-lossy",
)

# Phase 2: Validate headers, then cast per-column
```

### Header Validation

After reading the CSV header row:
- Required columns missing → `MissingRequiredColumnNotice` (ERROR)
- Recommended columns missing → `MissingRecommendedColumnNotice` (WARNING)
- Unknown columns (not in schema) → `UnknownColumnNotice` (INFO)
- Duplicate columns → `DuplicateColumnNotice` (ERROR)
- Empty file (header only, no data rows) → `EmptyFileNotice` (ERROR for required tables, WARNING for recommended)

### Row-Level Parsing

For each column, apply type-specific parsing and validation. Use Polars expressions for vectorized checks wherever possible.

**Whitespace handling (vectorized):**
```python
# Detect leading/trailing whitespace
has_whitespace = col.str.strip_chars() != col
# Emit LeadingOrTrailingWhitespacesNotice (WARNING) for affected rows
# Then trim all values
trimmed = col.str.strip_chars()
```

**Empty/null semantics:**
- Empty string `""` → treated as null/missing
- Whitespace-only string → treated as missing (after trim)
- Normalize: after trimming, replace `""` with `null`

**Row length validation:**
- Polars `read_csv` handles this natively if `truncate_ragged_lines=False`
- On parse error, capture row numbers and emit `InvalidRowLengthNotice` (ERROR)
- If the entire CSV is unparseable, emit `CsvParsingFailedNotice` (ERROR) and return an invalid container

### Type Conversion Rules

Each column is cast from `pl.Utf8` to its target type. Failed conversions emit a notice and set the value to `null`.

| Schema Type | Polars Target | Validation | Notice on Failure |
|---|---|---|---|
| `ID` | `pl.Utf8` | optional field-level validator | `InvalidIdNotice` |
| `TEXT` | `pl.Utf8` | required/recommended presence | `MissingRequired/RecommendedValueNotice` |
| `URL` | `pl.Utf8` | URL format regex | `InvalidUrlNotice` |
| `EMAIL` | `pl.Utf8` | email format regex | `InvalidEmailNotice` |
| `PHONE` | `pl.Utf8` | country-specific (via `phonenumbers` lib or regex) | `InvalidPhoneNumberNotice` |
| `INTEGER` | `pl.Int32` | numeric parse + bounds | `InvalidIntegerNotice`, `NumberOutOfRangeNotice` |
| `FLOAT` | `pl.Float64` | numeric parse + bounds | `InvalidFloatNotice`, `NumberOutOfRangeNotice` |
| `DECIMAL` | `pl.Float64` | numeric parse | `InvalidDecimalNotice` |
| `LATITUDE` | `pl.Float64` | range [-90, 90] + near-origin check | `InvalidFloatNotice`, `PointNearOriginNotice` |
| `LONGITUDE` | `pl.Float64` | range [-180, 180] + near-origin check | `InvalidFloatNotice`, `PointNearOriginNotice` |
| `COLOR` | `pl.Utf8` | 6-char hex, no `#` prefix | `InvalidColorNotice` |
| `DATE` | `pl.Utf8` | `YYYYMMDD` format, valid calendar date | `InvalidDateNotice` |
| `TIME` | `pl.Utf8` | `H:MM:SS` or `HH:MM:SS`, hours may exceed 23 | `InvalidTimeNotice` |
| `TIMEZONE` | `pl.Utf8` | IANA timezone (validate with `zoneinfo.available_timezones()`) | `InvalidTimezoneNotice` |
| `LANGUAGE` | `pl.Utf8` | BCP 47 tag (validate with `langcodes` or regex) | `InvalidLanguageCodeNotice` |
| `CURRENCY` | `pl.Utf8` | ISO 4217 3-letter code | `InvalidCurrencyNotice` |
| `ENUM` | `pl.Int32` | integer within defined enum range | `UnexpectedEnumValueNotice` |

**Critical:** `TIME` values allow hours > 23 (e.g., `25:30:00` for next-day service). Never use `pl.Time` or `pl.Duration`. Store as `pl.Utf8` or convert to total-seconds `pl.Int32` for arithmetic.

**Critical:** `DATE` values use `YYYYMMDD` (no separators). Store as `pl.Utf8` for GTFS compatibility. Convert to `pl.Date` only when date arithmetic is needed in validators.

### Numeric Bounds Checking

Applied during type conversion based on schema annotations:

| Constraint | Check | Notice |
|---|---|---|
| `NON_NEGATIVE` | value >= 0 | `NumberOutOfRangeNotice` |
| `POSITIVE` | value > 0 | `NumberOutOfRangeNotice` |
| `NON_ZERO` | value != 0 | `NumberOutOfRangeNotice` |

Use Polars filter expressions:
```python
# Example: NON_NEGATIVE check
violations = df.filter(pl.col(field_name) < 0)
```

### Table Status

Each loaded table carries a status:

```python
class TableStatus(enum.Enum):
    MISSING_FILE = "missing_file"
    EMPTY_FILE = "empty_file"
    UNPARSABLE_ROWS = "unparsable_rows"
    PARSABLE = "parsable"
```

Status determines whether downstream validators run against this table.

### Parallel Table Loading

Use `concurrent.futures.ThreadPoolExecutor(max_workers=config.num_threads)` for loading tables in parallel. Each table load produces its own notice list; merge after all tables complete.

Polars `read_csv` releases the GIL during I/O, making threads effective here.

---

## 7. Load-Time Validation (`load_validators.py`)

Schema-driven checks executed after each table is loaded. These replace Java's annotation-generated validators.

### Primary Key Uniqueness

```python
def check_primary_key(
    df: pl.DataFrame,
    key_columns: list[str],
    filename: str,
) -> list[Notice]:
```

- Single-column PK: `df.filter(pl.col(key).is_duplicated())`
- Composite PK: `df.group_by(key_columns).count().filter(pl.col("count") > 1)`
- Emit `DuplicateKeyNotice` (ERROR) for each duplicate group
- Notice fields: `filename`, `csvRowNumber`, key column values

### Foreign Key Existence

```python
def check_foreign_key(
    source_df: pl.DataFrame,
    source_filename: str,
    source_field: str,
    target_df: pl.DataFrame,
    target_filename: str,
    target_field: str,
) -> list[Notice]:
```

- Use `anti_join` to find source values not in target:
  ```python
  orphans = source_df.join(
      target_df.select(pl.col(target_field).unique()),
      left_on=source_field,
      right_on=target_field,
      how="anti",
  )
  ```
- Emit `ForeignKeyViolationNotice` (ERROR) per orphan row
- Notice fields: `filename`, `csvRowNumber`, `fieldName`, `fieldValue`, `foreignFilename`, `foreignFieldName`
- Skip if target table has status `UNPARSABLE_ROWS` or `MISSING_FILE`

### Range Ordering (`@EndRange`)

```python
def check_end_range(
    df: pl.DataFrame,
    filename: str,
    start_field: str,
    end_field: str,
    allow_equal: bool,
) -> list[Notice]:
```

- If `allow_equal`: check `start > end`
- If not `allow_equal`: check `start >= end`
- Emit `StartAndEndRangeOutOfOrderNotice` (ERROR)

### Mixed Case (`@MixedCase`)

- Check if string value is all uppercase: `col.str.to_uppercase() == col`
- Emit `MixedCaseRecommendedFieldNotice` (WARNING)

### Currency Amount Precision (`@CurrencyAmount`)

- Look up currency's minor unit count (e.g., USD = 2, JPY = 0)
- Validate decimal places match
- Emit `InvalidCurrencyAmountNotice` (ERROR)

### Default Value Substitution (`@DefaultValue`)

- Replace null values with the schema-defined default
- This is a data transform, not a validation — no notice emitted
- Applied after null normalization, before other checks

### Near-Origin Check

- When both latitude and longitude are present and both are close to (0, 0):
  `abs(lat) < 1.0 and abs(lon) < 1.0`
- Emit `PointNearOriginNotice` (WARNING)

---

## 8. Validator Dispatch

### Validator Categories

**Single-entity validators** (Java concept) are replaced by vectorized Polars expressions in `load_validators.py`. No row-by-row Python loops.

**Single-file validators** become functions that take a single DataFrame:
```python
def validate_stops_file(
    stops: pl.DataFrame,
    ctx: ValidationContext,
) -> list[Notice]:
```

**Multi-file validators** (cross-table) are the primary validator type in Python:
```python
def validate_foreign_key_consistency(
    feed: dict[str, pl.DataFrame],
    ctx: ValidationContext,
) -> list[Notice]:
```

### Validator Registry

Explicit, ordered list in `validators/__init__.py`:

```python
from gtfs_validator.context import ValidationContext
from gtfs_validator.notices import Notice

ValidatorFn = Callable[
    [dict[str, pl.DataFrame], ValidationContext],
    list[Notice],
]

VALIDATOR_REGISTRY: list[ValidatorEntry] = [
    ValidatorEntry(
        name="feed_expiration_date",
        fn=validate_feed_expiration_date,
        requires=["feed_info.txt", "calendar.txt"],
    ),
    # ... all validators listed explicitly
]
```

### Dependency-Based Skipping

```python
@dataclass(frozen=True)
class ValidatorEntry:
    name: str
    fn: ValidatorFn
    requires: list[str]  # table filenames this validator needs
```

Before running a validator:
- Check that all required tables have status `PARSABLE` or `EMPTY_FILE`
- If any required table has status `UNPARSABLE_ROWS` or `MISSING_FILE` → skip the validator
- Track skipped validators and reasons for the system errors report

### Safe Execution

Every validator call is wrapped in a try/except:

```python
def safe_validate(entry: ValidatorEntry, feed, ctx) -> list[Notice]:
    try:
        return entry.fn(feed, ctx)
    except Exception as exc:
        return [RuntimeExceptionInValidatorError(
            validator_name=entry.name,
            exception=type(exc).__name__,
            message=str(exc),
        )]
```

Runtime exceptions produce `RuntimeExceptionInValidatorError` system errors. Validator crashes never stop the pipeline.

### Parallel Validator Execution

Multi-file validators run in parallel using `concurrent.futures.ThreadPoolExecutor(max_workers=config.num_threads)`.

Each validator receives its own notice list — no shared mutable state. Merge after all validators complete.

---

## 9. Notice System (`notices.py`)

### Notice Base

```python
from dataclasses import dataclass, field
from enum import IntEnum

class Severity(IntEnum):
    INFO = 0
    WARNING = 1
    ERROR = 2

@dataclass(frozen=True)
class Notice:
    code: str           # snake_case identifier (e.g., "foreign_key_violation")
    severity: Severity
    fields: dict[str, Any]  # context fields serialized to JSON
```

Validator-specific notice constructors are factory functions or subclasses that set `code`, `severity`, and required `fields`.

### System Errors

```python
@dataclass(frozen=True)
class SystemError:
    code: str
    fields: dict[str, Any]
    severity: Severity = Severity.ERROR  # always ERROR
```

System errors go to `system_errors.json`, never to `report.json`.

### NoticeContainer

```python
class NoticeContainer:
    MAX_PER_TYPE_AND_SEVERITY: int = 100_000
    MAX_TOTAL: int = 10_000_000
    MAX_EXPORTS_PER_TYPE: int = 1_000
```

**Behavior:**
- Notices exceeding the per-type limit are silently dropped (count still tracked)
- Notices exceeding the total limit are silently dropped
- System errors have no capacity limit
- Container is **not** thread-safe — each thread gets its own, merged after completion

```python
class NoticeContainer:
    def __init__(self) -> None:
        self._notices: dict[tuple[str, Severity], list[Notice]] = defaultdict(list)
        self._counts: dict[tuple[str, Severity], int] = defaultdict(int)
        self._total: int = 0
        self._system_errors: list[SystemError] = []

    def add(self, notice: Notice) -> None:
        key = (notice.code, notice.severity)
        self._counts[key] += 1
        if self._total >= self.MAX_TOTAL:
            return
        if len(self._notices[key]) >= self.MAX_PER_TYPE_AND_SEVERITY:
            return
        self._notices[key].append(notice)
        self._total += 1

    def add_system_error(self, error: SystemError) -> None:
        self._system_errors.append(error)

    def merge(self, other: "NoticeContainer") -> None:
        """Merge another container into this one (post-thread-completion)."""
        for key, notices in other._notices.items():
            for notice in notices:
                self.add(notice)
        self._system_errors.extend(other._system_errors)
        # Counts must also be merged accurately
        for key, count in other._counts.items():
            self._counts[key] += count
```

### Notice Code Derivation

Notice codes are `snake_case` strings. Convention: derive from class name by stripping `Notice`/`Error` suffix and converting to snake_case.

Example: `ForeignKeyViolationNotice` -> `"foreign_key_violation"`

### Notice Field Serialization

Special types serialize as:
- `GtfsDate` (string) → `"YYYYMMDD"` string
- `GtfsTime` (string) → `"HH:MM:SS"` string
- `GtfsColor` (string) → `"FFFFFF"` (6-char hex)
- Lat/lng pair → `[lat, lng]` array
- All other fields → JSON-native types (str, int, float, bool)

---

## 10. Feature Detection (`features.py`)

Runs after all tables are loaded and validated. Two detection methods:

### File-Based Detection

Check if table exists and has >= 1 row:

```python
FILE_BASED_FEATURES: list[tuple[str, str, str]] = [
    # (feature_name, filename, group)
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
```

### Field-Based Detection

Check if at least one row has a non-null, non-default value for a specific field:

```python
def has_non_default_values(df: pl.DataFrame, column: str) -> bool:
    if column not in df.columns:
        return False
    return df.select(pl.col(column).is_not_null().any()).item()
```

Full list of 27 field-based features as defined in the analysis (Section 8).

### Output

```python
def detect_features(
    feed: dict[str, pl.DataFrame],
) -> list[str]:
    """Return list of detected feature names (only features that are present)."""
```

Result is a `list[str]` of feature names, stored in `report.json` → `summary.gtfsFeatures`.

---

## 11. Report Generation (`report.py`)

### report.json

```python
def generate_validation_report(
    config: ValidationConfig,
    feed: dict[str, pl.DataFrame],
    notices: NoticeContainer,
    features: list[str],
    validation_time_seconds: float,
) -> dict[str, Any]:
```

Structure:

```json
{
  "summary": {
    "validatorVersion": "<package version>",
    "validatedAt": "<ISO 8601 UTC timestamp>",
    "gtfsInput": "<source path or URL>",
    "threads": 4,
    "outputDirectory": "<output path>",
    "systemErrorsReportName": "system_errors.json",
    "validationReportName": "report.json",
    "htmlReportName": "report.html",
    "countryCode": "US",
    "dateForValidation": "2024-01-15",
    "feedInfo": { ... },
    "agencies": [ ... ],
    "files": ["agency.txt", "stops.txt", ...],
    "counts": {
      "Shapes": 150,
      "Stops": 2400,
      "Routes": 45,
      "Trips": 3200,
      "Agencies": 1,
      "Blocks": 800
    },
    "gtfsFeatures": ["Shapes", "Route Colors", ...],
    "validationTimeSeconds": 12.345
  },
  "notices": [
    {
      "code": "foreign_key_violation",
      "severity": "ERROR",
      "totalNotices": 42,
      "sampleNotices": [ ... ]
    }
  ]
}
```

### Entity Counts

Populated from loaded DataFrames:
- `Shapes` → `len(feed["shapes.txt"])`
- `Stops` → `len(feed["stops.txt"])`
- `Routes` → `len(feed["routes.txt"])`
- `Trips` → `len(feed["trips.txt"])`
- `Agencies` → `len(feed["agency.txt"])`
- `Blocks` → `feed["trips.txt"].select("block_id").drop_nulls().n_unique()`

### Feed Info

Extract from `feed_info.txt` (single-row table):
- `publisherName`, `publisherUrl`, `feedLanguage`, `feedStartDate`, `feedEndDate`
- If `feed_info.txt` missing or empty → `feedInfo` key omitted

### Agencies

Extract from `agency.txt`: `name`, `url`, `phone`, `email` for each row.

### Service Window

Computed from:
1. `calendar.txt` → min `start_date`, max `end_date`
2. `calendar_dates.txt` → min/max `date`
3. Combined overall min start, max end

### Notice Grouping

Notices grouped by `(code, severity)`. For each group:
- `code`: string
- `severity`: `"ERROR"` | `"WARNING"` | `"INFO"`
- `totalNotices`: total count (including dropped notices beyond capacity)
- `sampleNotices`: up to 1,000 serialized notice field dicts

### system_errors.json

```json
{
  "notices": [
    {
      "code": "runtime_exception_in_validator",
      "severity": "ERROR",
      "totalNotices": 1,
      "sampleNotices": [
        {
          "validatorClassName": "...",
          "exception": "...",
          "message": "..."
        }
      ]
    }
  ]
}
```

### HTML Report

Use Jinja2 templates. Data model mirrors JSON report:
- `metadata` — feed info, counts, agencies, service window, features
- `summary` — notices grouped by severity, then by code
- `config` — validation configuration
- `date` — validation date

Template files stored in `src/gtfs_validator/templates/`.

### JSON Serialization

- Use `json.dumps()` with `indent=2` when `config.pretty_json` is True
- Use compact output (no indent) otherwise
- Custom serializer for `date` → ISO format string

### File Output

- Create `config.output_directory` if it doesn't exist
- Write `report.json`, `report.html`, `system_errors.json` with configured filenames
- Filenames are just basenames — always written into `output_directory`

---

## 12. Notice Schema Export

When `--export_notices_schema` is set:

1. Introspect all registered notice types (from explicit registry, not runtime discovery)
2. For each notice: code, severity, field names and types, description
3. Write `notice_schema.json` to output directory
4. Exit without running validation

```python
def export_notice_schema(output_dir: Path) -> None:
    schema = []
    for notice_cls in NOTICE_REGISTRY:
        schema.append({
            "code": notice_cls.code,
            "severity": notice_cls.severity.name,
            "fields": {
                name: _python_type_to_json_type(type_hint)
                for name, type_hint in get_type_hints(notice_cls).items()
                if name not in ("code", "severity")
            },
        })
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "notice_schema.json").write_text(json.dumps(schema, indent=2))
```

---

## 13. Runner (`runner.py`)

Orchestrates the full pipeline:

```python
def run(config: ValidationConfig) -> RunResult:
    notices = NoticeContainer()
    system_errors: list[SystemError] = []
    start_time = time.monotonic()

    # Stage 1: Open input
    try:
        gtfs_input = open_input(config)
    except Exception as exc:
        return RunResult.EXCEPTION

    # Stage 2: Load tables (parallel)
    feed, table_statuses, load_notices = load_feed(
        gtfs_input, config.num_threads
    )
    notices.merge(load_notices)

    # Stage 3: Load-time validation (schema-driven)
    load_val_notices = run_load_validators(feed, table_statuses)
    notices.merge(load_val_notices)

    # Stage 4: Multi-file validators (parallel)
    for entry in VALIDATOR_REGISTRY:
        if should_skip(entry, table_statuses):
            continue
        result_notices = safe_validate(entry, feed, ctx)
        notices.merge(result_notices)

    # Stage 5: Feature detection
    features = detect_features(feed)

    # Stage 6: Report generation
    elapsed = time.monotonic() - start_time
    generate_reports(config, feed, notices, features, elapsed)

    # Stage 7: Cleanup
    gtfs_input.close()

    if notices.has_system_errors():
        return RunResult.SYSTEM_ERRORS
    return RunResult.SUCCESS
```

### ValidationContext

```python
@dataclass(frozen=True)
class ValidationContext:
    country_code: str
    date_for_validation: date
    # Cached computations added as needed by validators
```

Created once before validator dispatch, passed to all validators.

---

## 14. Error Handling Summary

| Scenario | Behavior | Output |
|---|---|---|
| CSV parse failure | `CsvParsingFailedNotice` (ERROR), table status = `UNPARSABLE_ROWS` | Dependent validators skipped |
| Missing required file | `MissingRequiredFileNotice` (ERROR), empty DataFrame created | Dependent validators may skip |
| Missing recommended file | `MissingRecommendedFileNotice` (WARNING), empty DataFrame created | Validators proceed |
| IO error on input close | `IOError` system error logged | Pipeline continues |
| Validator runtime exception | `RuntimeExceptionInValidatorError` system error | Other validators continue |
| Download failure | Exception propagated | `RunResult.EXCEPTION` (exit code 2) |
| Notice capacity exceeded | Notice silently dropped, count still tracked | Report shows accurate `totalNotices` |

---

## 15. Threading Model

- `concurrent.futures.ThreadPoolExecutor` with `max_workers=config.num_threads`
- Phase 1 (table loading): each table loaded as independent future
- Phase 2 (multi-file validators): each validator executed as independent future
- Each thread owns a private `NoticeContainer`; merged into main container after completion
- Thread pool created once, reused for both phases, shut down after pipeline completes
- Default `num_threads=1` (sequential) — safe default for small feeds

Polars already handles intra-operation parallelism (e.g., CSV parsing, expression evaluation). The thread pool handles inter-table and inter-validator parallelism.

---

## 16. Dependencies

Required Python packages:

| Package | Purpose |
|---|---|
| `polars` | CSV parsing, DataFrames, vectorized validation |
| `click` | CLI argument parsing |
| `jinja2` | HTML report templating |
| `httpx` | HTTP downloads (URL input mode) |

Optional/conditional:
| Package | Purpose |
|---|---|
| `phonenumbers` | Country-specific phone validation (if full parity needed) |
| `langcodes` | BCP 47 language tag validation |

Standard library:
- `zipfile`, `io`, `json`, `pathlib`, `datetime`, `zoneinfo`, `concurrent.futures`, `dataclasses`, `enum`, `re`, `time`

---

## 17. Testing Strategy

### Unit Tests

- **Loading**: Test CSV parsing with minimal CSV strings. Test header validation, type conversion, whitespace handling, empty files, malformed rows.
- **Load validators**: Test PK uniqueness, FK violations, range checks, mixed case, currency amounts with small DataFrames.
- **Validators**: Each validator tested with minimal `dict[str, pl.DataFrame]` feeds. At least one positive case (no notices) and one negative case (notices emitted). Assert exact notice code, severity, and key fields.
- **Features**: Test detection with minimal DataFrames — present and absent cases.
- **Report**: Test JSON structure, notice grouping, capacity truncation.

### Integration Tests

- End-to-end: Run `runner.run()` on small GTFS fixture directories
- Compare output JSON structure against expected snapshots
- Regression fixtures kept small and stable in `tests/fixtures/`

### Equivalence Testing

- Run both Java and Python validators on the same real GTFS feed
- Diff `report.json` output — notice codes, severities, counts, and feature lists must match
- Use `tools/output_comparator.py` for structured comparison

---

## 18. Implementation Order

1. `notices.py` — Notice system (standalone, no deps)
2. `schemas.py` — Schema registry (standalone, pure data)
3. `config.py` — ValidationConfig (standalone)
4. `context.py` — ValidationContext (standalone)
5. `input.py` — GTFS input abstraction (depends on: stdlib only)
6. `loading.py` — CSV loading with Polars (depends on: schemas, notices, input)
7. `load_validators.py` — Schema-driven checks (depends on: schemas, notices)
8. `validators/` — Cross-table validators (depends on: notices, context)
9. `features.py` — Feature detection (depends on: loaded DataFrames)
10. `report.py` — Report generation (depends on: notices, features, config)
11. `runner.py` — Pipeline orchestrator (depends on: everything above)
12. `cli.py` — Entry point (depends on: runner, config)
