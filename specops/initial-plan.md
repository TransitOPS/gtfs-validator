# SpecOps: Java → Python/Polars Migration Plan

## Context

**GTFS Validator** validates General Transit Feed Specification data feeds for correctness and compliance, and discovers which GTFS features are present in a feed (~35 feature checks across 7 feature groups: accessibility, base add-ons, fares, pathways, flexible services, etc.).

**File counts** (613 Java files, excluding deprecated and build artifacts):

- `main/` — 292 files: 82 validators, 61 table schemas/enums, 10 report summary, 1 notice, 14 utilities, 3 runner
- `core/` — 178 files: 60 notices, 16 validator infrastructure, 12 table infrastructure, 6 input/loading, 5 parsing, 5 notice schema, 5 report summary base, 4 performance, 3 types, 2 testing, 2 model, 1 I/O + 51 test files
- `processor/` — 56 files: 21 code generators + 13 test files — **eliminated in target** (no codegen needed)
- `model/` — 31 files: 27 annotations, 2 table base, 2 notice base — **collapses to schema dicts** in target
- `output-comparator/` — 27 files: regression comparison tool (8 I/O, 5 report model, 3 CLI)
- `cli/` — 3 files: command-line entry point
- `app/` — 9 files: GUI desktop app + packager
- `web/` — 17 files: web service, client, pipeline

**Current stack**: Java 17, Gradle, AutoValue, JCommander, GSON, Univocity (CSV), JavaPoet (codegen), ClassGraph (runtime discovery), Guava.

**Target stack**: Python 3.12+, Polars (CSV parsing, DataFrames, parallel execution, type enforcement), click or typer (CLI), dataclasses or pydantic (notices/config), pytest, mypy or pyright, uv (package management).

**Test coverage**: 199 test files across 11 test directories. Primary coverage in `main/` (114 tests) and `core/` (51 tests). JUnit test runner → pytest in target.

**Existing documentation**: `docs/ARCHITECTURE.md`, `RULES.md`, `BUILD.md`, `CONTRIBUTING.md`, `NEW_RULES.md`, `NOTICE_MIGRATION.md`, `FEATURES.md`, `USAGE.md`, `ACCEPTANCE_TESTS.md` — comprehensive and current.

**Migration strategy**: **Parallel rewrite with behavioral equivalence testing**. No runtime interop between Java and Python, so Strangler Fig doesn't apply literally. Instead: build Python implementation alongside Java, validate output equivalence on real GTFS feeds, switch over when report.json output matches.

**Core design principle**: **Orchestrator owns data; validators own logic.** The orchestrator handles all I/O (ZIP/URL/directory), CSV parsing, schema enforcement, caching, and data delivery. Validators are pure functions that receive data and return notices — they never read files, manage state, or know about Polars internals. This clean separation is the target architecture; the analysis specs exist to extract the behavioral contracts that validators must satisfy.

**Polars boundary rule**: Use Polars for data loading and tabular validations (null checks, type checks, foreign keys, uniqueness, range checks, enum validation, feature detection). Use plain Python for validators that require graph traversal (PathwayLoopValidator, PathwayReachableLocationValidator), geometry (ShapeToStopMatchingValidator), overlapping interval detection (BlockTripsWithOverlappingStopTimesValidator), color math (RouteColorContrastValidator), or complex date logic (ServiceGapValidator). These validators pull data from DataFrames into Python structures, compute, and emit notices. Do not force imperative algorithms into Polars expressions.

---

## Phase 1: Foundation Setup

### 1. Orchestrator Design Doc

Before analyzing the Java codebase, define the Python orchestrator contract. This is designed fresh — not reverse-engineered from Java — because the Java infrastructure (code generation, reflection-based DI, custom containers) is being replaced entirely.

The orchestrator design doc defines:

- **Feed type**: `dict[str, pl.DataFrame]` — the orchestrator reads GTFS input, parses all CSV files using Polars with schema-driven dtype casting, and provides this dict to validators.
- **ValidationContext**: dataclass carrying country code, reference date, and any cached computations (service intervals, calendar date sets) that multiple validators need.
- **Validator contract**: every validator is a callable with signature `(feed: dict[str, pl.DataFrame], ctx: ValidationContext) -> list[Notice]`. Validators receive data, return notices. No side effects, no I/O, no framework coupling.
- **Notice type**: `@dataclass` with severity (ERROR/WARNING/INFO), code (string identifier), and fields (dict of context). NoticeContainer handles capacity limits (10M total, 100K per type/severity, 1K export cap per type/severity).
- **Validator registry**: explicit list of validator functions/classes. No reflection, no discovery. The registry also declares each validator's table dependencies so the orchestrator can skip validators when required tables are missing or failed to parse.
- **Orchestration pipeline**: input → load (schema-enforced) → load-time validation (required fields, types, enums, primary keys, foreign keys — all driven by schema, not by individual validators) → cross-table validators → feature detection → report generation.
- **Load-time validation**: the orchestrator itself handles all schema-driven checks during loading (required/recommended fields, type validation, enum ranges, primary key uniqueness, foreign key existence). These are the checks Java generates from annotations (`@Required`, `@ForeignKey`, `@PrimaryKey`, `@NonNegative`, `@EndRange`, etc.). They are not separate validators in the Python target — they are orchestrator responsibilities driven by the schema registry.

### 2. AGENTS.md (Legacy Analysis Agent)

Instruction set for analyzing the existing Java codebase. Focused on extracting **validator behavioral contracts**, not infrastructure details. Must cover:

- **What each validator checks**: input tables, conditions checked, notices emitted, edge cases
- **What the orchestrator must provide**: which schema-driven checks the Java codegen produces (required fields, foreign keys, enum ranges, primary key uniqueness, end-range validation, non-negative/positive/non-zero checks, mixed-case checks, currency validation, lat/lon validation) — these become orchestrator load-time logic in Python
- **Notice system contracts**: severity levels, capacity limits, JSON output format, notice schema export
- **Feature detection rules**: the ~35 feature presence checks and their cross-table logic
- **Report format**: exact JSON structure of `report.json`, `system_errors.json`, and HTML report

### 3. Spec Directory Structure

```
specops/specs/
├── orchestrator.md              # Designed fresh for Python (Phase 1)
├── analysis/
│   ├── 01-schemas-and-load-rules/
│   ├── 02-notice-system/
│   ├── 03-validators-stops-trips/
│   ├── 04-validators-fares-transfers/
│   ├── 05-validators-calendar-service/
│   ├── 06-validators-shapes-pathways/
│   ├── 07-validators-feed-agency-routes/
│   ├── 08-utilities/
│   ├── 09-feature-detection/
│   ├── 10-report-format/
│   ├── 11-cli-interface/
│   └── 12-output-comparator/
└── implementation/
    ├── 01-schemas-and-load-rules/
    ├── ...  (mirrors analysis/)
    └── 12-output-comparator/
```

Note: No analysis specs for Java's parsing infrastructure, table containers, validator infrastructure, code generator, or runner pipeline. Those are Java-specific machinery being replaced by the orchestrator. The analysis specs focus exclusively on **behavioral contracts**: what data goes in, what notices/reports come out.

---

## Phase 2: Discovery & Specification Generation

### What each analysis spec captures

For **validator specs** (03-07):
- Each validator's name and one-line purpose
- Input: which tables and fields it reads
- Logic: the check it performs (expressed as rules, not Java code)
- Output: notice code(s), severity, fields captured in each notice
- Edge cases: what happens with missing tables, null fields, empty datasets
- Test contracts: representative input/output pairs from Java tests

For **non-validator specs** (01-02, 08-12):
- Purpose and responsibilities
- Data shapes (schema definitions, notice structure, report JSON format)
- Behavioral contracts (what the Java system produces, which the Python system must match)

### Spec Breakdown

**Schemas & Load-Time Rules (1 spec)**

1. `schemas-and-load-rules.md` — All ~30 table schemas (field names, types, constraints) and ~31 enum types. For each schema: the complete field list with types, required/recommended/conditionally-required flags, primary keys (simple and composite), foreign keys (target table + field), default values, indexes, and annotation-driven validation rules (@NonNegative, @Positive, @NonZero, @EndRange, @MixedCase, @CurrencyAmount, @FieldType for lat/lon/color/URL/email/phone). These annotation-driven rules become orchestrator load-time logic — they are not separate validators.

**Notice System (1 spec)**

2. `notice-system.md` — Complete catalog of all notice codes, their severity, and their fields. NoticeContainer capacity limits. JSON serialization format. Notice schema export format. System error handling. This spec defines the contract for report.json output.

**Validators: Stops, Trips, StopTimes (1 spec)**

3. `validators-stops-trips.md` — ~25 validators. For each: input tables, check logic, notice codes, edge cases. Key validators: ParentStationValidator, StopNameValidator, StopRequiredLocationValidator, MissingTripEdgeValidator, TripUsabilityValidator, StopTimeArrivalAndDepartureTimeValidator, StopTimeTravelSpeedValidator, StopTimeIncreasingDistanceValidator, BlockTripsWithOverlappingStopTimesValidator, LocationTypeSingleEntityValidator, LocationHasStopTimesValidator, StopTimesShapeDistTraveledPresenceValidator, StopTimesTripBlockOrderValidator, TimepointTimeValidator, PickupDropOffTypeValidator, PickupDropOffWindowValidator, ContinuousPickupDropOffValidator, etc.

**Validators: Fares & Transfers (1 spec)**

4. `validators-fares-transfers.md` — ~15 validators. FareTransferRuleTransferCountValidator, FareLegJoinRuleValidator, FareMediaNameValidator, TransfersStopTypeValidator, TransfersTripReferenceValidator, TransfersInSeatTransferTypeValidator, TransferDistanceValidator, DuplicateFareMediaValidator, FareProductDefaultRiderCategoriesValidator, FareTransferRuleDurationLimitTypeValidator, FareAttributeAgencyIdValidator, OverlappingPickupDropOffZoneValidator, StopTimesGeographyIdPresenceValidator, etc.

**Validators: Calendar & Service (1 spec)**

5. `validators-calendar-service.md` — ~8 validators. ExpiredCalendarValidator, MissingCalendarAndCalendarDateValidator, ServiceHasNoActiveDayOfTheWeekValidator, FeedExpirationDateValidator, FeedServiceDateValidator, FeedValidTodayValidator, ServiceGapValidator, DateTripsValidator.

**Validators: Shapes & Pathways (1 spec)**

6. `validators-shapes-pathways.md` — ~10 validators. ShapeToStopMatchingValidator (geometry — plain Python), ShapeIncreasingDistanceValidator, ShapeUsageValidator, SingleShapePointValidator, PathwayLoopValidator (graph — plain Python), PathwayReachableLocationValidator (graph — plain Python), PathwayEndpointTypeValidator, PathwayDanglingGenericNodeValidator, PathwayStopAccessValidator, BidirectionalExitGateValidator, TripAndShapeDistanceValidator.

**Validators: Feed, Agency, Routes (1 spec)**

7. `validators-feed-agency-routes.md` — ~12 validators. AgencyConsistencyValidator, RouteColorContrastValidator (color math — plain Python), RouteNameValidator, DuplicateRouteNameValidator, RouteAgencyIdValidator, FeedContactValidator, MatchingFeedAndAgencyLangValidator, MissingFeedInfoValidator, UrlConsistencyValidator, AttributionWithoutRoleValidator, NetworkIdConsistencyValidator, InconsistentRouteTypeForBlockIdValidator, etc.

**Utilities (1 spec)**

8. `utilities.md` — Pure functions and caches used by validators: CalendarUtil, TripCalendarUtil, ServiceIntervalCache, ServiceIdIntersectionCache, StopUtil, SetUtil; shape matching (StopToShapeMatcher, ShapePoints, StopPoints, Assignment, Problem); GeoJSON utilities; VersionInfo/VersionResolver. Document each utility's inputs, outputs, and which validators use it.

**Feature Detection (1 spec)**

9. `feature-detection.md` — The ~35 feature presence checks from `docs/FEATURES.md`. For each feature: which tables and fields must be present (with the exact cross-table logic). These are Polars column-existence and non-null-any checks, run by the orchestrator after loading.

**Report Format (1 spec)**

10. `report-format.md` — Exact JSON structure of `report.json` (notices grouped by code, with counts and samples), `system_errors.json`, and HTML report layout. FeedMetadata fields: row counts, agency info, service windows, feature list, validation time, memory usage. This is the ultimate equivalence target — if the Python report matches the Java report, migration succeeds.

**CLI Interface (1 spec)**

11. `cli-interface.md` — All CLI parameters from `docs/USAGE.md`: `--input`, `--url`, `--output`, `--country_code`, `--date`, `--threads`, `--storage_directory`, `--pretty`, `--export_notices_schema`, report name overrides. Exit codes. This defines the user-facing contract.

**Output Comparator (1 spec)**

12. `output-comparator.md` — Regression comparison tool: compares two validation reports, detects new/dropped notices, generates acceptance test reports.

**Total: 12 analysis specs + 1 orchestrator design doc.** The 5 validator specs are the bulk of the work. The other 7 are smaller supporting specs. No specs for Java infrastructure that we're replacing.

---

## Phase 3: Verification

For each analysis spec:

- Cross-reference every listed validator/rule against actual Java source files
- Verify notice codes and severity levels against Java notice class definitions
- Verify behavioral contracts against corresponding test files in `main/src/test/` and `core/src/test/`
- Flag undocumented edge cases: what happens when a table is empty? When a foreign key target table failed to parse? When fields have unexpected whitespace?
- Mark spec as **verified** only after source + test cross-check passes

---

## Phase 4: Implementation Specification

### 1. AGENTS.md v2 (Build Agent)

Python/Polars-specific instruction set. Must cover:

- **Orchestrator pattern**: single pipeline module that owns all I/O, parsing, schema enforcement, and validator dispatch. Validators are pure functions — no framework, no base classes, no DI.
- **Polars idioms**: lazy vs eager evaluation; expression-based validation; join strategies for cross-table checks; `read_csv` with schema dicts
- **Validator signature**: `(feed: dict[str, pl.DataFrame], ctx: ValidationContext) -> list[Notice]`. Validators may use Polars expressions or plain Python internally — their choice — but they never do I/O.
- **Load-time validation**: orchestrator uses schema registry to generate Polars expressions for required-field checks, type validation, enum ranges, primary key uniqueness, foreign key existence, non-negative/positive/non-zero, end-range, mixed-case, lat/lon bounds, color format, URL format. One module, schema-driven, not 30 separate validator files.
- **Notice dataclasses**: `@dataclass` with severity, code, fields. NoticeContainer preserves Java's capacity limits.
- **Testing**: pytest, parametrize for validator tests, fixtures for sample DataFrames. Each validator test: build minimal DataFrames, call validator, assert notices.
- **Packaging**: `pyproject.toml`, uv, single `gtfs_validator` package
- **Type checking**: mypy or pyright in strict mode

### 2. Analysis → Implementation Translation

For each verified analysis spec, produce an implementation spec that:

- Specifies Python module, function signatures, and types
- For each validator: whether it uses Polars expressions or plain Python
- For schema-driven load-time rules: the Polars expression that implements each check
- Preserves all behavioral contracts from analysis specs

---

## Phase 5: Implementation

### 1. Initialize Python Project

```
gtfs_validator_py/
├── pyproject.toml
├── src/
│   └── gtfs_validator/
│       ├── __init__.py
│       ├── schemas.py            # Schema registry: dict of table definitions
│       ├── notices.py            # Notice dataclass + NoticeContainer
│       ├── context.py            # ValidationContext dataclass
│       ├── loading.py            # Input handling (ZIP/URL/dir) + Polars CSV loading
│       ├── load_validators.py    # Schema-driven load-time checks (orchestrator logic)
│       ├── validators/           # All validation rules (pure functions)
│       │   ├── __init__.py       # Validator registry (explicit list)
│       │   ├── stops_trips.py
│       │   ├── fares_transfers.py
│       │   ├── calendar_service.py
│       │   ├── shapes_pathways.py
│       │   └── feed_agency_routes.py
│       ├── features.py           # Feature detection (~35 checks)
│       ├── report.py             # JSON + HTML report generation
│       ├── runner.py             # Orchestrator: load → validate → report
│       └── cli.py                # CLI entry point
├── tests/
│   ├── conftest.py               # Shared fixtures (sample DataFrames)
│   ├── test_loading.py
│   ├── test_load_validators.py
│   ├── test_validators/
│   │   ├── test_stops_trips.py
│   │   ├── test_fares_transfers.py
│   │   ├── test_calendar_service.py
│   │   ├── test_shapes_pathways.py
│   │   └── test_feed_agency_routes.py
│   ├── test_features.py
│   └── test_report.py
└── tools/
    └── output_comparator.py      # Report diff tool
```

Key difference from previous plan: `load_validators.py` is orchestrator logic (schema-driven checks run during loading), not a validator module. The `validators/` directory contains only cross-table business logic validators.

### 2. Implementation Order

1. **`schemas.py`** — Schema registry. Pure data. No logic. Defines all ~30 tables.
2. **`notices.py`** — Notice dataclass + NoticeContainer with capacity limits. Standalone.
3. **`context.py`** — ValidationContext. Standalone.
4. **`loading.py`** — Read GTFS input into `dict[str, pl.DataFrame]`. Depends on schemas.
5. **`load_validators.py`** — Schema-driven load-time checks (required fields, types, enums, PKs, FKs, ranges, etc.). Depends on schemas + notices. This one module replaces Java's entire `processor/` + all generated validators.
6. **`validators/`** — Migrate in groups: calendar_service → feed_agency_routes → stops_trips → fares_transfers → shapes_pathways. Each file is a set of pure functions. Depends on notices + context.
7. **`features.py`** — Feature detection. Depends on loaded DataFrames.
8. **`report.py`** — JSON + HTML report generation. Depends on notices + features.
9. **`runner.py`** — Orchestrator pipeline. Depends on everything above.
10. **`cli.py`** — Entry point. Depends on runner.
11. **`output_comparator.py`** — Independent tool.

### 3. Per-Module Implementation Cycle

For each module:

1. Build from implementation spec
2. Write pytest tests (translate behavioral contracts from Java tests)
3. Type-check with mypy/pyright
4. Once runner is complete: run equivalence check against Java on same GTFS feed

---

## Phase 6: Retirement

**Criteria**: Python validator produces identical `report.json` output as Java validator on a reference set of GTFS feeds (including edge cases: empty files, missing required files, corrupt data, very large feeds).

**Steps**:

1. Remove Java source directories (`main/`, `core/`, `model/`, `processor/`, `cli/`, `app/`)
2. Remove Gradle build files (`build.gradle`, `settings.gradle`, `gradlew*`, `gradle/`)
3. Move Python project from `gtfs_validator_py/` to project root (or keep as-is)
4. Update CI/CD pipelines (GitHub Actions) for Python: pytest, mypy, uv
5. Update `docs/` to reflect Python architecture
6. Archive analysis specs in `specops/specs/analysis/` as permanent documentation

---

## What We Do First

1. **Write `specops/specs/orchestrator.md`** — Design the orchestrator contract fresh: feed type, validator signature, notice system, load-time validation, registry, pipeline
2. **Write AGENTS.md** — Legacy analysis agent focused on extracting validator behavioral contracts
3. **Create `specops/specs/analysis/` directory tree** — Set up the 12 spec directories
4. **Write `schemas-and-load-rules.md`** — Document all 30 table schemas and the annotation-driven rules the orchestrator must replicate
5. **Write `notice-system.md`** — Notice catalog, capacity limits, JSON format — this defines the equivalence target

---

## Key Files to Reference

| File | Role |
|------|------|
| `main/src/main/java/.../runner/ValidationRunner.java` | Pipeline orchestrator — reference for pipeline stages |
| `core/src/main/java/.../table/GtfsFeedLoader.java` | Loading pipeline — reference for load-time checks |
| `core/src/main/java/.../notice/NoticeContainer.java` | Capacity limits — must preserve exact limits |
| `core/src/main/java/.../notice/ValidationNotice.java` | Notice structure — defines report.json format |
| `core/src/main/java/.../parsing/RowParser.java` | Type conversion edge cases — must replicate |
| `core/src/main/java/.../input/GtfsInput.java` | Input handling — ZIP/directory/URL patterns |
| `processor/src/main/java/.../processor/Analyser.java` | Documents what behavior annotations produce |
| `main/src/main/java/.../table/GtfsRouteSchema.java` | Representative schema — shows annotation usage |
| `main/src/main/java/.../table/GtfsStopTimeSchema.java` | Complex schema (high-volume table) |
| `main/src/main/java/.../validator/FeedExpirationDateValidator.java` | Representative simple validator |
| `main/src/main/java/.../validator/ShapeToStopMatchingValidator.java` | Representative complex validator (geometry) |
| `main/src/main/java/.../validator/PathwayLoopValidator.java` | Representative graph validator |
| `main/src/main/java/.../validator/BlockTripsWithOverlappingStopTimesValidator.java` | Representative interval logic validator |
| `main/src/main/java/.../reportsummary/model/FeedMetadata.java` | Report data model + feature detection |
| `cli/src/main/java/.../cli/Main.java` | CLI parameters |
| `docs/ARCHITECTURE.md` | Canonical architecture documentation |
| `docs/FEATURES.md` | Feature detection rules (~35 features, cross-table) |
| `docs/USAGE.md` | CLI parameters and usage patterns |
| `RULES.md` | Validation rule catalog |

---

## Verification

- **Analysis specs vs source**: For each validator listed in a spec, verify notice codes and logic against actual Java source
- **Implementation specs typecheck**: `mypy --strict` passes after each module
- **Test equivalence**: Each migrated validator passes pytest tests ported from Java tests
- **Feature detection parity**: Compare `features` section in report output between Java and Python for the same feed
- **End-to-end smoke test**: Run both Java and Python validators against the same GTFS feed, diff `report.json` output — notices and feature list should be identical. Leverage existing GitHub Actions E2E workflow (`.github/workflows/end_to_end.yml`) as a model
- **Performance baseline**: Validate that Python/Polars processes a large feed (e.g., a major metro transit agency) within reasonable time
