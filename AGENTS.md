# AGENTS.md

## Project Overview

This repository is migrating GTFS validation from Java to modern Python with Polars.

Primary objective:
- Build a Python validation service that is behaviorally equivalent to the existing validator outputs (`report.json`, `system_errors.json`) while using a cleaner orchestrator architecture.

Core architecture rule:
- The orchestrator owns I/O, parsing, schema enforcement, validator dispatch, feature detection, and reporting.
- Validators own domain logic only and must be pure functions.

Non-goals:
- Do not add new Java code for migration work.
- Do not port Java framework patterns (reflection discovery, DI containers, code generation).

## Required Stack

- Python 3.12+
- Polars for tabular data loading and validation
- `uv` for environment and dependency management
- `pytest` for tests
- `mypy` or `pyright` in strict mode
- `ruff` for linting/formatting

## Repository Conventions

Target Python package layout:
- `src/gtfs_validator/`
- `tests/`
- `tools/`

Key modules to maintain:
- `schemas.py`: schema registry only (data, no behavior)
- `loading.py`: input handling + Polars loading
- `load_validators.py`: schema-driven load-time checks
- `validators/*.py`: cross-table business validations
- `features.py`: feature detection checks
- `report.py`: output serialization and rendering
- `runner.py`: orchestration pipeline
- `cli.py`: user-facing CLI

## Build, Test, and Quality Commands

Run from repository root.

Setup:
- `uv sync`

Run tests:
- `uv run pytest`
- `uv run pytest -q`
- `uv run pytest tests/test_validators -k <validator_name>`

Static checks:
- `uv run ruff check .`
- `uv run ruff format .`
- `uv run mypy src` (or `uv run pyright`)

## Coding Rules (Do)

- Keep validator signature consistent:
  - `(feed: dict[str, pl.DataFrame], ctx: ValidationContext) -> list[Notice]`
- Keep validators deterministic and side-effect free.
- Use explicit schema definitions for every GTFS table and enum.
- Perform load-time checks from schema metadata (required fields, type checks, enum/domain checks, PK/FK checks, range checks).
- Use Polars expressions for tabular checks and joins.
- Use plain Python algorithms for graph/geometry/interval logic when clearer and safer than forcing Polars.
- Emit notices through typed dataclasses with stable code + severity + fields.
- Preserve strict notice capacity limits and truncation behavior.
- Use explicit validator registry ordering; avoid runtime auto-discovery.
- Make error handling explicit: parser/system failures go to system error output, not notice stream.
- Use small, composable functions and module-level pure helpers.
- Add type hints everywhere public.

## Coding Rules (Avoid)

- Do not perform file/network I/O inside validators.
- Do not mutate shared global state during validation.
- Do not rely on implicit DataFrame column types.
- Do not convert large DataFrames to pandas for convenience.
- Do not collect full DataFrames to Python lists unless algorithmically required.
- Do not use row-by-row Python loops for checks that are naturally vectorized.
- Do not swallow parse/type errors.
- Do not invent new notice codes or severities without explicit spec updates.
- Do not couple validators directly to CLI, report rendering, or storage logic.
- Do not depend on reflection/plugin magic for validator registration.

## Polars Guidance

- Prefer lazy execution (`scan_*`, lazy transformations, `collect`) when processing large feeds and multi-step checks.
- Prefer eager DataFrames when data is already loaded in memory and checks are simple.
- Use explicit dtypes in schema-driven reading/casting.
- Normalize null/empty semantics once during load-time, then keep consistent.
- Use joins for FK and cross-table checks; avoid nested loops.
- Keep expression pipelines readable; split long expressions into named helpers.

## Testing Instructions

For every new validator or load-time rule:
- Add focused unit tests with minimal DataFrames.
- Add at least one positive and one negative case.
- Add edge cases: missing tables, missing columns, nulls, empty files, malformed values.
- Assert exact notice code, severity, and key fields.

For pipeline-level behavior:
- Add integration tests that run `runner` on representative GTFS fixtures.
- Compare generated report JSON against expected structures.
- Keep regression fixtures small and stable.

Equivalence expectation:
- Validate behavior against existing contracts/specs before changing logic.

## Security and Safety

- Treat all GTFS input as untrusted.
- Validate and constrain file paths for zip extraction and directory input.
- Prevent zip-slip/path traversal on archive extraction.
- Bound memory usage by selecting only required columns and using streaming/lazy operations when possible.
- Enforce safe limits for notice accumulation and exported samples.
- Do not execute external content from feeds.
- Avoid logging sensitive local filesystem details in user-facing errors.

## Performance Priorities

- Optimize load-time validation first (it runs for every feed).
- Reuse precomputed lookup structures in `ValidationContext` for repeated checks.
- Avoid repeated joins/scans for the same key relationships.
- Profile before micro-optimizing.

## Change Management for Agents

Before coding:
- Read `specops/initial-plan.md` and active spec docs.
- Confirm whether change belongs to load-time validation, cross-table validator logic, feature detection, or reporting.

When coding:
- Keep changes scoped to one concern.
- Update tests in the same change.
- Preserve backward-compatible output shape unless a spec explicitly changes it.

Before finishing:
- Run tests and static checks.
- Summarize behavioral impact and any unresolved risks.

## LLM-Assisted Validator Authoring

Post-migration, use LLM generation to accelerate net-new Python validator additions
and targeted updates to existing Python validators.
Treat generated code as untrusted until tests and checks pass.

Required workflow:
- Start from the template in `specops/llm-validator-authoring.md`.
- Provide the model with exact Python source context (related validators, tests, notice contracts).
- Require strict output scope: one validator concern, one registry update, matching tests.
- Require no new notice codes/severities unless the spec explicitly authorizes it.
- Require deterministic, pure validator functions with no I/O.
- Require explicit assumptions when rule intent is ambiguous.

Acceptance gate for generated changes:
- `uv run pytest -q`
- `uv run ruff check .`
- `uv run mypy src` (or `uv run pyright`)
- Manual reviewer confirms behavior matches existing contracts and output shape.

## Definition of Done

A change is complete only when:
- Code follows orchestrator/validator separation.
- Tests cover normal + edge behavior.
- Lint and type checks pass.
- Output contract remains stable or is intentionally updated with spec changes.
