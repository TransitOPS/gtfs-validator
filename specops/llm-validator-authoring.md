# LLM Prompt Guide For New Python Validators

This guide is for adding new validation rules, or extending existing rules, in
the current Python/Polars architecture.

It does not assume Java migration work.

## Purpose

Use an LLM to generate validator code safely by enforcing project contracts:
- Pure validator logic
- Explicit registry wiring
- Stable notice contract
- Focused test coverage
- Passing lint/type/test gates

## Required Context To Include In Prompt

- `AGENTS.md`
- `src/gtfs_validator/validators/__init__.py`
- `src/gtfs_validator/notices.py`
- `src/gtfs_validator/context.py`
- Target validator module (or closest related module) under `src/gtfs_validator/validators/`
- Existing related tests under `tests/`
- Any active spec doc for the new rule (if one exists)

If any required context is missing, instruct the LLM to stop and list what is missing.

## Non-Negotiable Rules

- Validator signature must be:
  `(feed: dict[str, pl.DataFrame], ctx: ValidationContext) -> list[Notice]`
- Validators are pure and deterministic.
- No file/network I/O inside validators.
- Keep orchestrator concerns out of validator modules.
- Use Polars expressions/joins for tabular checks when practical.
- Do not introduce new notice code/severity unless explicitly approved.
- Add positive, negative, and edge-case tests.
- Tests must assert exact notice code, severity, and key fields.

## Prompt Template: Create New Validator

```text
You are editing a Python GTFS validator project that uses Polars.

Follow these rules exactly:
1) Read and follow AGENTS.md.
2) Keep changes scoped to one validator concern.
3) Validator signature is fixed:
   (feed: dict[str, pl.DataFrame], ctx: ValidationContext) -> list[Notice]
4) Validator must be pure/deterministic; no file/network I/O.
5) Keep orchestrator/reporting logic out of validator modules.
6) Use Polars expressions/joins for tabular validation where practical.
7) Do not invent new notice code/severity unless explicitly allowed below.
8) Add focused pytest coverage with positive, negative, and edge cases.

Goal:
Create a new validation rule for: <RULE DESCRIPTION>

Notice contract:
<EXISTING NOTICE CODE/SEVERITY/FIELDS OR EXPLICIT APPROVAL FOR NEW ONE>

Required files to read before editing:
- AGENTS.md
- src/gtfs_validator/context.py
- src/gtfs_validator/notices.py
- src/gtfs_validator/validators/__init__.py
- src/gtfs_validator/validators/<TARGET_MODULE>.py
- tests/<RELATED_TEST_FILE>.py

Deliverables:
1) Validator implementation in src/gtfs_validator/validators/<TARGET_MODULE>.py
2) Registry update in src/gtfs_validator/validators/__init__.py
3) Tests in tests/<RELATED_TEST_FILE>.py
4) Short rationale mapping logic to the rule requirements

Before finishing, run:
- uv run pytest -q
- uv run ruff check .
- uv run mypy src (or uv run pyright)

If anything is ambiguous, stop and list assumptions explicitly.
```

## Prompt Template: Add Logic To Existing Validator

```text
You are editing an existing validator in a Python GTFS validator project.

Constraints:
1) Read and follow AGENTS.md.
2) Modify only the target validator concern.
3) Keep validator pure/deterministic with signature:
   (feed: dict[str, pl.DataFrame], ctx: ValidationContext) -> list[Notice]
4) No file/network I/O in validators.
5) Keep output contract stable unless explicitly approved.
6) Prefer Polars expressions/joins for tabular checks.
7) Update tests to cover the new behavior and prevent regressions.

Change request:
<DESCRIBE ADDITION OR BEHAVIOR CHANGE>

Existing contract to preserve:
<NOTICE CODES/SEVERITIES/FIELDS TO KEEP STABLE>

Files to read first:
- AGENTS.md
- src/gtfs_validator/notices.py
- src/gtfs_validator/validators/<TARGET_MODULE>.py
- src/gtfs_validator/validators/__init__.py
- tests/<TARGET_TEST_FILE>.py

Deliverables:
1) Updated validator logic
2) Registry update only if required
3) Updated tests for old + new behavior
4) Brief compatibility note on what stayed the same vs changed

Before finishing, run:
- uv run pytest -q
- uv run ruff check .
- uv run mypy src (or uv run pyright)

If requirements conflict, stop and explain the conflict.
```

## Human Review Checklist

- Change is scoped to one validator concern
- Validator remains pure and deterministic
- Notice contracts are stable or explicitly approved to change
- Registry wiring is correct (`requires` and function entry)
- Tests cover happy path, failure path, and edge cases
- `pytest`, `ruff`, and `mypy/pyright` all pass

## Common Failure Modes

- Prompt lacks concrete notice contract
- Multiple unrelated validators changed in one output
- Tests assert only notice count, not notice identity
- Row-by-row Python loops used where Polars join/filter is expected
- Missing registry update for a new validator
