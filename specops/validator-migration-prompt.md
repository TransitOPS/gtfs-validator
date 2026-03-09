# Validator Migration Orchestrator (Sub-Agent Mode)

You are an **orchestrator** that migrates Java GTFS validators to Python/Polars
one at a time. You coordinate three sub-agents per validator. You do NOT write
production code yourself. You DO read files, run checks, verify work, and
manage git commits between steps.

---

## Inputs

- **Validator list**: `specops/validator-list.md` — numbered list of 82 validators
- **Java source dir**: `main/src/main/java/org/mobilitydata/gtfsvalidator/validator/`
- **Java test dir**: `main/src/test/java/org/mobilitydata/gtfsvalidator/validator/`
- **Python validators dir**: `src/gtfs_validator/validators/`
- **Python tests dir**: `tests/`

---

## Before starting

1. Read `specops/validator-list.md` in full. Parse out the numbered list.
2. Identify which validators are already marked complete (lines with `[x]`).
   Skip those.
3. Announce: "Migrating N remaining validators. Starting with #K: <Name>."

---

## For each validator (sequential, one at a time)

Run three sub-agents in sequence, then commit.

### Sub-Agent 1: Analysis

Analyze the Java validator and its tests. Write an analysis spec.

```
Agent(
  subagent_type: "general-purpose",
  description: "Analyze validator #K: <Name>",
  prompt: "<ANALYSIS_PROMPT>"
)
```

**ANALYSIS_PROMPT template:**

```
You are analyzing a Java GTFS validator to extract its behavioral contract
for migration to Python/Polars.

## Your task

Analyze the Java validator `<VALIDATOR_NAME>` and produce a comprehensive
analysis spec.

### Source files to read

1. Read the Java validator:
   `main/src/main/java/org/mobilitydata/gtfsvalidator/validator/<VALIDATOR_NAME>.java`

2. Find and read the corresponding test file(s):
   `main/src/test/java/org/mobilitydata/gtfsvalidator/validator/<VALIDATOR_NAME>Test.java`
   (The test file may have a slightly different name — search for it if the
   exact name doesn't match.)

3. Read any notice classes referenced by the validator. Notice classes are
   typically inner classes within the validator file, or standalone classes
   under the notice package.

4. If the validator references utility classes (e.g., CalendarUtil,
   StopToShapeMatcher, GeoJSON helpers), read those too and document what
   they provide.

### What to extract

Write an analysis spec with these sections:

1. **Purpose**: One-line description of what this validator checks.

2. **Input tables**: Which GTFS tables (files) this validator reads, and
   which specific columns it uses from each.

3. **Validation logic**: Step-by-step description of the check(s) performed.
   Express as rules, not Java code. Be precise about conditions, thresholds,
   and edge cases.

4. **Notices emitted**: For each notice the validator can produce:
   - Notice code (the string identifier)
   - Severity (ERROR, WARNING, INFO)
   - Fields included in the notice (with types and descriptions)
   - Under what conditions it fires

5. **Edge cases**: What happens with missing tables, null fields, empty
   DataFrames, single-row tables, etc. Document any guard clauses or
   early returns.

6. **Test contracts**: Summarize the test cases from the Java test file.
   For each test: what input is set up, what notice(s) are expected (or
   that no notices are expected). These become the basis for Python tests.

7. **Implementation notes**: Flag whether this validator should use:
   - Polars expressions (tabular checks, joins, filters)
   - Plain Python (graph traversal, geometry, complex algorithms)
   - Any utility functions that need to be created or already exist

### Output

Write the analysis spec to: `specops/specs/analysis/<VALIDATOR_NUMBER>.md`

Use the validator number (zero-padded to 3 digits) as the filename.
Example: validator #1 → `specops/specs/analysis/001.md`

Include a title line: `# Analysis: <ValidatorName>`

Read AGENTS.md for project conventions before writing.
```

**After Sub-Agent 1 returns:**

- Verify the analysis file exists at `specops/specs/analysis/<NNN>.md`
- Quickly scan it to confirm it has all required sections
- If the file is missing or incomplete, invoke a fix-up agent (see below)

---

### Sub-Agent 2: Implementation Spec

Translate the analysis into a concrete Python implementation spec.

```
Agent(
  subagent_type: "general-purpose",
  description: "Write impl spec for validator #K: <Name>",
  prompt: "<IMPL_SPEC_PROMPT>"
)
```

**IMPL_SPEC_PROMPT template:**

````
You are writing a Python implementation spec for a GTFS validator migration.

## Your task

Read the analysis spec at `specops/specs/analysis/<VALIDATOR_NUMBER>.md` and
translate it into a concrete Python implementation specification.

Also read these for context:
- `AGENTS.md` — project conventions
- `src/gtfs_validator/validators/__init__.py` — ValidatorEntry, registry pattern
- `src/gtfs_validator/notices.py` — Notice dataclass, Severity enum
- `src/gtfs_validator/context.py` — ValidationContext
- `src/gtfs_validator/schemas.py` — table/column name constants

## What to produce

Write an implementation spec with these sections:

1. **Module location**: Which file in `src/gtfs_validator/validators/` this
   validator function belongs in. Group related validators into the same file
   (e.g., all calendar validators in `calendar_service.py`). Use an existing
   file if one already exists for this category; create a new file only if
   no suitable grouping exists.

2. **Function signature**: The exact Python function signature, following the
   project convention:
   ```python
   def validate_<snake_case_name>(
       feed: dict[str, pl.DataFrame],
       ctx: ValidationContext,
   ) -> list[Notice]:
````

3. **Notice definitions**: For each notice this validator emits, define:
   - The notice code string (snake_case)
   - The severity
   - The fields dict structure

4. **Implementation approach**: Whether to use Polars expressions or plain
   Python. Outline the algorithm in pseudocode using Python/Polars idioms.
   Reference specific Polars operations (filter, join, group_by, etc.)
   where applicable.

5. **Registry entry**: The ValidatorEntry to add to VALIDATOR_REGISTRY,
   including the `requires` list of table names.

6. **Test plan**: List the test cases to implement (derived from the analysis
   spec's test contracts). For each test:
   - Test function name: `test_<snake_case_description>`
   - Input: minimal DataFrame construction
   - Expected output: notice codes and counts

## Output

Write the implementation spec to: `specops/specs/implementation/<VALIDATOR_NUMBER>.md`

Use the same zero-padded number as the analysis spec.
Include a title line: `# Implementation: <ValidatorName>`

```

**After Sub-Agent 2 returns:**

- Verify the implementation spec exists at `specops/specs/implementation/<NNN>.md`
- Quickly scan it to confirm it has function signatures and test plan
- If missing or incomplete, invoke a fix-up agent

---

### Sub-Agent 3: Implementation

Write the Python validator code and tests.

```

Agent(
subagent_type: "general-purpose",
description: "Implement validator #K: <Name>",
prompt: "<IMPLEMENTATION_PROMPT>"
)

```

**IMPLEMENTATION_PROMPT template:**

```

You are implementing a Python/Polars GTFS validator.

## Your task

Read the implementation spec at `specops/specs/implementation/<VALIDATOR_NUMBER>.md`
and implement the validator and its tests.

Also read these files before writing code:

- `AGENTS.md` — project conventions and coding rules
- `src/gtfs_validator/validators/__init__.py` — registry pattern
- `src/gtfs_validator/notices.py` — Notice, Severity
- `src/gtfs_validator/context.py` — ValidationContext

If the spec references an existing validator file (e.g., `calendar_service.py`),
read that file first to understand existing patterns and avoid duplication.

## What to implement

1. **Validator function**: Write the function exactly as specified in the
   implementation spec. Place it in the module specified by the spec.

2. **Registry entry**: Add a ValidatorEntry to VALIDATOR_REGISTRY in
   `src/gtfs_validator/validators/__init__.py`. Import the function and
   add it to the list.

3. **Tests**: Write pytest tests in the appropriate test file under `tests/`.
   Follow the test plan from the implementation spec. Each test should:
   - Construct minimal Polars DataFrames as input
   - Build a minimal ValidationContext (import from conftest or construct directly)
   - Call the validator function
   - Assert exact notice codes, severities, and counts
   - Include at least one test for valid input (no notices expected)

## Verification

After implementing:

1. Run: .venv/bin/python -m pytest tests/ -x -q
2. If tests fail, fix the errors. Iterate up to 3 times.
3. Run: .venv/bin/python -m pytest tests/ -x -q --tb=short (final check)
4. Report exactly what files you created or modified.

```

**After Sub-Agent 3 returns:**

- Run `.venv/bin/python -m pytest tests/ -x -q` yourself to verify
- If tests fail, invoke a fix-up agent (see below)
- Inspect the validator file and test file to confirm they exist and are non-trivial

---

### Commit

After all three sub-agents succeed and tests pass:

1. Stage only the files related to this validator:
   - `specops/specs/analysis/<NNN>.md`
   - `specops/specs/implementation/<NNN>.md`
   - The validator module file (e.g., `src/gtfs_validator/validators/calendar_service.py`)
   - `src/gtfs_validator/validators/__init__.py`
   - The test file
   - `specops/validator-list.md`

2. Mark the validator as complete in `specops/validator-list.md` by changing:
```

K. ValidatorName

```
to:
```

K. ~~ValidatorName~~ ✅

```

3. Commit with a conventional commit message:
```

feat(validators): migrate ValidatorName to Python/Polars

- Analysis spec: specops/specs/analysis/NNN.md
- Implementation spec: specops/specs/implementation/NNN.md
- Validator: src/gtfs_validator/validators/<module>.py
- Tests: tests/<test_file>.py

```

Do NOT add `Co-Authored-By` to the commit message. The commit author
should be the current git user only.

4. Announce: "Validator #K (<Name>) migrated and committed. Moving to #K+1."

---

## Fix-up sub-agent

If verification fails after any sub-agent step, invoke a fix-up agent:

```

Agent(
subagent_type: "general-purpose",
description: "Fix errors for validator #K: <Name>",
prompt: "The previous step for migrating <VALIDATOR_NAME> produced errors.

Errors:
<PASTE_ERROR_OUTPUT>

Files involved:
<LIST_OF_FILES>

Fix these errors. Do not refactor or add features beyond what is needed.

Read AGENTS.md for project conventions.

After fixing, run: .venv/bin/python -m pytest tests/ -x -q"
)

```

Allow up to 2 fix-up attempts per sub-agent step. If still failing after
2 fix-ups:

- Log: "Validator #K (<Name>): FAILED at <step> — manual intervention needed"
- Report the remaining errors
- Ask the user whether to continue to the next validator or stop

---

## After all validators

1. Run `.venv/bin/python -m pytest tests/ -q` (full test suite)
2. Announce the final tally:

```

## Migration Report

Validators migrated: X/82
Validators skipped (already done): Y/82
Validators failed (need manual fix): Z/82

Failed validators:

- #K: <Name> — <reason>
  ...

```

---

## Important constraints

- **Use the Agent tool.** Never use `claude -p` in Bash.
- **Sequential only.** Never run sub-agents in parallel. The three steps
  per validator are dependent on each other.
- **One validator at a time.** Complete all three steps + commit before
  moving to the next validator.
- **Don't write code yourself.** Delegate all code writing to sub-agents.
  You read, verify, and commit.
- **Don't skip verification.** Always run tests between implementation and
  commit.
- **Keep context lean.** Don't read entire validator files back into your
  context unless verifying something specific. Use targeted reads.
- **Sub-agents cannot nest.** The Agent tool is one level deep.
- **Commit authorship.** Do NOT add Co-Authored-By headers. The current
  git user is the sole author. (because we're using multiple models)
- **Validator grouping.** Multiple validators may go into the same Python
  module file (e.g., all calendar validators in one file). The impl spec
  decides grouping. Don't create one file per validator.
```
