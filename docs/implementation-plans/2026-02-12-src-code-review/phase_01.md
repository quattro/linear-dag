# src Review Implementation Plan

**Goal:** Execute a source-level code review for the CLI layer and produce a severity-ranked findings report.

**Architecture:** Review `src/linear_dag/cli.py` as the orchestration boundary, then trace into directly-invoked helpers to validate input contracts, failure behavior, and compatibility assumptions. Output is evidence-based findings with file/line references.

**Tech Stack:** Python 3, Polars, argparse, HDF5-backed LinearARG workflows

**Scope:** 4 phases from user-requested review (`cli`, `core`, `association`, `structure`), this file covers phase 1 (`cli`).

**Codebase verified:** 2026-02-12

**Last updated:** 2026-02-12

**Status:** Active and partially completed (review + CLI hardening/refactor implemented)

---

## Plan Revision (2026-02-12)

This phase was updated from review-only to include implementation work based on CLI findings.

Updated goals:
- Keep CLI boundary behavior stable while hardening failure paths.
- Reduce redundant code and dead paths.
- Make logging lifecycle deterministic across repeated CLI invocations.
- Preserve existing user-facing command semantics.

---

## Implementation Status

Completed implementation in this phase:
- Added fail-fast block metadata validation for `score` and `assoc/rhe` paths.
- Added reusable column-selection helper for `_prep_data`.
- Removed dead/unused CLI code:
  - unused global variables for multiprocessing helpers
  - unused parser wiring (`prs = _create_common_parser(...)`)
  - unused `_create_common_build_parser` helper
- Added robust CLI version fallback (`vunknown`) for editable/non-installed contexts.
- Added deterministic cleanup for CLI-attached logging handlers.
- Added warning/help text to make `compress --region` downstream compatibility clearer.

Completed test/verification work in this phase:
- Added `test_cli_version_fallback`.
- Added `test_prep_data_requires_block_metadata`.
- Verified Python compilation of updated files.
- Verified targeted runtime checks for new guard behavior and version fallback.

Known verification gap:
- Full `pytest` run in this environment is blocked by shared-memory multiprocessing permissions (`/psm_*`), so full CLI smoke validation remains to be re-run in a non-restricted environment.

---

## Acceptance Criteria Coverage

This phase implements and tests:

### src-review.AC1: CLI review completed
- **src-review.AC1.1 Success:** Findings for `src/linear_dag/cli.py` are documented with severity and exact file references.
- **src-review.AC1.2 Success:** Each finding includes concrete impact and a recommended remediation direction.
- **src-review.AC1.3 Success:** If no issue exists for a category, residual risk/testing gaps are explicitly stated.

---

<!-- START_TASK_1 -->
### Task 1: Build CLI review surface map

**Verifies:** src-review.AC1.1

**Status:** Completed

**Files:**
- Read: `src/linear_dag/cli.py`
- Read: `src/linear_dag/pipeline.py`
- Read: `src/linear_dag/core/lineararg.py`

**Implementation:**
- Enumerate CLI subcommands and the internal function they dispatch to.
- Identify cross-module assumptions required for CLI execution (block metadata presence, IID parsing contracts, package metadata availability).
- Record candidate failure boundaries for deeper validation in Task 2.

**Verification:**
- Run: `rg -n "set_defaults\(|def _prep_data|def _main" src/linear_dag/cli.py`
- Expected: Command returns CLI dispatch and data-prep anchors used in review notes.

**Commit:** `N/A (review phase, no code changes expected)`
<!-- END_TASK_1 -->

<!-- START_TASK_2 -->
### Task 2: Perform CLI defect analysis

**Verifies:** src-review.AC1.1, src-review.AC1.2

**Status:** Completed

**Files:**
- Read: `src/linear_dag/cli.py`
- Read: `src/linear_dag/core/lineararg.py`

**Implementation:**
- Validate null/shape assumptions and error paths for block metadata, logging setup, and package version retrieval.
- Confirm each suspected defect with line-level evidence and execution impact.
- Rank each finding as Critical, Important, or Minor.

**Verification:**
- Run: `nl -ba src/linear_dag/cli.py | sed -n '416,520p'`
- Run: `nl -ba src/linear_dag/cli.py | sed -n '796,860p'`
- Expected: Line-number evidence for each CLI finding.

**Commit:** `N/A (review phase, no code changes expected)`
<!-- END_TASK_2 -->

<!-- START_TASK_3 -->
### Task 3: Write CLI review section

**Verifies:** src-review.AC1.1, src-review.AC1.2, src-review.AC1.3

**Status:** Completed

**Files:**
- Create: `review output in assistant response` (CLI section)

**Implementation:**
- Produce final CLI findings grouped by severity.
- Include one entry for residual risk if a category has zero defects.
- Ensure each entry includes file path and line reference.

**Verification:**
- Manual check: every CLI finding contains `path:line` reference and a concrete impact statement.
- Expected: CLI review section is self-contained and actionable.

**Commit:** `N/A (review phase, no code changes expected)`
<!-- END_TASK_3 -->

<!-- START_TASK_4 -->
### Task 4: Implement CLI hardening and streamlining changes

**Verifies:** src-review.AC1.2

**Status:** Completed

**Files:**
- Modify: `src/linear_dag/cli.py`

**Implementation:**
- Added `_require_block_metadata` and wired it into `score` and `_prep_data`.
- Refactored duplicated column-selection logic into `_select_columns`.
- Removed redundant parser/helper code and unused globals.
- Standardized logger handler lifecycle via explicit add/remove/close behavior.
- Added resilient version lookup with `_resolve_cli_version`.

**Verification:**
- Run: `python -m compileall -q src/linear_dag/cli.py`
- Expected: no syntax errors.

**Commit:** `N/A (no commit requested in this session)`
<!-- END_TASK_4 -->

<!-- START_TASK_5 -->
### Task 5: Add focused CLI regression tests for new boundary behavior

**Verifies:** src-review.AC1.3

**Status:** Completed

**Files:**
- Modify: `tests/test_cli.py`

**Implementation:**
- Added test coverage for package-version fallback behavior.
- Added test coverage for fail-fast block metadata validation path.

**Verification:**
- Run: `python -m compileall -q tests/test_cli.py`
- Expected: no syntax errors.
- Manual runtime checks executed for `_resolve_cli_version` fallback and `_prep_data` error path.

**Commit:** `N/A (no commit requested in this session)`
<!-- END_TASK_5 -->
