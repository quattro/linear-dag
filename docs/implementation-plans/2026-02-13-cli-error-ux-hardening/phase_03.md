# CLI Error UX Hardening Implementation Plan

**Goal:** Add explicit pre-validation for out-of-range phenotype/covariate integer column selections.

**Architecture:** Keep integer-column validation inside `_read_pheno_or_covar` so both `assoc` and `rhe` flows get consistent behavior. Validate bounds from header metadata before full `pl.read_csv(..., columns=[...])` execution.

**Tech Stack:** Python 3.11+, Polars, pytest.

**Scope:** Phase 3 of 4 from `/Users/nicholas/Projects/linear-dag/docs/design-plans/2026-02-13-cli-error-ux-hardening.md`.

**Codebase verified:** 2026-02-13 13:14:00 PST

---

## Acceptance Criteria Coverage

This phase implements and tests:

### cli-error-ux-hardening.AC2: Column-index selection errors fail early with explicit bounds
- **cli-error-ux-hardening.AC2.1 Success:** Out-of-range `--pheno-col-nums` values fail before full CSV load with a clear message including observed index, valid range, and column count.
- **cli-error-ux-hardening.AC2.2 Success:** Out-of-range `--covar-col-nums` values fail with the same explicit bounds format.
- **cli-error-ux-hardening.AC2.3 Failure:** Negative indices remain rejected with an explicit validation error.

---

<!-- START_SUBCOMPONENT_A (tasks 1-2) -->
<!-- START_TASK_1 -->
### Task 1: Add integer column-index bounds validation helper

**Verifies:** cli-error-ux-hardening.AC2.1, cli-error-ux-hardening.AC2.2

**Files:**
- Modify: `src/linear_dag/cli.py:255`

**Implementation:**
- Add a private helper in `cli.py` that validates integer column indices against the file header width:
  - read `n_rows=0` header to obtain column count;
  - compute min/max requested index;
  - if any requested index is out of bounds, raise `ValueError` with:
    - offending index (or indices),
    - valid range (`0..N-1`),
    - total column count and file path.
- Keep the helper pure and reusable by `_read_pheno_or_covar`.

**Testing:**
- N/A (tests added in Task 3).

**Verification:**
- Run: `python -m compileall src/linear_dag/cli.py`
- Expected: compile succeeds.

**Commit:** `feat: add column-index bounds pre-validation helper`
<!-- END_TASK_1 -->

<!-- START_TASK_2 -->
### Task 2: Wire bounds validation into `_read_pheno_or_covar`

**Verifies:** cli-error-ux-hardening.AC2.1, cli-error-ux-hardening.AC2.2, cli-error-ux-hardening.AC2.3

**Files:**
- Modify: `src/linear_dag/cli.py:268`

**Implementation:**
- In the `all_int` branch of `_read_pheno_or_covar`:
  - preserve existing negative-index guard (`x < 0`) for AC2.3;
  - call the new bounds helper before executing `pl.read_csv(..., columns=columns)`.
- Ensure error type remains `ValueError` for both negative and out-of-range cases.

**Testing:**
- N/A (tests added in Task 3).

**Verification:**
- Run: `python -m compileall src/linear_dag/cli.py`
- Expected: compile succeeds.

**Commit:** `feat: enforce early bounds checks for integer column selections`
<!-- END_TASK_2 -->
<!-- END_SUBCOMPONENT_A -->

<!-- START_SUBCOMPONENT_B (tasks 3-4) -->
<!-- START_TASK_3 -->
### Task 3: Add tests for pheno/covar out-of-range index failures

**Verifies:** cli-error-ux-hardening.AC2.1, cli-error-ux-hardening.AC2.2, cli-error-ux-hardening.AC2.3

**Files:**
- Modify: `tests/test_cli.py:241`
- Modify: `tests/test_cli.py:247`
- Modify: `tests/test_cli.py:522`

**Implementation:**
- Add direct helper tests:
  - `_read_pheno_or_covar(..., columns=[999])` raises `ValueError` with bounds details;
  - negative index still raises the existing explicit validation error.
- Add integration-level tests through `_prep_data`:
  - invalid `pheno_col_nums` path raises AC2.1-style message;
  - invalid `covar_col_nums` path raises AC2.2-style message.
- Add one `run_cli` test that uses `--pheno-col-nums` or `--covar-col-nums` out-of-range and asserts stderr includes bounds text.

**Testing:**
- Run: `pytest -q tests/test_cli.py -k "col_nums or column indices or out-of-range"`
- Expected: all new bounds tests pass.

**Verification:**
- Run: `pytest -q tests/test_cli.py -k "read_pheno_or_covar or prep_data or run_cli"`
- Expected: no regressions in existing CLI validation tests.

**Commit:** `test: cover early bounds validation for pheno and covar column indices`
<!-- END_TASK_3 -->

<!-- START_TASK_4 -->
### Task 4: Validate full CLI test module after bounds changes

**Verifies:** cli-error-ux-hardening.AC2.1, cli-error-ux-hardening.AC2.2, cli-error-ux-hardening.AC2.3

**Files:**
- Validate: `src/linear_dag/cli.py`
- Validate: `tests/test_cli.py`

**Implementation:**
- Run formatter/lint checks on touched files.
- Run the full CLI test module for final phase verification.

**Testing:**
- Run: `ruff format src/linear_dag/cli.py tests/test_cli.py`
- Run: `ruff check src/linear_dag/cli.py tests/test_cli.py`
- Run: `pytest -q tests/test_cli.py`

**Verification:**
- Expected: formatting/lint pass and full CLI tests pass.

**Commit:** `chore: finalize phase 3 column-index validation hardening`
<!-- END_TASK_4 -->
<!-- END_SUBCOMPONENT_B -->
