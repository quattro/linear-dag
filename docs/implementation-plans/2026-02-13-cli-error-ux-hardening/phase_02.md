# CLI Error UX Hardening Implementation Plan

**Goal:** Wire closest-match suggestions into invalid block/chromosome/column-name error paths.

**Architecture:** Reuse Phase 1 suggestion helpers from `src/linear_dag/cli.py` and enrich existing `ValueError` messages in `_filter_blocks` and `_read_pheno_or_covar`. Keep existing validation flow intact while appending suggestion fragments when close matches exist.

**Tech Stack:** Python 3.11+, standard library (`difflib`), Polars, pytest.

**Scope:** Phase 2 of 4 from `/Users/nicholas/Projects/linear-dag/docs/design-plans/2026-02-13-cli-error-ux-hardening.md`.

**Codebase verified:** 2026-02-13 13:14:00 PST

---

## Acceptance Criteria Coverage

This phase implements and tests:

### cli-error-ux-hardening.AC1: Invalid selection inputs provide actionable suggestions
- **cli-error-ux-hardening.AC1.1 Success:** Invalid `--block-names` errors include nearest valid block-name suggestions when close matches exist.
- **cli-error-ux-hardening.AC1.2 Success:** Invalid `--chromosomes` errors include nearest valid chromosome suggestions when close matches exist, while preserving accepted forms such as `21` and `chr21`.
- **cli-error-ux-hardening.AC1.3 Success:** Missing column-name selection errors (for phenotype/covariate files) include nearest valid column-name suggestions when close matches exist.
- **cli-error-ux-hardening.AC1.4 Failure:** When no close matches exist, errors still include requested invalid values and a bounded list of available values.

---

<!-- START_SUBCOMPONENT_A (tasks 1-2) -->
<!-- START_TASK_1 -->
### Task 1: Add suggestion fragments to block/chromosome selection errors

**Verifies:** cli-error-ux-hardening.AC1.1, cli-error-ux-hardening.AC1.2, cli-error-ux-hardening.AC1.4

**Files:**
- Modify: `src/linear_dag/cli.py:557`

**Implementation:**
- In `_filter_blocks`, update invalid block-name error construction to append suggestion text from Phase 1 helper when close matches exist.
- Update invalid chromosome error construction similarly, using normalized values for matching and displaying canonical available chromosomes.
- Keep existing required content in all failure messages:
  - requested invalid value(s)
  - bounded available value list
- Ensure no suggestion fragment is appended when helper returns no match.

**Testing:**
- N/A (tests added in Task 3).

**Verification:**
- Run: `python -m compileall src/linear_dag/cli.py`
- Expected: compile succeeds.

**Commit:** `feat: add suggestions to block and chromosome validation errors`
<!-- END_TASK_1 -->

<!-- START_TASK_2 -->
### Task 2: Add suggestion fragments to named-column selection errors

**Verifies:** cli-error-ux-hardening.AC1.3, cli-error-ux-hardening.AC1.4

**Files:**
- Modify: `src/linear_dag/cli.py:255`

**Implementation:**
- In `_read_pheno_or_covar`, extend missing-name message to include close-match suggestions when available.
- Keep current message structure (path, missing names, available names) and append suggestion fragment only when match exists.
- Ensure multi-missing-name behavior remains deterministic (preserve order; avoid duplicate suggestions).

**Testing:**
- N/A (tests added in Task 3).

**Verification:**
- Run: `python -m compileall src/linear_dag/cli.py`
- Expected: compile succeeds.

**Commit:** `feat: add suggestions to missing column-name validation errors`
<!-- END_TASK_2 -->
<!-- END_SUBCOMPONENT_A -->

<!-- START_SUBCOMPONENT_B (tasks 3-4) -->
<!-- START_TASK_3 -->
### Task 3: Add CLI tests for suggestion-present and suggestion-absent paths

**Verifies:** cli-error-ux-hardening.AC1.1, cli-error-ux-hardening.AC1.2, cli-error-ux-hardening.AC1.3, cli-error-ux-hardening.AC1.4

**Files:**
- Modify: `tests/test_cli.py:195`
- Modify: `tests/test_cli.py:522`

**Implementation:**
- Add helper-level tests:
  - invalid block name typo (near existing block name) includes `Did you mean` with suggested block;
  - invalid chromosome typo (near existing chromosome form) includes suggestion;
  - missing column typo (for example `heigt`) includes suggestion `height`;
  - unrelated invalid values still produce actionable base message without suggestion text.
- Add `run_cli` stderr-level assertions for at least one suggestion-present case to confirm user-facing output path.
- Keep tests deterministic by using `tests/testdata/test_chr21_50.h5` and `tests/testdata/phenotypes_50.tsv`.

**Testing:**
- Run: `pytest -q tests/test_cli.py -k "suggestion and (block or chromosome or covar or column)"`
- Expected: all suggestion-path tests pass.

**Verification:**
- Run: `pytest -q tests/test_cli.py -k "load_required_block_metadata or read_pheno_or_covar or run_cli_invalid"`
- Expected: existing invalid-input tests and new suggestion tests pass together.

**Commit:** `test: cover cli suggestion enrichment for invalid selections`
<!-- END_TASK_3 -->

<!-- START_TASK_4 -->
### Task 4: Run full CLI test file and finalize phase

**Verifies:** cli-error-ux-hardening.AC1.1, cli-error-ux-hardening.AC1.2, cli-error-ux-hardening.AC1.3, cli-error-ux-hardening.AC1.4

**Files:**
- Validate: `src/linear_dag/cli.py`
- Validate: `tests/test_cli.py`

**Implementation:**
- Run formatter/lint on touched files.
- Run complete CLI test file to catch regressions across unrelated command paths.

**Testing:**
- Run: `ruff format src/linear_dag/cli.py tests/test_cli.py`
- Run: `ruff check src/linear_dag/cli.py tests/test_cli.py`
- Run: `pytest -q tests/test_cli.py`

**Verification:**
- Expected: formatting/lint pass and full CLI test module passes.

**Commit:** `chore: finalize phase 2 selection error suggestion coverage`
<!-- END_TASK_4 -->
<!-- END_SUBCOMPONENT_B -->
