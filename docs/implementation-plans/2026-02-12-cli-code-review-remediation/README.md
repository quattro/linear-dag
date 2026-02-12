# CLI Review/Remediation Plan Status

Last updated: 2026-02-12

## Goal
Run a focused review/remediation loop for `src/linear_dag/cli.py` with emphasis on:
- redundant code reduction
- CLI argument grouping and consistency
- consistent `from_hdf5` boundary argument handling
- consistent user-facing behavior across `assoc`, `rhe`, and `score`

## Phase Status
- `phase_01.md`: completed (review + implementation + re-review complete; pytest rerun pending non-restricted runtime)

## Work Completed
- Completed severity-ranked CLI review focused on redundancy, argument grouping, and operator-constructor consistency.
- Implemented parser argument groups for related options in `assoc`, `rhe`, `score`, and common parser paths.
- Reduced repeated boundary setup logic with shared helper functions.
- Aligned `assoc` and `rhe` operator initialization behavior around filtered block metadata and normalized process-count validation.
- Added focused tests in `tests/test_cli.py` for the new CLI contracts.

## Verification Notes
- `ruff check` and `compileall` passed for modified Python files.
- Targeted manual checks for new assertions passed.
- Full `pytest` remains pending in this sandbox-constrained environment.
