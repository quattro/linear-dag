# CLI Review/Remediation Implementation Plan

**Goal:** Review and remediate `src/linear_dag/cli.py` for redundancy, argument-UX consistency, and operator-construction consistency while preserving user-visible command contracts.

**Architecture:** Keep CLI as a thin boundary layer: parser + validation + dispatch. Centralize repeated block/operator setup and standardize parser argument groups per workflow (`assoc`, `rhe`, `score`, `compress`, `multi-step-compress`).

**Tech Stack:** Python 3, argparse, Polars, `ParallelOperator.from_hdf5`, `GRMOperator.from_hdf5`

**Scope:** Single-phase CLI remediation focused on `src/linear_dag/cli.py` and `tests/test_cli.py`.

**Codebase verified:** 2026-02-12

**Last updated:** 2026-02-12

**Status:** Completed (review + remediation + re-review complete; pytest blocked by environment)

---

## Implementation Status

Completed implementation in this phase:
- Added CLI argument groups for related options in common parser and `assoc`/`rhe`/`score` subcommands.
- Added shared helpers to reduce redundancy for boundary setup:
  - `_validate_num_processes`
  - `_load_required_block_metadata`
  - `_build_parallel_operator_kwargs`
- Aligned operator-constructor behavior:
  - `assoc` call sites now use one shared kwargs builder for `ParallelOperator.from_hdf5`
  - `rhe` now passes filtered `block_metadata` into `GRMOperator.from_hdf5`
- Added fail-fast score input checks for required boundary inputs.
- Added focused tests for:
  - operator kwargs consistency across assoc modes
  - `rhe` propagation of filtered block metadata
  - argument-group visibility in CLI help output
  - non-positive `num_processes` boundary rejection

Completed verification in this phase:
- `ruff check src/linear_dag/cli.py tests/test_cli.py`
- `python -m compileall -q src/linear_dag/cli.py tests/test_cli.py`
- Targeted manual runtime checks that mirror new tests and assertions.

Known verification gap:
- `pytest` exits with code `-1` in this sandbox, so full pytest verification remains pending in a non-restricted runtime.

---

## Acceptance Criteria Coverage

This phase implements and tests:

### cli-remed.AC1: Redundant code reduced and CLI structure clarified
- **cli-remed.AC1.1 Success:** Repeated CLI setup logic (block loading/filtering/validation, operator kwargs assembly) is centralized without changing command semantics.
- **cli-remed.AC1.2 Success:** Parser definitions use argument groups for related options and keep subcommand contracts stable.

### cli-remed.AC2: Operator construction and user behavior are consistent
- **cli-remed.AC2.1 Success:** `from_hdf5` call sites in CLI entrypoints pass consistent boundary arguments (`num_processes`, filtered `block_metadata`, and operation-specific options).
- **cli-remed.AC2.2 Success:** `rhe` respects the same block-selection behavior (`--chromosomes`, `--block-names`) already exposed by common parser paths.
- **cli-remed.AC2.3 Success:** CLI-level validation errors remain fail-fast and actionable.

### cli-remed.AC3: Requesting-code-review loop completed
- **cli-remed.AC3.1 Success:** Initial findings are severity-ranked for `src/linear_dag/cli.py`.
- **cli-remed.AC3.2 Success:** Post-fix re-review confirms all critical/important/minor findings are resolved.

---

<!-- START_TASK_1 -->
### Task 1: Baseline CLI review (requesting-code-review pass 1)

**Verifies:** cli-remed.AC3.1

**Status:** Completed

**Files:**
- Read: `src/linear_dag/cli.py`
- Read: `tests/test_cli.py`

**Implementation:**
- Produce severity-ranked findings focused on:
  - redundant code and duplicated setup paths
  - parser ergonomics and missing argument grouping
  - inconsistent `from_hdf5` boundary argument usage
  - user-facing behavior mismatches across `assoc`, `rhe`, and `score`
- Capture each finding with exact `path:line`.

**Verification:**
- Run: `rg -n "from_hdf5|add_argument\\(|_prep_data|_filter_blocks|_require_block_metadata|set_defaults\\(func" src/linear_dag/cli.py`
- Expected: Anchor points for all targeted CLI concerns.

**Commit:** `N/A (review task)`
<!-- END_TASK_1 -->

<!-- START_TASK_2 -->
### Task 2: Refactor parser definitions with argument groups

**Verifies:** cli-remed.AC1.2, cli-remed.AC2.3

**Status:** Completed

**Files:**
- Modify: `src/linear_dag/cli.py`

**Implementation:**
- Introduce `argparse` argument groups for related options, while preserving existing flag names and defaults:
  - common parser groups: input/selection, phenotype-covariate columns, execution/output
  - `assoc` groups: association model options, variant metadata/filtering options
  - `rhe` groups: estimator/sampling options
  - `score` groups: input/selection, execution/output
- Keep subcommand dispatch via `set_defaults(func=...)`.

**Testing:**
- Add parser-structure tests in `tests/test_cli.py` that verify key arguments remain available under expected subcommands.

**Verification:**
- Run: `python -m compileall -q src/linear_dag/cli.py tests/test_cli.py`
- Expected: no syntax errors.

**Commit:** `refactor: group CLI arguments by workflow concerns`
<!-- END_TASK_2 -->

<!-- START_TASK_3 -->
### Task 3: Centralize block/operator boundary setup and align `from_hdf5` usage

**Verifies:** cli-remed.AC1.1, cli-remed.AC2.1, cli-remed.AC2.2, cli-remed.AC2.3

**Status:** Completed

**Files:**
- Modify: `src/linear_dag/cli.py`
- Modify: `tests/test_cli.py`

**Implementation:**
- Introduce shared helper(s) for CLI boundary setup:
  - load/filter/validate block metadata once per command path
  - validate/normalize `num_processes` consistently
  - assemble consistent `from_hdf5` kwargs
- Update call sites so operator construction is explicit and consistent:
  - `ParallelOperator.from_hdf5` in `assoc` paths
  - `GRMOperator.from_hdf5` in `rhe` path, now passing filtered `block_metadata`
- Preserve operation-specific settings (`max_num_traits`, MAF/BED filtering where applicable) without widening public CLI surface unexpectedly.

**Testing:**
- Add regression tests to verify:
  - `rhe` uses filtered blocks selected by `--chromosomes`/`--block-names`
  - call-site kwargs consistency for operator constructors via monkeypatch/spies

**Verification:**
- Run: `ruff check src/linear_dag/cli.py tests/test_cli.py`
- Run: `python -m compileall -q src/linear_dag/cli.py tests/test_cli.py`
- Expected: lint and compile checks pass.

**Commit:** `fix: align CLI operator setup across assoc/rhe/score`
<!-- END_TASK_3 -->

<!-- START_TASK_4 -->
### Task 4: Requesting-code-review re-review (pass 2+) and resolve all findings

**Verifies:** cli-remed.AC3.2

**Status:** Completed

**Files:**
- Read/Modify as needed: `src/linear_dag/cli.py`, `tests/test_cli.py`

**Implementation:**
- Re-run severity-ranked code review on CLI changes.
- Fix all remaining issues (critical, important, and minor) before closing phase.

**Verification:**
- Run: `ruff check src/linear_dag/cli.py tests/test_cli.py`
- Run: `pytest -q tests/test_cli.py`
- Expected: zero unresolved review findings; tests pass (or document sandbox-specific execution limits).

**Commit:** `fix: address CLI re-review findings`
<!-- END_TASK_4 -->
