# src Review Implementation Plan

**Goal:** Execute a source-level code review for the structure layer and implement numerical-contract hardening while preserving the existing public API shape.

**Architecture:** Review `src/linear_dag/structure` for solver choice, output determinism, and boundary validation; keep `pca`/`svd` thin wrappers around SciPy with stronger contracts.

**Tech Stack:** Python 3, SciPy sparse eigensolvers (`eigsh`, `svds`)

**Scope:** 4 phases from user-requested review (`cli`, `core`, `association`, `structure`), this file covers phase 4 (`structure`) plus final report assembly.

**Codebase verified:** 2026-02-12

**Last updated:** 2026-02-12

**Status:** Active and partially completed (review + selected structure remediation implemented)

---

## Plan Revision (2026-02-12)

Execution priority moved to CLI-first hardening, then core and association remediation. Structure work is now active with targeted goals:
- remove redundant ambiguity in solver outputs by enforcing deterministic ordering
- harden boundary validation for rank `k`
- keep structure APIs thin and consistent with existing usage

Current state:
- Structure review findings were produced.
- Selected remediation changes are now implemented in `infer.py` with focused tests.

---

## Implementation Status

Completed implementation in this phase:
- Switched PCA from `eigs` to `eigsh` for symmetric GRM inputs to avoid avoidable complex-eigenpair behavior.
- Added explicit rank validation for both `svd` and `pca` (`k` type/range checks with actionable errors).
- Enforced deterministic descending ordering for returned singular/eigen values and corresponding vectors.

Completed test/verification work in this phase:
- Added focused structure tests in `tests/test_structure.py`:
  - `test_svd_returns_sorted_singular_values`
  - `test_pca_returns_sorted_real_eigenpairs`
  - `test_structure_rank_validation`
- Verified Python compilation and lint checks for updated structure files.
- Verified behavior with targeted manual runtime checks for ordering and validation semantics.

Known verification gap:
- `pytest` in this environment exits abnormally with code `-1` and no diagnostics, so direct pytest execution remains pending in a non-restricted runtime.

---

## Acceptance Criteria Coverage

This phase implements and tests:

### src-review.AC4: Structure review completed
- **src-review.AC4.1 Success:** Findings for `src/linear_dag/structure/*` are documented with severity and exact file references.
- **src-review.AC4.2 Success:** Numerical-method risks that affect downstream outputs are identified.

### src-review.AC5: Consolidated review report delivered
- **src-review.AC5.1 Success:** Final report is broken down by `cli`, `core`, `association`, and `structure`.
- **src-review.AC5.2 Success:** Findings are ordered by severity with a brief residual-risk note where no issues are found.

---

<!-- START_TASK_1 -->
### Task 1: Perform structure defect analysis

**Verifies:** src-review.AC4.1, src-review.AC4.2

**Status:** Completed

**Files:**
- Read: `src/linear_dag/structure/infer.py`

**Implementation:**
- Validate eigensolver selection against GRM properties.
- Check output-type behavior (real vs complex) and ordering guarantees.
- Rank findings by downstream statistical impact.

**Verification:**
- Run: `nl -ba src/linear_dag/structure/infer.py`
- Expected: Evidence for method-choice findings.

**Commit:** `N/A (review phase, no code changes expected)`
<!-- END_TASK_1 -->

<!-- START_TASK_2 -->
### Task 2: Assemble consolidated review report

**Verifies:** src-review.AC5.1, src-review.AC5.2

**Status:** Completed

**Files:**
- Create: `review output in assistant response` (final 4-section review)

**Implementation:**
- Merge findings from all phases into one report.
- Order issues by severity, then by module.
- Include explicit "No findings" statement for any module with none.

**Verification:**
- Manual check: report has exactly four module sections (`cli`, `core`, `association`, `structure`) and includes file/line references.
- Expected: Review output is complete and directly actionable.

**Commit:** `N/A (review phase, no code changes expected)`
<!-- END_TASK_2 -->

<!-- START_TASK_3 -->
### Task 3: Implement structure numerical-contract fixes in `infer.py`

**Verifies:** src-review.AC4.2

**Status:** Completed

**Files:**
- Modify: `src/linear_dag/structure/infer.py`
- Modify: `src/linear_dag/structure/AGENTS.md`

**Implementation:**
- Replaced non-symmetric solver usage in PCA with `eigsh`.
- Added shared rank-validation helper used by `pca` and `svd`.
- Sorted returned spectral values/vectors in descending order.
- Updated structure-domain invariants in AGENTS context to match new behavior.

**Verification:**
- Run: `python -m compileall -q src/linear_dag/structure/infer.py`
- Run: `ruff check src/linear_dag/structure/infer.py`
- Expected: no syntax/lint errors.

**Commit:** `N/A (no commit requested yet for this phase)`
<!-- END_TASK_3 -->

<!-- START_TASK_4 -->
### Task 4: Add focused regression tests for structure behavior

**Verifies:** src-review.AC4.1, src-review.AC4.2

**Status:** Completed

**Files:**
- Create: `tests/test_structure.py`

**Implementation:**
- Added deterministic-output tests for SVD and PCA ordering/contracts.
- Added explicit rank/type validation tests for `k` boundary behavior.

**Verification:**
- Run: `python -m compileall -q tests/test_structure.py`
- Run: `pytest -q tests/test_structure.py`
- Expected: tests pass (note: pytest currently exits `-1` in this environment; equivalent manual runtime checks passed).

**Commit:** `N/A (no commit requested yet for this phase)`
<!-- END_TASK_4 -->
