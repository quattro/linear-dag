# src Review Implementation Plan

**Goal:** Execute a source-level code review for the core graph/operator layer and produce a severity-ranked findings report.

**Architecture:** Review `src/linear_dag/core` as the foundational contract layer (graph state, serialization, block metadata, and parallel operators). Validate invariants expected by CLI and association layers.

**Tech Stack:** Python 3, NumPy/SciPy sparse operators, HDF5, multiprocessing shared memory

**Scope:** 4 phases from user-requested review (`cli`, `core`, `association`, `structure`), this file covers phase 2 (`core`).

**Codebase verified:** 2026-02-12

**Last updated:** 2026-02-12

**Status:** Active and partially completed (review + selected core remediation implemented)

---

## Plan Revision (2026-02-12)

Execution priority moved to CLI-first hardening; core remediation has now started.

Current state:
- Core review findings were produced.
- Selected core remediation changes have been applied in `lineararg.py`.

---

## Implementation Status

Completed implementation in this phase:
- Implemented `LinearARG.copy()` with deep-copy behavior for matrix/index arrays and metadata fields.
- Fixed `add_individual_nodes(sex=...)` to preserve explicit `sex` input on returned object.
- Hardened `list_blocks()` sorting to support non-numeric chromosome names (e.g., `chrX`) without crashing.

Completed test/verification work in this phase:
- Added focused tests in `tests/test_lineararg.py`:
  - `test_lineararg_copy_independent_arrays`
  - `test_add_individual_nodes_propagates_explicit_sex`
  - `test_list_blocks_handles_non_numeric_chromosomes`
- Verified behavior via targeted runtime checks for each added test scenario.
- Verified Python compilation of updated files.

Known verification gap:
- `pytest` invocation in this environment exits abnormally without diagnostics (`exit -1`), so full suite execution remains pending in a non-restricted runtime.

---

## Acceptance Criteria Coverage

This phase implements and tests:

### src-review.AC2: Core review completed
- **src-review.AC2.1 Success:** Findings for `src/linear_dag/core/*` are documented with severity and exact file references.
- **src-review.AC2.2 Success:** Public-contract breakages and invariant violations are prioritized ahead of style issues.
- **src-review.AC2.3 Success:** Review includes serialization and block-metadata edge cases.

---

<!-- START_TASK_1 -->
### Task 1: Build core contract map

**Verifies:** src-review.AC2.1

**Status:** Completed

**Files:**
- Read: `src/linear_dag/core/__init__.py`
- Read: `src/linear_dag/core/lineararg.py`
- Read: `src/linear_dag/core/parallel_processing.py`

**Implementation:**
- Enumerate public exports and core invariants (shape/index alignment, serialization requirements, block semantics).
- Identify high-risk surfaces: `copy`, `add_individual_nodes`, `list_blocks`, worker orchestration paths.

**Verification:**
- Run: `rg -n "def copy\(|def add_individual_nodes\(|def list_blocks\(" src/linear_dag/core/lineararg.py`
- Expected: Anchors for core high-risk contract review.

**Commit:** `N/A (review phase, no code changes expected)`
<!-- END_TASK_1 -->

<!-- START_TASK_2 -->
### Task 2: Perform core defect analysis

**Verifies:** src-review.AC2.1, src-review.AC2.2, src-review.AC2.3

**Status:** Completed

**Files:**
- Read: `src/linear_dag/core/lineararg.py`
- Read: `src/linear_dag/core/parallel_processing.py`
- Read: `src/linear_dag/pipeline.py`

**Implementation:**
- Confirm contract violations (incorrect return behavior, dropped state, unsupported chromosome naming, root-block behavior).
- Validate cross-module impact from core defects into CLI/association.
- Rank findings by downstream blast radius.

**Verification:**
- Run: `nl -ba src/linear_dag/core/lineararg.py | sed -n '420,920p'`
- Run: `nl -ba src/linear_dag/pipeline.py | sed -n '109,170p'`
- Expected: Evidence for contract defects and their impact paths.

**Commit:** `N/A (review phase, no code changes expected)`
<!-- END_TASK_2 -->

<!-- START_TASK_3 -->
### Task 3: Write core review section

**Verifies:** src-review.AC2.1, src-review.AC2.2, src-review.AC2.3

**Status:** Completed

**Files:**
- Create: `review output in assistant response` (Core section)

**Implementation:**
- Produce final core findings grouped by severity with direct impact statements.
- Include remediation direction for each finding.

**Verification:**
- Manual check: every core finding has `path:line` and traces to a contract/invariant.
- Expected: Core section is prioritized and technically defensible.

**Commit:** `N/A (review phase, no code changes expected)`
<!-- END_TASK_3 -->

<!-- START_TASK_4 -->
### Task 4: Implement core contract fixes in `lineararg.py`

**Verifies:** src-review.AC2.2, src-review.AC2.3

**Status:** Completed

**Files:**
- Modify: `src/linear_dag/core/lineararg.py`

**Implementation:**
- Implemented `LinearARG.copy()` (previously unimplemented).
- Corrected `add_individual_nodes()` to retain explicit sex vector.
- Updated block-name sorting logic in `list_blocks()` to support non-numeric chromosomes.

**Verification:**
- Run: `python -m compileall -q src/linear_dag/core/lineararg.py`
- Expected: no syntax errors.

**Commit:** `N/A (no commit requested yet for this phase)`
<!-- END_TASK_4 -->

<!-- START_TASK_5 -->
### Task 5: Add targeted regression tests for core fixes

**Verifies:** src-review.AC2.1, src-review.AC2.3

**Status:** Completed

**Files:**
- Modify: `tests/test_lineararg.py`

**Implementation:**
- Added tests for copy semantics, explicit sex propagation, and non-numeric chromosome block sorting.

**Verification:**
- Run: `python -m compileall -q tests/test_lineararg.py`
- Expected: no syntax errors.
- Manual runtime checks executed for all new scenarios.

**Commit:** `N/A (no commit requested yet for this phase)`
<!-- END_TASK_5 -->
