# src Review Implementation Plan

**Goal:** Execute a source-level code review for the core graph/operator layer and produce a severity-ranked findings report.

**Architecture:** Review `src/linear_dag/core` as the foundational contract layer (graph state, serialization, block metadata, and parallel operators). Validate invariants expected by CLI and association layers.

**Tech Stack:** Python 3, NumPy/SciPy sparse operators, HDF5, multiprocessing shared memory

**Scope:** 4 phases from user-requested review (`cli`, `core`, `association`, `structure`), this file covers phase 2 (`core`).

**Codebase verified:** 2026-02-12

**Last updated:** 2026-02-12

**Status:** Queued (review findings captured; remediation not started)

---

## Plan Revision (2026-02-12)

Execution priority has moved to CLI-first hardening. This phase remains valid and is next in the remediation queue after CLI validation is fully complete in a non-restricted runtime.

Current state:
- Core review findings were produced.
- No core remediation changes have been applied yet in this phase.

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
