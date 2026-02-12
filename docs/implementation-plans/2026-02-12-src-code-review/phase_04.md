# src Review Implementation Plan

**Goal:** Execute a source-level code review for the structure layer and finalize a complete `src/` review report.

**Architecture:** Review `src/linear_dag/structure` for numerical method choices and output contracts, then assemble all phase findings (`cli`, `core`, `association`, `structure`) into one consolidated response.

**Tech Stack:** Python 3, SciPy sparse eigensolvers (`eigs`, `svds`)

**Scope:** 4 phases from user-requested review (`cli`, `core`, `association`, `structure`), this file covers phase 4 (`structure`) plus final report assembly.

**Codebase verified:** 2026-02-12

**Last updated:** 2026-02-12

**Status:** Queued (review findings captured; remediation not started)

---

## Plan Revision (2026-02-12)

Execution priority has moved to CLI-first hardening. This phase remains valid and is currently pending until core and association remediation phases complete.

Current state:
- Structure review findings were produced.
- Consolidated review report was delivered.
- No structure remediation changes have been applied yet in this phase.

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
