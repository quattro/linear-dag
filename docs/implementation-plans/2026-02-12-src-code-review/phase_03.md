# src Review Implementation Plan

**Goal:** Execute a source-level code review for the association/statistics layer and produce a severity-ranked findings report.

**Architecture:** Review `src/linear_dag/association` for algorithmic assumptions, data-alignment guards, and API contracts crossing into core operators and CLI execution.

**Tech Stack:** Python 3, NumPy/SciPy, Polars, shared-memory operator integration

**Scope:** 4 phases from user-requested review (`cli`, `core`, `association`, `structure`), this file covers phase 3 (`association`).

**Codebase verified:** 2026-02-12

**Last updated:** 2026-02-12

**Status:** Queued (review findings captured; remediation not started)

---

## Plan Revision (2026-02-12)

Execution priority has moved to CLI-first hardening. This phase remains valid and is sequenced after core remediation work.

Current state:
- Association review findings were produced.
- No association remediation changes have been applied yet in this phase.

---

## Acceptance Criteria Coverage

This phase implements and tests:

### src-review.AC3: Association review completed
- **src-review.AC3.1 Success:** Findings for `src/linear_dag/association/*` are documented with severity and exact file references.
- **src-review.AC3.2 Success:** Statistical-contract and data-alignment risks are explicitly identified.
- **src-review.AC3.3 Success:** Runtime guardrails are validated for non-HWE and missing-data paths.

---

<!-- START_TASK_1 -->
### Task 1: Build association contract map

**Verifies:** src-review.AC3.1

**Files:**
- Read: `src/linear_dag/association/__init__.py`
- Read: `src/linear_dag/association/gwas.py`
- Read: `src/linear_dag/association/heritability.py`
- Read: `src/linear_dag/association/prs.py`

**Implementation:**
- Enumerate exported APIs and required input contracts.
- Identify guard conditions for intercept, sample-ID merge, and non-HWE execution.

**Verification:**
- Run: `rg -n "assume_hwe|number_of_heterozygotes|covariates\[:, 0\]|Merge failed" src/linear_dag/association/gwas.py src/linear_dag/association/heritability.py`
- Expected: Anchor points for contract review.

**Commit:** `N/A (review phase, no code changes expected)`
<!-- END_TASK_1 -->

<!-- START_TASK_2 -->
### Task 2: Perform association defect analysis

**Verifies:** src-review.AC3.1, src-review.AC3.2, src-review.AC3.3

**Files:**
- Read: `src/linear_dag/association/gwas.py`
- Read: `src/linear_dag/core/parallel_processing.py`

**Implementation:**
- Validate whether non-HWE preconditions are correctly enforced before expensive computation.
- Confirm that guardrails fail early with clear, actionable errors.
- Rank issues by user-facing failure mode.

**Verification:**
- Run: `nl -ba src/linear_dag/association/gwas.py | sed -n '178,236p'`
- Run: `nl -ba src/linear_dag/core/parallel_processing.py | sed -n '463,490p'`
- Expected: Evidence for guard-check correctness and runtime behavior.

**Commit:** `N/A (review phase, no code changes expected)`
<!-- END_TASK_2 -->

<!-- START_TASK_3 -->
### Task 3: Write association review section

**Verifies:** src-review.AC3.1, src-review.AC3.2, src-review.AC3.3

**Files:**
- Create: `review output in assistant response` (Association section)

**Implementation:**
- Produce final association findings grouped by severity.
- Include concrete fixes that preserve existing public APIs where possible.

**Verification:**
- Manual check: every association finding includes concrete impact + fix direction.
- Expected: Association section is operationally actionable.

**Commit:** `N/A (review phase, no code changes expected)`
<!-- END_TASK_3 -->
