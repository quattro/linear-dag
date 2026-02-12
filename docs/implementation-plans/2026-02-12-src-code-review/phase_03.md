# src Review Implementation Plan

**Goal:** Execute a source-level code review for the association/statistics layer and implement high-impact fixes with regression coverage.

**Architecture:** Review `src/linear_dag/association` for algorithmic assumptions, data-alignment guards, and API contracts crossing into core operators and CLI execution.

**Tech Stack:** Python 3, NumPy/SciPy, Polars, shared-memory operator integration

**Scope:** 4 phases from user-requested review (`cli`, `core`, `association`, `structure`), this file covers phase 3 (`association`).

**Codebase verified:** 2026-02-12

**Last updated:** 2026-02-12

**Status:** Active and partially completed (review + selected association remediation implemented)

---

## Plan Revision (2026-02-12)

Execution priority moved to CLI-first hardening and then core remediation. Association work is now active with targeted changes aligned to current goals:
- reduce redundant code in hot association paths
- streamline data processing for `run_gwas`
- keep failure behavior and logging at the existing boundary contracts

Current state:
- Association review findings were produced.
- Selected remediation changes are now implemented in `gwas.py` with focused tests.

---

## Implementation Status

Completed implementation in this phase:
- Hardened non-HWE preconditions via `_validate_non_hwe_genotypes(...)` so `run_gwas(..., assume_hwe=False)` fails early unless `genotypes` exposes `n_individuals`, `iids`, and `number_of_heterozygotes()`.
- Removed redundant `data.select(...).collect()` calls in `run_gwas` by collecting required `iid`/phenotype/covariate columns once and reusing them.
- Corrected `get_gwas_beta_se` return contract annotation/docstring to match runtime behavior (`beta`, `var_numerator`, `var_denominator`, `allele_counts`).

Completed test/verification work in this phase:
- Added focused association regression tests in `tests/test_association.py`:
  - `test_run_gwas_non_hwe_requires_heterozygote_counter`
  - `test_get_gwas_beta_se_returns_four_arrays`
- Verified Python compilation for updated files.
- Verified new behavior with targeted manual runtime checks that exercise the new guard and return-shape contract.

Known verification gap:
- `pytest` in this environment exits abnormally with code `-1` and no diagnostics, so direct pytest execution remains pending in a non-restricted runtime.

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

**Status:** Completed

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

**Status:** Completed

**Files:**
- Read: `src/linear_dag/association/gwas.py`
- Read: `src/linear_dag/core/parallel_processing.py`

**Implementation:**
- Validate whether non-HWE preconditions are correctly enforced before expensive computation.
- Confirm that guardrails fail early with clear, actionable errors.
- Rank issues by user-facing failure mode.

**Verification:**
- Run: `nl -ba src/linear_dag/association/gwas.py | sed -n '188,270p'`
- Run: `nl -ba src/linear_dag/core/parallel_processing.py | sed -n '250,350p'`
- Expected: Evidence for guard-check correctness and runtime behavior.

**Commit:** `N/A (review phase, no code changes expected)`
<!-- END_TASK_2 -->

<!-- START_TASK_3 -->
### Task 3: Write association review section

**Verifies:** src-review.AC3.1, src-review.AC3.2, src-review.AC3.3

**Status:** Completed

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

<!-- START_TASK_4 -->
### Task 4: Implement association guard and streamlining fixes in `gwas.py`

**Verifies:** src-review.AC3.2, src-review.AC3.3

**Status:** Completed

**Files:**
- Modify: `src/linear_dag/association/gwas.py`

**Implementation:**
- Added explicit non-HWE boundary validation for required genotype capabilities.
- Removed redundant data collection calls by reusing one collected association input frame.
- Corrected `get_gwas_beta_se` return contract docs/types to match runtime outputs.

**Verification:**
- Run: `python -m compileall -q src/linear_dag/association/gwas.py`
- Expected: no syntax errors.

**Commit:** `N/A (no commit requested yet for this phase)`
<!-- END_TASK_4 -->

<!-- START_TASK_5 -->
### Task 5: Add focused regression tests for association guard and return contract

**Verifies:** src-review.AC3.1, src-review.AC3.3

**Status:** Completed

**Files:**
- Modify: `tests/test_association.py`

**Implementation:**
- Added low-overhead regression test for non-HWE guardrail behavior.
- Added low-overhead regression test for `get_gwas_beta_se` output contract.

**Verification:**
- Run: `python -m compileall -q tests/test_association.py`
- Run: `pytest -q tests/test_association.py -k "non_hwe_requires_heterozygote_counter or returns_four_arrays"`
- Expected: new tests pass (note: pytest currently exits `-1` in this environment; equivalent manual runtime checks passed).

**Commit:** `N/A (no commit requested yet for this phase)`
<!-- END_TASK_5 -->
