# CLI Logging Unification Implementation Plan

**Goal:** Remove remaining CLI-relevant downstream print paths and ensure logger-based reporting is available through core/genotype execution routes touched by CLI and pipeline workflows.

**Architecture:** Audit downstream print usage in `genotype` and `core` modules, then extend optional logger threading for CLI-relevant call chains. Preserve non-CLI compatibility while eliminating unconditional print output in paths used by CLI commands and pipeline steps.

**Tech Stack:** Python 3.11+, pytest, caplog/capsys.

**Scope:** Phase 4 of 5 from `/Users/nicholas/Projects/linear-dag/docs/design-plans/2026-02-13-cli-logging-unification.md`.

**Codebase verified:** 2026-02-13 14:11 PST

---

## Acceptance Criteria Coverage

This phase implements and tests:

### cli-logging-unification.AC2: Pipeline and CLI-relevant downstream paths use logging, not raw prints
- **cli-logging-unification.AC2.4 Edge:** CLI-relevant downstream modules with print-based progress output are migrated or gated to logger-based behavior.

---

<!-- START_SUBCOMPONENT_A (tasks 1-2) -->
<!-- START_TASK_1 -->
### Task 1: Audit and classify downstream print statements by CLI reachability

**Verifies:** cli-logging-unification.AC2.4

**Files:**
- Review: `src/linear_dag/genotype.py:232`
- Review: `src/linear_dag/core/lineararg.py:159`
- Review: `src/linear_dag/core/linear_arg_inference.py:41`
- Document: `docs/implementation-plans/2026-02-13-cli-logging-unification/phase_04.md`

**Implementation:**
- Produce an explicit reachability map for each print site:
  - directly reachable from CLI command path,
  - reachable through pipeline steps,
  - currently non-CLI (defer with rationale).
- Define whether each reachable print is migrated now or wrapped behind a logger-aware flag.

**Testing:**
- N/A (classification task).

**Verification:**
- Run: `rg -n "print\(" src/linear_dag/genotype.py src/linear_dag/core/lineararg.py src/linear_dag/core/linear_arg_inference.py`
- Expected: all print sites are cataloged with migration decision.

**Commit:** `docs: classify downstream print sites by cli reachability`
<!-- END_TASK_1 -->

<!-- START_TASK_2 -->
### Task 2: Add logger-aware pathways in downstream modules used by CLI flows

**Verifies:** cli-logging-unification.AC2.4

**Files:**
- Modify: `src/linear_dag/genotype.py:232`
- Modify: `src/linear_dag/core/lineararg.py:159`
- Modify: `src/linear_dag/core/linear_arg_inference.py:41`
- Modify: `src/linear_dag/pipeline.py:569`

**Implementation:**
- Introduce optional logger arguments (or logger-callback path) for CLI-relevant functions currently printing progress.
- Replace unconditional prints in reachable paths with logger calls.
- Preserve existing behavior for direct non-CLI usage when logger is omitted.

**Testing:**
- Add tests asserting no stdout print output for migrated CLI-relevant paths under test while log messages are captured via `caplog`.

**Verification:**
- Run: `python -m compileall src/linear_dag/genotype.py src/linear_dag/core/lineararg.py src/linear_dag/core/linear_arg_inference.py`
- Expected: all files compile cleanly.

**Commit:** `refactor: migrate cli-relevant downstream progress output to logger`
<!-- END_TASK_2 -->
<!-- END_SUBCOMPONENT_A -->

<!-- START_SUBCOMPONENT_B (tasks 3-4) -->
<!-- START_TASK_3 -->
### Task 3: Thread unified logger through newly logger-aware downstream calls

**Verifies:** cli-logging-unification.AC2.4

**Files:**
- Modify: `src/linear_dag/pipeline.py:569`
- Modify: `src/linear_dag/cli.py:789`
- Modify: `tests/test_cli.py:19`
- Modify: `tests/test_multi_step_compress.py:15`

**Implementation:**
- Ensure pipeline and CLI callsites pass unified logger to new downstream logger-aware parameters.
- Update tests for new function signatures and assert logging behavior remains coherent with quiet/verbose policies.

**Testing:**
- Run: `pytest -q tests/test_cli.py -k "compress or logger"`
- Run: `pytest -q tests/test_multi_step_compress.py -k "step"`
- Expected: tests pass with no unexpected stdout print output in migrated paths.

**Verification:**
- Run: `ruff check src/linear_dag/cli.py src/linear_dag/pipeline.py src/linear_dag/genotype.py src/linear_dag/core/lineararg.py src/linear_dag/core/linear_arg_inference.py tests/test_cli.py tests/test_multi_step_compress.py`
- Expected: no lint issues.

**Commit:** `feat: propagate unified logger into downstream cli-relevant call chains`
<!-- END_TASK_3 -->

<!-- START_TASK_4 -->
### Task 4: Request code review and finalize phase-4 commits

**Verifies:** cli-logging-unification.AC2.4

**Files:**
- Validate: `src/linear_dag/genotype.py`
- Validate: `src/linear_dag/core/lineararg.py`
- Validate: `src/linear_dag/core/linear_arg_inference.py`
- Validate: `src/linear_dag/pipeline.py`
- Validate: `src/linear_dag/cli.py`
- Validate: `tests/test_cli.py`
- Validate: `tests/test_multi_step_compress.py`

**Implementation:**
- Execute format/lint/tests on touched files.
- Run `requesting-code-review`; resolve all feedback (critical/important/minor) before committing implementation.

**Testing:**
- Run: `ruff format src/linear_dag/genotype.py src/linear_dag/core/lineararg.py src/linear_dag/core/linear_arg_inference.py src/linear_dag/pipeline.py src/linear_dag/cli.py tests/test_cli.py tests/test_multi_step_compress.py`
- Run: `ruff check src/linear_dag/genotype.py src/linear_dag/core/lineararg.py src/linear_dag/core/linear_arg_inference.py src/linear_dag/pipeline.py src/linear_dag/cli.py tests/test_cli.py tests/test_multi_step_compress.py`
- Run: `pytest -q tests/test_cli.py tests/test_multi_step_compress.py`

**Verification:**
- Expected: checks pass and code review reports zero unresolved issues.

**Commit:** `chore: finalize phase 4 downstream print audit and migration`
<!-- END_TASK_4 -->
<!-- END_SUBCOMPONENT_B -->
