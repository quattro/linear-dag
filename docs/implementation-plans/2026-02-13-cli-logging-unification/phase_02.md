# CLI Logging Unification Implementation Plan

**Goal:** Route CLI command handler logging through a single invocation logger and remove per-handler logger construction.

**Architecture:** Thread one logger from `_main()` into command handlers (`_assoc_scan`, `_estimate_h2g`, `_prs`, `_compress`, `_step0`..`_step5`) and helper calls requiring logging. Preserve command behavior, output artifacts, and runtime error contracts while unifying logger ownership.

**Tech Stack:** Python 3.11+, `logging`, pytest.

**Scope:** Phase 2 of 5 from `/Users/nicholas/Projects/linear-dag/docs/design-plans/2026-02-13-cli-logging-unification.md`.

**Codebase verified:** 2026-02-13 14:11 PST

---

## Acceptance Criteria Coverage

This phase implements and tests:

### cli-logging-unification.AC1: One unified logger is used across each CLI invocation
- **cli-logging-unification.AC1.2 Success:** CLI command handlers use the invocation logger instead of creating independent logger instances.
- **cli-logging-unification.AC1.3 Success:** Log routing/formatting (stdout/file and memory annotation) is consistent across `assoc`, `rhe`, `score`, `compress`, and `multi-step-compress` paths.

### cli-logging-unification.AC3: Existing CLI contracts remain stable
- **cli-logging-unification.AC3.1 Success:** `run_cli()` exit-code behavior remains unchanged (`0` success, `1` runtime failures, `2` parse errors).
- **cli-logging-unification.AC3.2 Success:** Runtime error stderr contract remains unchanged in shape/content policy (including existing subcommand-context behavior).
- **cli-logging-unification.AC3.3 Success:** Existing subcommands/flags and output artifact generation behavior remain backward compatible.

---

<!-- START_SUBCOMPONENT_A (tasks 1-2) -->
<!-- START_TASK_1 -->
### Task 1: Refactor CLI handler signatures to consume injected logger

**Verifies:** cli-logging-unification.AC1.2, cli-logging-unification.AC1.3

**Files:**
- Modify: `src/linear_dag/cli.py:372`
- Modify: `src/linear_dag/cli.py:396`
- Modify: `src/linear_dag/cli.py:511`
- Modify: `src/linear_dag/cli.py:789`
- Modify: `src/linear_dag/cli.py:808`

**Implementation:**
- Update command handler functions to accept a logger argument (standard `logging.Logger` interface).
- Remove direct `MemoryLogger(__name__)` construction from handlers.
- Ensure helper paths (`_prep_data`, `_load_required_block_metadata`, `_attach_variant_info`) accept and use injected logger types consistently.

**Testing:**
- Adjust handler-level tests to pass explicit logger where required.

**Verification:**
- Run: `python -m compileall src/linear_dag/cli.py`
- Expected: file compiles with no errors.

**Commit:** `refactor: inject unified logger into cli command handlers`
<!-- END_TASK_1 -->

<!-- START_TASK_2 -->
### Task 2: Wire unified logger through `_main()` dispatch path

**Verifies:** cli-logging-unification.AC1.2, cli-logging-unification.AC1.3, cli-logging-unification.AC3.3

**Files:**
- Modify: `src/linear_dag/cli.py:1115`

**Implementation:**
- In `_main()`, ensure selected command function receives unified logger (for example via `args.logger = log` or a wrapper dispatch helper).
- Keep parser contract and subcommand registration unchanged.
- Preserve existing masthead/log preamble behavior.

**Testing:**
- Add tests ensuring each CLI subcommand path receives the same logger object instance.

**Verification:**
- Run: `pytest -q tests/test_cli.py -k "assoc or score or rhe or run_cli"`
- Expected: relevant command-path tests pass with unchanged outputs.

**Commit:** `feat: dispatch cli subcommands with shared invocation logger`
<!-- END_TASK_2 -->
<!-- END_SUBCOMPONENT_A -->

<!-- START_SUBCOMPONENT_B (tasks 3-4) -->
<!-- START_TASK_3 -->
### Task 3: Add regression tests for contract stability during logger refactor

**Verifies:** cli-logging-unification.AC3.1, cli-logging-unification.AC3.2, cli-logging-unification.AC3.3

**Files:**
- Modify: `tests/test_cli.py:671`

**Implementation:**
- Extend existing `run_cli` and invalid-input stderr tests to ensure logger refactor does not change:
  - exit-code behavior,
  - stderr runtime-error formatting,
  - expected output file creation in smoke tests.
- Add targeted assertions that command handlers no longer instantiate independent logger objects.

**Testing:**
- Run: `pytest -q tests/test_cli.py -k "run_cli or invalid_ or smoke"`
- Expected: CLI contract tests remain green.

**Verification:**
- Run: `ruff check src/linear_dag/cli.py tests/test_cli.py`
- Expected: no lint violations.

**Commit:** `test: lock cli runtime contracts during logger unification`
<!-- END_TASK_3 -->

<!-- START_TASK_4 -->
### Task 4: Request code review and finalize phase-2 commits

**Verifies:** cli-logging-unification.AC1.2, cli-logging-unification.AC1.3, cli-logging-unification.AC3.1, cli-logging-unification.AC3.2, cli-logging-unification.AC3.3

**Files:**
- Validate: `src/linear_dag/cli.py`
- Validate: `tests/test_cli.py`

**Implementation:**
- Run formatter/lint/targeted tests for touched files.
- Run `requesting-code-review` and resolve all findings (including minor) before committing implementation.

**Testing:**
- Run: `ruff format src/linear_dag/cli.py tests/test_cli.py`
- Run: `ruff check src/linear_dag/cli.py tests/test_cli.py`
- Run: `pytest -q tests/test_cli.py -k "run_cli or smoke or invalid"`

**Verification:**
- Expected: checks pass and code review reports zero unresolved issues.

**Commit:** `chore: finalize phase 2 cli handler logging unification`
<!-- END_TASK_4 -->
<!-- END_SUBCOMPONENT_B -->
