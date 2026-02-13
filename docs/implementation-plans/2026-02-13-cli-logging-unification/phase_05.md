# CLI Logging Unification Implementation Plan

**Goal:** Verify unified logging behavior end-to-end, preserve CLI contracts, and finalize rollout with explicit review gates.

**Architecture:** Consolidate verification across CLI, pipeline, and downstream touched modules; ensure test and documentation updates align with the logging contract. This phase is the guardrail phase that confirms behavior and quality requirements before merge.

**Tech Stack:** Python 3.11+, pytest, ruff.

**Scope:** Phase 5 of 5 from `/Users/nicholas/Projects/linear-dag/docs/design-plans/2026-02-13-cli-logging-unification.md`.

**Codebase verified:** 2026-02-13 14:11 PST

---

## Acceptance Criteria Coverage

This phase implements and tests:

### cli-logging-unification.AC1: One unified logger is used across each CLI invocation
- **cli-logging-unification.AC1.3 Success:** Log routing/formatting (stdout/file and memory annotation) is consistent across `assoc`, `rhe`, `score`, `compress`, and `multi-step-compress` paths.
- **cli-logging-unification.AC1.4 Failure/Guardrail:** Repeated CLI invocations do not accumulate duplicate handlers.

### cli-logging-unification.AC2: Pipeline and CLI-relevant downstream paths use logging, not raw prints
- **cli-logging-unification.AC2.1 Success:** `src/linear_dag/pipeline.py` step status/skip messages currently emitted via `print(...)` are emitted through logger calls.
- **cli-logging-unification.AC2.2 Success:** Pipeline entrypoints accept optional logger injection and use the injected logger when provided from CLI.
- **cli-logging-unification.AC2.3 Success:** When pipeline entrypoints are used outside CLI without a logger, fallback logging remains functional.
- **cli-logging-unification.AC2.4 Edge:** CLI-relevant downstream modules with print-based progress output are migrated or gated to logger-based behavior.

### cli-logging-unification.AC3: Existing CLI contracts remain stable
- **cli-logging-unification.AC3.1 Success:** `run_cli()` exit-code behavior remains unchanged (`0` success, `1` runtime failures, `2` parse errors).
- **cli-logging-unification.AC3.2 Success:** Runtime error stderr contract remains unchanged in shape/content policy (including existing subcommand-context behavior).
- **cli-logging-unification.AC3.3 Success:** Existing subcommands/flags and output artifact generation behavior remain backward compatible.

---

<!-- START_SUBCOMPONENT_A (tasks 1-2) -->
<!-- START_TASK_1 -->
### Task 1: Finalize docstrings and interface documentation for logger injection

**Verifies:** cli-logging-unification.AC2.2, cli-logging-unification.AC2.3

**Files:**
- Modify: `src/linear_dag/pipeline.py:22`
- Modify: `src/linear_dag/cli.py:372`
- Modify: `src/linear_dag/memory_logger.py:9`

**Implementation:**
- Update docstrings/signatures to document optional logger parameters and fallback semantics.
- Ensure documentation style matches project rules (markdown sections, concise behavioral constraints).

**Testing:**
- N/A (documentation/interface alignment).

**Verification:**
- Run: `python -m compileall src/linear_dag/cli.py src/linear_dag/pipeline.py src/linear_dag/memory_logger.py`
- Expected: files compile with no syntax errors.

**Commit:** `docs: document unified logger injection and fallback behavior`
<!-- END_TASK_1 -->

<!-- START_TASK_2 -->
### Task 2: Execute comprehensive logging-regression test suite

**Verifies:** cli-logging-unification.AC1.3, cli-logging-unification.AC1.4, cli-logging-unification.AC2.1, cli-logging-unification.AC2.2, cli-logging-unification.AC2.3, cli-logging-unification.AC2.4, cli-logging-unification.AC3.1, cli-logging-unification.AC3.2, cli-logging-unification.AC3.3

**Files:**
- Validate: `tests/test_cli.py`
- Validate: `tests/test_multi_step_compress.py`
- Validate: any new logging-focused tests introduced in prior phases

**Implementation:**
- Run targeted tests for CLI contracts and logging behavior.
- Run full touched-module checks to capture regressions from signature threading.

**Testing:**
- Run: `pytest -q tests/test_cli.py`
- Run: `pytest -q tests/test_multi_step_compress.py`

**Verification:**
- Expected: all logging and CLI contract tests pass.

**Commit:** `test: validate end-to-end logging unification behavior`
<!-- END_TASK_2 -->
<!-- END_SUBCOMPONENT_A -->

<!-- START_SUBCOMPONENT_B (tasks 3-4) -->
<!-- START_TASK_3 -->
### Task 3: Run requesting-code-review workflow and close all findings

**Verifies:** cli-logging-unification.AC1.4, cli-logging-unification.AC3.3

**Files:**
- Validate: all files touched in phases 1-5

**Implementation:**
- Run `requesting-code-review` over the full implementation diff.
- Fix all Critical/Important/Minor findings.
- Re-run review until zero unresolved issues remain.

**Testing:**
- Re-run tests/lint affected by each fix cycle.

**Verification:**
- Expected: code review loop reaches zero unresolved issues.

**Commit:** `chore: resolve code review findings for logging unification rollout`
<!-- END_TASK_3 -->

<!-- START_TASK_4 -->
### Task 4: Final release-grade verification and integration commit

**Verifies:** cli-logging-unification.AC1.3, cli-logging-unification.AC1.4, cli-logging-unification.AC2.1, cli-logging-unification.AC2.2, cli-logging-unification.AC2.3, cli-logging-unification.AC2.4, cli-logging-unification.AC3.1, cli-logging-unification.AC3.2, cli-logging-unification.AC3.3

**Files:**
- Validate: `src/linear_dag/memory_logger.py`
- Validate: `src/linear_dag/cli.py`
- Validate: `src/linear_dag/pipeline.py`
- Validate: `src/linear_dag/genotype.py`
- Validate: `src/linear_dag/core/lineararg.py`
- Validate: `src/linear_dag/core/linear_arg_inference.py`
- Validate: relevant tests under `tests/`

**Implementation:**
- Execute final formatting/lint/test commands.
- Ensure no unresolved review findings remain.
- Create final integration commit for the logging unification feature.

**Testing:**
- Run: `ruff format src/linear_dag/memory_logger.py src/linear_dag/cli.py src/linear_dag/pipeline.py src/linear_dag/genotype.py src/linear_dag/core/lineararg.py src/linear_dag/core/linear_arg_inference.py tests/test_cli.py tests/test_multi_step_compress.py`
- Run: `ruff check src/linear_dag/memory_logger.py src/linear_dag/cli.py src/linear_dag/pipeline.py src/linear_dag/genotype.py src/linear_dag/core/lineararg.py src/linear_dag/core/linear_arg_inference.py tests/test_cli.py tests/test_multi_step_compress.py`
- Run: `pytest -q tests/test_cli.py tests/test_multi_step_compress.py`

**Verification:**
- Expected: all checks pass and integration commit is ready for merge.

**Commit:** `feat: unify cli logging across pipeline and downstream flows`
<!-- END_TASK_4 -->
<!-- END_SUBCOMPONENT_B -->
