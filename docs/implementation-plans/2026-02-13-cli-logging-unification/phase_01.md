# CLI Logging Unification Implementation Plan

**Goal:** Establish a unified logger contract that can be configured once by CLI and injected through downstream code while preserving `MemoryLogger` compatibility.

**Architecture:** Add a shared logging configuration path in `src/linear_dag/memory_logger.py` that produces standard `logging.Logger` instances with memory annotation and deduplicated handlers. Wire `src/linear_dag/cli.py` to use this contract as the single logger bootstrap point for command execution.

**Tech Stack:** Python 3.11+, `logging`, `psutil`, pytest.

**Scope:** Phase 1 of 5 from `/Users/nicholas/Projects/linear-dag/docs/design-plans/2026-02-13-cli-logging-unification.md`.

**Codebase verified:** 2026-02-13 14:11 PST

---

## Acceptance Criteria Coverage

This phase implements and tests:

### cli-logging-unification.AC1: One unified logger is used across each CLI invocation
- **cli-logging-unification.AC1.1 Success:** `_main()` configures exactly one invocation logger based on `--verbose`, `--quiet`, and `--out`.
- **cli-logging-unification.AC1.4 Failure/Guardrail:** Repeated CLI invocations do not accumulate duplicate handlers.

---

<!-- START_SUBCOMPONENT_A (tasks 1-2) -->
<!-- START_TASK_1 -->
### Task 1: Introduce memory-aware logger configuration primitives

**Verifies:** cli-logging-unification.AC1.1

**Files:**
- Modify: `src/linear_dag/memory_logger.py:1`

**Implementation:**
- Add a reusable record filter (or adapter) that ensures `memory_usage` exists on every emitted log record.
- Add a logger factory/configuration helper (for example `configure_logger(...)`) that:
  - accepts logger name, level, optional stream handler, optional file handler path;
  - attaches formatter compatible with current CLI log shape;
  - marks handlers with an internal sentinel for safe cleanup.
- Keep `MemoryLogger` as a compatibility wrapper around configured `logging.Logger` usage; do not remove existing API in this phase.

**Testing:**
- Add/extend tests that verify configured log records include `memory_usage` and expected formatting fields.

**Verification:**
- Run: `python -m compileall src/linear_dag/memory_logger.py`
- Expected: file compiles with no errors.

**Commit:** `feat: add unified logger configuration primitives with memory annotation`
<!-- END_TASK_1 -->

<!-- START_TASK_2 -->
### Task 2: Add CLI logger bootstrap helper and lifecycle wiring

**Verifies:** cli-logging-unification.AC1.1, cli-logging-unification.AC1.4

**Files:**
- Modify: `src/linear_dag/cli.py:1130`
- Modify: `src/linear_dag/cli.py:1290`

**Implementation:**
- Extract logger setup from `_main()` into a dedicated helper (for example `_build_cli_logger_context(...)`) that returns:
  - configured unified logger,
  - handler references/managed streams for deterministic teardown.
- Preserve existing user-facing command preamble writes (`masthead`, command string, `Starting log...`).
- Ensure `_remove_cli_handlers` (or replacement) removes only CLI-managed handlers and avoids leakage across repeated `_main()` calls.

**Testing:**
- Add tests in `tests/test_cli.py` that call `_main()` in controlled monkeypatched mode across repeated invocations and assert handler count does not grow.

**Verification:**
- Run: `python -m compileall src/linear_dag/cli.py`
- Expected: file compiles with no errors.

**Commit:** `feat: bootstrap unified cli logger context with deterministic teardown`
<!-- END_TASK_2 -->
<!-- END_SUBCOMPONENT_A -->

<!-- START_SUBCOMPONENT_B (tasks 3-4) -->
<!-- START_TASK_3 -->
### Task 3: Add focused tests for unified bootstrap behavior

**Verifies:** cli-logging-unification.AC1.1, cli-logging-unification.AC1.4

**Files:**
- Modify: `tests/test_cli.py:671`
- Create: `tests/test_memory_logger.py`

**Implementation:**
- Add unit tests for logger bootstrap helpers and teardown idempotency.
- Add tests confirming repeated CLI invocation does not duplicate handlers.
- Keep tests isolated with monkeypatches around `_main` command execution to avoid heavy data operations.

**Testing:**
- Run: `pytest -q tests/test_cli.py -k "logger or handler or run_cli"`
- Run: `pytest -q tests/test_memory_logger.py`
- Expected: targeted logging bootstrap tests pass.

**Verification:**
- Run: `ruff check src/linear_dag/memory_logger.py src/linear_dag/cli.py tests/test_cli.py tests/test_memory_logger.py`
- Expected: no lint violations.

**Commit:** `test: cover unified cli logger bootstrap and handler leak prevention`
<!-- END_TASK_3 -->

<!-- START_TASK_4 -->
### Task 4: Request code review and finalize phase-1 commits

**Verifies:** cli-logging-unification.AC1.1, cli-logging-unification.AC1.4

**Files:**
- Validate: `src/linear_dag/memory_logger.py`
- Validate: `src/linear_dag/cli.py`
- Validate: `tests/test_cli.py`
- Validate: `tests/test_memory_logger.py`

**Implementation:**
- Run final format/lint/tests for touched files.
- Run the `requesting-code-review` workflow and fix all Critical/Important/Minor issues before committing implementation changes.
- Only commit after review returns zero unresolved issues.

**Testing:**
- Run: `ruff format src/linear_dag/memory_logger.py src/linear_dag/cli.py tests/test_cli.py tests/test_memory_logger.py`
- Run: `ruff check src/linear_dag/memory_logger.py src/linear_dag/cli.py tests/test_cli.py tests/test_memory_logger.py`
- Run: `pytest -q tests/test_cli.py -k "logger or run_cli"`
- Run: `pytest -q tests/test_memory_logger.py`

**Verification:**
- Expected: all checks pass and review reports zero unresolved issues.

**Commit:** `chore: finalize phase 1 unified logger infrastructure`
<!-- END_TASK_4 -->
<!-- END_SUBCOMPONENT_B -->
