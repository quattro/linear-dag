# CLI Logging Unification Implementation Plan

**Goal:** Extend unified logger usage through `src/linear_dag/pipeline.py` and eliminate print-based status output from CLI-driven pipeline flows.

**Architecture:** Add optional logger injection parameters to pipeline entrypoints and key helpers. Use injected logger when provided (CLI path) and local fallback logger when omitted (library path). Replace step-level `print(...)` status messages with logger calls to unify routing and formatting.

**Tech Stack:** Python 3.11+, `logging`, pytest, h5py/polars test fixtures.

**Scope:** Phase 3 of 5 from `/Users/nicholas/Projects/linear-dag/docs/design-plans/2026-02-13-cli-logging-unification.md`.

**Codebase verified:** 2026-02-13 14:11 PST

---

## Acceptance Criteria Coverage

This phase implements and tests:

### cli-logging-unification.AC2: Pipeline and CLI-relevant downstream paths use logging, not raw prints
- **cli-logging-unification.AC2.1 Success:** `src/linear_dag/pipeline.py` step status/skip messages currently emitted via `print(...)` are emitted through logger calls.
- **cli-logging-unification.AC2.2 Success:** Pipeline entrypoints accept optional logger injection and use the injected logger when provided from CLI.
- **cli-logging-unification.AC2.3 Success:** When pipeline entrypoints are used outside CLI without a logger, fallback logging remains functional.

---

<!-- START_SUBCOMPONENT_A (tasks 1-2) -->
<!-- START_TASK_1 -->
### Task 1: Add logger-injection signatures for pipeline entrypoints and helpers

**Verifies:** cli-logging-unification.AC2.2, cli-logging-unification.AC2.3

**Files:**
- Modify: `src/linear_dag/pipeline.py:22`
- Modify: `src/linear_dag/pipeline.py:92`
- Modify: `src/linear_dag/pipeline.py:187`
- Modify: `src/linear_dag/pipeline.py:245`
- Modify: `src/linear_dag/pipeline.py:288`
- Modify: `src/linear_dag/pipeline.py:351`
- Modify: `src/linear_dag/pipeline.py:413`
- Modify: `src/linear_dag/pipeline.py:569`
- Modify: `src/linear_dag/pipeline.py:646`
- Modify: `src/linear_dag/pipeline.py:692`
- Modify: `src/linear_dag/pipeline.py:799`
- Modify: `src/linear_dag/pipeline.py:958`

**Implementation:**
- Add `logger: Optional[logging.Logger] = None` (or project-compatible type alias) to pipeline entrypoints and helper functions used by CLI step flows.
- Add one fallback helper (for example `_get_pipeline_logger(logger, log_file=None)`) to avoid repeated logger-construction logic.
- Keep default behavior equivalent for direct non-CLI callers.

**Testing:**
- Add unit-level tests that pass a stub/mock logger and assert usage path does not instantiate fallback logger.

**Verification:**
- Run: `python -m compileall src/linear_dag/pipeline.py`
- Expected: file compiles with no errors.

**Commit:** `refactor: add optional logger injection across pipeline entrypoints`
<!-- END_TASK_1 -->

<!-- START_TASK_2 -->
### Task 2: Replace pipeline print statements with structured logging

**Verifies:** cli-logging-unification.AC2.1, cli-logging-unification.AC2.2

**Files:**
- Modify: `src/linear_dag/pipeline.py:183`
- Modify: `src/linear_dag/pipeline.py:222`
- Modify: `src/linear_dag/pipeline.py:238`
- Modify: `src/linear_dag/pipeline.py:283`
- Modify: `src/linear_dag/pipeline.py:320`
- Modify: `src/linear_dag/pipeline.py:384`

**Implementation:**
- Replace existing `print(...)` skip/status lines in `msc_step0`..`msc_step4` with `logger.info(...)` or `logger.warning(...)` messages.
- Ensure skipped-path messages preserve key context fields (job id, region, reason).
- Ensure CLI quiet/verbose behavior now applies to these messages through injected logger configuration.

**Testing:**
- Add tests that assert skip messages are captured via `caplog` and not `capsys.out`.

**Verification:**
- Run: `pytest -q tests/test_multi_step_compress.py -k "step"`
- Expected: step-path tests pass and logging assertions hold.

**Commit:** `feat: migrate multi-step pipeline status output from print to logger`
<!-- END_TASK_2 -->
<!-- END_SUBCOMPONENT_A -->

<!-- START_SUBCOMPONENT_B (tasks 3-4) -->
<!-- START_TASK_3 -->
### Task 3: Wire CLI step handlers to pass invocation logger into pipeline

**Verifies:** cli-logging-unification.AC2.2, cli-logging-unification.AC2.3

**Files:**
- Modify: `src/linear_dag/cli.py:789`
- Modify: `src/linear_dag/cli.py:808`
- Modify: `src/linear_dag/pipeline.py:431`

**Implementation:**
- Update `_compress` and `_step0`..`_step5` handlers in CLI to pass invocation logger to corresponding pipeline functions.
- Remove redundant per-function `MemoryLogger` creation in pipeline where injected logger is available.
- Keep file-specific pipeline logs as optional behavior when required by step semantics.

**Testing:**
- Add integration-style tests that monkeypatch pipeline entrypoints to capture logger identity and assert it matches CLI invocation logger.

**Verification:**
- Run: `pytest -q tests/test_cli.py -k "step or compress or logger"`
- Expected: CLI step dispatch tests pass with injected logger assertions.

**Commit:** `feat: propagate cli invocation logger into pipeline step calls`
<!-- END_TASK_3 -->

<!-- START_TASK_4 -->
### Task 4: Request code review and finalize phase-3 commits

**Verifies:** cli-logging-unification.AC2.1, cli-logging-unification.AC2.2, cli-logging-unification.AC2.3

**Files:**
- Validate: `src/linear_dag/pipeline.py`
- Validate: `src/linear_dag/cli.py`
- Validate: `tests/test_multi_step_compress.py`
- Validate: `tests/test_cli.py`

**Implementation:**
- Run format/lint/tests on touched files.
- Run `requesting-code-review` and resolve all review issues before any implementation commit.

**Testing:**
- Run: `ruff format src/linear_dag/pipeline.py src/linear_dag/cli.py tests/test_multi_step_compress.py tests/test_cli.py`
- Run: `ruff check src/linear_dag/pipeline.py src/linear_dag/cli.py tests/test_multi_step_compress.py tests/test_cli.py`
- Run: `pytest -q tests/test_multi_step_compress.py`
- Run: `pytest -q tests/test_cli.py -k "step or compress or logger"`

**Verification:**
- Expected: checks pass and code review reports zero unresolved issues.

**Commit:** `chore: finalize phase 3 pipeline logger injection and print removal`
<!-- END_TASK_4 -->
<!-- END_SUBCOMPONENT_B -->
