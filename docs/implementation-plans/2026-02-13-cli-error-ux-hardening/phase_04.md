# CLI Error UX Hardening Implementation Plan

**Goal:** Include inferred subcommand context in runtime error output emitted by `run_cli()`.

**Architecture:** Keep `run_cli()` as the single runtime-failure boundary and add a small argv-based context helper. `_main` behavior and exit-code contract remain unchanged; only stderr error string formatting is enriched when context is inferable.

**Tech Stack:** Python 3.11+, argparse conventions, pytest.

**Scope:** Phase 4 of 4 from `/Users/nicholas/Projects/linear-dag/docs/design-plans/2026-02-13-cli-error-ux-hardening.md`.

**Codebase verified:** 2026-02-13 13:14:00 PST

---

## Acceptance Criteria Coverage

This phase implements and tests:

### cli-error-ux-hardening.AC3: Runtime CLI failures include subcommand context
- **cli-error-ux-hardening.AC3.1 Success:** `run_cli()` runtime failures print `error: <subcommand>: <message>` when a subcommand can be inferred from argv.
- **cli-error-ux-hardening.AC3.2 Success:** Existing exit-code behavior remains unchanged (`1` runtime failures, `2` parsing errors).
- **cli-error-ux-hardening.AC3.3 Edge:** If subcommand cannot be inferred, fallback remains `error: <message>`.

---

<!-- START_SUBCOMPONENT_A (tasks 1-2) -->
<!-- START_TASK_1 -->
### Task 1: Add argv-based subcommand context inference helper

**Verifies:** cli-error-ux-hardening.AC3.1, cli-error-ux-hardening.AC3.3

**Files:**
- Modify: `src/linear_dag/cli.py:1192`

**Implementation:**
- Add helper near `run_cli()` to infer primary CLI subcommand from argv tokens.
- Behavior requirements:
  - ignore global flags (`-v`, `--verbose`, `-q`, `--quiet`);
  - return first non-flag token when available;
  - return `None` when no subcommand token can be inferred.
- Keep helper deterministic and independent of parser internals.

**Testing:**
- N/A (tests added in Task 3).

**Verification:**
- Run: `python -m compileall src/linear_dag/cli.py`
- Expected: compile succeeds.

**Commit:** `feat: infer cli subcommand context from argv`
<!-- END_TASK_1 -->

<!-- START_TASK_2 -->
### Task 2: Prefix runtime stderr messages with subcommand context when present

**Verifies:** cli-error-ux-hardening.AC3.1, cli-error-ux-hardening.AC3.2, cli-error-ux-hardening.AC3.3

**Files:**
- Modify: `src/linear_dag/cli.py:1207`

**Implementation:**
- In `run_cli()` runtime exception handler (`except Exception as exc`):
  - compute context from `sys.argv`;
  - when context exists, emit `error: <context>: <message>`;
  - otherwise keep `error: <message>` fallback.
- Preserve existing return values:
  - `SystemExit` path unchanged;
  - runtime exception path returns `1`.

**Testing:**
- N/A (tests added in Task 3).

**Verification:**
- Run: `python -m compileall src/linear_dag/cli.py`
- Expected: compile succeeds.

**Commit:** `feat: add subcommand context to run_cli runtime errors`
<!-- END_TASK_2 -->
<!-- END_SUBCOMPONENT_A -->

<!-- START_SUBCOMPONENT_B (tasks 3-4) -->
<!-- START_TASK_3 -->
### Task 3: Add run_cli tests for contextual and fallback stderr output

**Verifies:** cli-error-ux-hardening.AC3.1, cli-error-ux-hardening.AC3.2, cli-error-ux-hardening.AC3.3

**Files:**
- Modify: `tests/test_cli.py:503`
- Modify: `tests/test_cli.py:512`
- Modify: `tests/test_cli.py:522`

**Implementation:**
- Add/adjust tests so runtime error stderr behavior is explicit:
  - when `sys.argv` contains subcommand (`assoc`), stderr includes `error: assoc: ...`;
  - when no subcommand is inferable, stderr falls back to `error: ...`.
- Keep existing exit-code tests for `SystemExit(2)` and runtime errors returning `1`.
- Update existing invalid-input `run_cli` tests to account for contextual prefix while preserving actionable-message assertions.

**Testing:**
- Run: `pytest -q tests/test_cli.py -k "run_cli"`
- Expected: all `run_cli` tests pass with contextual stderr formatting.

**Verification:**
- Run: `pytest -q tests/test_cli.py -k "run_cli_maps_system_exit or run_cli_runtime_error or run_cli_invalid"`
- Expected: context and fallback paths both pass, exit-code expectations unchanged.

**Commit:** `test: cover run_cli subcommand-context error output`
<!-- END_TASK_3 -->

<!-- START_TASK_4 -->
### Task 4: Final phase validation across CLI module

**Verifies:** cli-error-ux-hardening.AC3.1, cli-error-ux-hardening.AC3.2, cli-error-ux-hardening.AC3.3

**Files:**
- Validate: `src/linear_dag/cli.py`
- Validate: `tests/test_cli.py`

**Implementation:**
- Run formatting/lint on touched files.
- Execute full CLI test module to ensure end-to-end compatibility across all four phases.

**Testing:**
- Run: `ruff format src/linear_dag/cli.py tests/test_cli.py`
- Run: `ruff check src/linear_dag/cli.py tests/test_cli.py`
- Run: `pytest -q tests/test_cli.py`

**Verification:**
- Expected: all checks pass and CLI test module remains green.

**Commit:** `chore: finalize phase 4 run_cli context hardening`
<!-- END_TASK_4 -->
<!-- END_SUBCOMPONENT_B -->
