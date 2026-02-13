# CLI Error UX Hardening Implementation Plan

**Goal:** Add reusable closest-match and suggestion-format helpers for CLI error messages.

**Architecture:** Extend the functional helper layer at the top of `src/linear_dag/cli.py` without changing existing command behavior yet. Keep suggestion computation deterministic and bounded so later validation paths can consume it consistently.

**Tech Stack:** Python 3.11+, standard library (`difflib`), pytest.

**Scope:** Phase 1 of 4 from `/Users/nicholas/Projects/linear-dag/docs/design-plans/2026-02-13-cli-error-ux-hardening.md`.

**Codebase verified:** 2026-02-13 13:14:00 PST

---

## Acceptance Criteria Coverage

This phase implements and tests:

### cli-error-ux-hardening.AC1: Invalid selection inputs provide actionable suggestions
- **cli-error-ux-hardening.AC1.4 Failure:** When no close matches exist, errors still include requested invalid values and a bounded list of available values.

Note: This phase delivers helper behavior used by user-facing validation paths. Full end-to-end AC1 wiring is completed in Phase 2.

---

<!-- START_SUBCOMPONENT_A (tasks 1-2) -->
<!-- START_TASK_1 -->
### Task 1: Add closest-match and suggestion-format helper functions

**Verifies:** cli-error-ux-hardening.AC1.4 (helper-level behavior)

**Files:**
- Modify: `src/linear_dag/cli.py:1`
- Modify: `src/linear_dag/cli.py:84`

**Implementation:**
- Add `import difflib` near existing stdlib imports.
- Add helper `_closest_matches(requested: str, available: list[str], *, limit: int = 3, cutoff: float = 0.75) -> list[str]`:
  - deduplicate and stringify available values;
  - return up to `limit` close matches from `difflib.get_close_matches`.
- Add helper `_format_suggestion_fragment(requested: str, available: list[str]) -> str`:
  - return empty string when no close match exists;
  - otherwise return a short fragment like `Did you mean: a, b?`.
- Keep helpers pure (no logging, no I/O).

**Testing:**
- N/A (implementation task; tests added in Task 2).

**Verification:**
- Run: `python -m compileall src/linear_dag/cli.py`
- Expected: compile succeeds.

**Commit:** `feat: add cli suggestion helper primitives`
<!-- END_TASK_1 -->

<!-- START_TASK_2 -->
### Task 2: Add focused helper tests for match/no-match behavior

**Verifies:** cli-error-ux-hardening.AC1.4 (helper-level behavior)

**Files:**
- Modify: `tests/test_cli.py:180`

**Implementation:**
- Add tests for `_closest_matches`:
  - typo case returns at least one close candidate;
  - unrelated token returns empty list;
  - returned list length is bounded.
- Add tests for `_format_suggestion_fragment`:
  - includes `Did you mean` when close matches exist;
  - returns `""` when no matches exist.
- Keep assertions substring-based to avoid brittle formatting coupling.

**Testing:**
- Run: `pytest -q tests/test_cli.py -k "closest_matches or suggestion_fragment"`
- Expected: new tests pass.

**Verification:**
- Run: `pytest -q tests/test_cli.py -k "closest_matches or suggestion_fragment"`
- Expected: all selected tests pass and no unrelated failures.

**Commit:** `test: cover cli suggestion helper behavior`
<!-- END_TASK_2 -->
<!-- END_SUBCOMPONENT_A -->

<!-- START_TASK_3 -->
### Task 3: Validate formatting/lint for phase changes

**Verifies:** cli-error-ux-hardening.AC1.4 (regression-safe helper behavior)

**Files:**
- Validate: `src/linear_dag/cli.py`
- Validate: `tests/test_cli.py`

**Implementation:**
- Run formatter/lint checks used by repository hooks.
- Ensure only helper-related code paths changed in this phase.

**Testing:**
- Run: `ruff format src/linear_dag/cli.py tests/test_cli.py`
- Run: `ruff check src/linear_dag/cli.py tests/test_cli.py`

**Verification:**
- Expected: format/check pass with no new violations.

**Commit:** `chore: validate phase 1 cli suggestion helper changes`
<!-- END_TASK_3 -->
