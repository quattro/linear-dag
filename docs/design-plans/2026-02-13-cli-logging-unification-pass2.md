# CLI Logging Unification Pass 2 Design

## Summary
This pass tightens the logging model introduced in the previous refactor: CLI command execution will use a single memory-aware logger context and pass a plain `logging.Logger` through command handlers without logger coercion. We remove `verbosity`-driven internal logger creation from core inference entry points and switch to optional logger injection, so progress messages are emitted only when a logger is explicitly provided by the caller.

The scope is intentionally narrow: simplify CLI logger assumptions, replace `verbosity` parameters with `logger` parameters in the affected core functions, and keep print removal behavior from earlier work. This keeps user-facing behavior stable while reducing API ambiguity and eliminating mixed logging control paths.

## Definition of Done
Simplify logging in `src/linear_dag/cli.py` so command handlers use one shared logger flow without `MemoryLogger`/`logging.Logger` coercion helpers. Replace function-level `verbosity` arguments with optional logger arguments for functions that currently expose verbosity controls, and emit progress logs only when logger is provided. Preserve existing CLI behavior and update tests to verify the new logger-based APIs.

## Acceptance Criteria
### cli-logging-unification-pass2.AC1: CLI logger flow is simplified and consistent
- **cli-logging-unification-pass2.AC1.1 Success:** `src/linear_dag/cli.py` no longer uses logger coercion helpers for command-level logging flow.
- **cli-logging-unification-pass2.AC1.2 Success:** `_main()` continues to dispatch subcommands as `args.func(args, logger)` with a single shared logger instance per invocation.
- **cli-logging-unification-pass2.AC1.3 Success:** CLI logger setup still supports stdout and optional disk log handlers from the same logger instance.
- **cli-logging-unification-pass2.AC1.4 Failure/Guardrail:** Repeated CLI invocations do not leak managed handlers.

### cli-logging-unification-pass2.AC2: Verbosity arguments are replaced with optional logger arguments
- **cli-logging-unification-pass2.AC2.1 Success:** `LinearARG.from_genotypes` and `LinearARG.from_vcf` remove `verbosity` and accept `logger: Optional[logging.Logger]`.
- **cli-logging-unification-pass2.AC2.2 Success:** `linear_arg_from_genotypes` removes `verbosity` and accepts `logger: Optional[logging.Logger]`.
- **cli-logging-unification-pass2.AC2.3 Success:** Progress logs in these functions emit only when logger is not `None`.
- **cli-logging-unification-pass2.AC2.4 Guardrail:** Existing call sites remain functional with default `logger=None`.

### cli-logging-unification-pass2.AC3: Tests enforce behavior and API compatibility
- **cli-logging-unification-pass2.AC3.1 Success:** CLI tests validate simplified logger wiring and no handler accumulation.
- **cli-logging-unification-pass2.AC3.2 Success:** Pipeline/core logging tests validate that replaced APIs use logger-based progress and avoid stdout prints.
- **cli-logging-unification-pass2.AC3.3 Success:** Targeted tests pass for all touched modules.

## Glossary
- **Invocation Logger:** The single logger configured for one CLI run and threaded through command handlers.
- **Logger Coercion:** Converting `MemoryLogger` wrappers or `None` into concrete loggers inside helper functions.
- **Optional Logger API:** Function signature pattern using `logger: Optional[logging.Logger] = None`.
- **Managed Handlers:** Handlers created by CLI setup and tagged for deterministic cleanup.
- **Progress Emission Gate:** Rule that progress logging occurs only when `logger is not None`.

## Architecture
The CLI remains the logging owner. `src/linear_dag/cli.py` configures one memory-annotated logger with managed handlers, then dispatches each command with `args.func(args, logger)`. Internal helper signatures in CLI that currently accept union logger types are narrowed to `logging.Logger` (or `Optional[logging.Logger]` where helper usage benefits from optionality) and no longer normalize via coercion.

Core inference entry points in `src/linear_dag/core/lineararg.py` and `src/linear_dag/core/linear_arg_inference.py` move from `verbosity` checks to optional logger gating. A local closure will emit `logger.info(...)` only when logger exists, preserving silent behavior for library calls that do not inject logging while allowing CLI and pipeline paths to inherit global verbosity via logger level.

## Existing Patterns
Existing code already routes most operational logs through injected logger arguments in `src/linear_dag/pipeline.py` and command handlers in `src/linear_dag/cli.py`. The remaining inconsistency is API-level: verbosity flags in `lineararg`/inference and union-type coercion helpers in CLI. This pass aligns with the established injected-logger pattern and removes mixed control mechanisms.

## Implementation Phases
<!-- START_PHASE_1 -->
### Phase 1: CLI Logger Contract Cleanup
**Goal:** Remove CLI logger coercion from command/helper flow while preserving managed handler lifecycle.

**Components:**
- `src/linear_dag/cli.py` logger helper/type updates for `_prep_data`, `_load_required_block_metadata`, `_attach_variant_info`, and module-level logger utilities.
- `tests/test_cli.py` updates for the revised logger contract.

**Dependencies:** None.

**Done when:** CLI code path no longer depends on logger coercion and existing handler cleanup tests still pass.
<!-- END_PHASE_1 -->

<!-- START_PHASE_2 -->
### Phase 2: Replace Verbosity with Optional Logger in Core Entry Points
**Goal:** Convert verbosity-driven progress in core constructors/inference to logger-based optional emission.

**Components:**
- `src/linear_dag/core/lineararg.py` (`from_genotypes`, `from_vcf`) signature and progress-log updates.
- `src/linear_dag/core/linear_arg_inference.py` (`linear_arg_from_genotypes`) signature and progress-log updates.
- Call-site propagation in `src/linear_dag/pipeline.py` or other callers where logger is available.

**Dependencies:** Phase 1.

**Done when:** Core APIs use `Optional[logging.Logger]`, emit progress only with logger, and compile with updated call signatures.
<!-- END_PHASE_2 -->

<!-- START_PHASE_3 -->
### Phase 3: Test-First Coverage and Regression Guardrails
**Goal:** Add/adjust tests to lock in simplified logger behavior and new API contracts.

**Components:**
- `tests/test_cli.py` for CLI logger wiring and managed-handler cleanup.
- `tests/test_pipeline_logging.py` for logger-based progress behavior in core inference paths.

**Dependencies:** Phases 1-2.

**Done when:** New/updated tests fail before code change, pass after change, and verify no stdout print regression in touched paths.
<!-- END_PHASE_3 -->

<!-- START_PHASE_4 -->
### Phase 4: Review and Verification
**Goal:** Ensure quality gates before commit.

**Components:**
- Targeted test and lint/format checks for touched files.
- Mandatory `requesting-code-review` loop with zero unresolved findings.

**Dependencies:** Phases 1-3.

**Done when:** Verification commands pass and review cycle reports no unresolved issues.
<!-- END_PHASE_4 -->

## Additional Considerations
No child logger hierarchy is introduced in this pass; a unified logger instance is intentionally preferred. This pass preserves fallback behavior for non-CLI callers by keeping optional logger defaults rather than introducing global singleton state.
