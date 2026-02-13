# CLI Error UX Hardening Design

## Summary
This design hardens CLI error ergonomics for `linear-dag` by keeping successful execution paths unchanged and enriching only failure paths. The implementation stays localized to `src/linear_dag/cli.py` and `tests/test_cli.py`, following existing project patterns for validation (`ValueError`-based checks in helper functions) and runtime exit-code handling (`run_cli` wrapper).

The selected approach is an incremental patch, not a structural refactor: add small reusable helpers for closest-match suggestions, add explicit bounds checks for integer column selections before backend exceptions occur, and add subcommand context to runtime error output. The result is clearer feedback for invalid block/chromosome/column inputs, predictable index-bound errors for `--pheno-col-nums`/`--covar-col-nums`, and more actionable `run_cli` stderr messages (`error: <subcommand>: <message>`) while preserving existing command interfaces and exit-code guarantees.

## Definition of Done
Improve CLI validation and runtime error ergonomics for common user mistakes while preserving existing command contracts. Specifically: add nearest-match suggestions for invalid block/chromosome/column-name inputs, add explicit pre-validation for out-of-range phenotype/covariate column indices, and include subcommand context in runtime error output emitted by `run_cli()`. Add test coverage in `tests/test_cli.py` for all new behaviors.

## Acceptance Criteria
### cli-error-ux-hardening.AC1: Invalid selection inputs provide actionable suggestions
- **cli-error-ux-hardening.AC1.1 Success:** Invalid `--block-names` errors include nearest valid block-name suggestions when close matches exist.
- **cli-error-ux-hardening.AC1.2 Success:** Invalid `--chromosomes` errors include nearest valid chromosome suggestions when close matches exist, while preserving accepted forms such as `21` and `chr21`.
- **cli-error-ux-hardening.AC1.3 Success:** Missing column-name selection errors (for phenotype/covariate files) include nearest valid column-name suggestions when close matches exist.
- **cli-error-ux-hardening.AC1.4 Failure:** When no close matches exist, errors still include requested invalid values and a bounded list of available values.

### cli-error-ux-hardening.AC2: Column-index selection errors fail early with explicit bounds
- **cli-error-ux-hardening.AC2.1 Success:** Out-of-range `--pheno-col-nums` values fail before full CSV load with a clear message including observed index, valid range, and column count.
- **cli-error-ux-hardening.AC2.2 Success:** Out-of-range `--covar-col-nums` values fail with the same explicit bounds format.
- **cli-error-ux-hardening.AC2.3 Failure:** Negative indices remain rejected with an explicit validation error.

### cli-error-ux-hardening.AC3: Runtime CLI failures include subcommand context
- **cli-error-ux-hardening.AC3.1 Success:** `run_cli()` runtime failures print `error: <subcommand>: <message>` when a subcommand can be inferred from argv.
- **cli-error-ux-hardening.AC3.2 Success:** Existing exit-code behavior remains unchanged (`1` runtime failures, `2` parsing errors).
- **cli-error-ux-hardening.AC3.3 Edge:** If subcommand cannot be inferred, fallback remains `error: <message>`.

## Glossary
- **CLI**: The command-line interface exposed by `kodama`, implemented in `src/linear_dag/cli.py`.
- **Subcommand Context**: Prefixing runtime error output with the active command name (for example `assoc`) so failures are easier to trace.
- **Nearest-Match Suggestion**: A “did you mean …” hint based on similarity between invalid user input and valid known values.
- **`difflib.get_close_matches`**: Python standard-library utility used to compute close textual matches for suggestions.
- **Pre-validation**: Validating input shape/range before invoking deeper library operations, to produce clearer user-facing errors.
- **Polars**: The DataFrame library used for CLI file loading and block/column filtering.
- **`stderr`**: Standard error stream used by `run_cli()` to emit runtime failure messages.
- **Exit-Code Contract**: The explicit `run_cli()` behavior where runtime failures return `1` and parsing failures return `2`.

## Architecture
This design keeps all changes inside the existing CLI module and test suite:
- `src/linear_dag/cli.py` for validation, suggestion formatting, and runtime error shaping.
- `tests/test_cli.py` for unit/contract coverage.

The approach adds a small functional helper layer for:
- nearest-match suggestion extraction (`difflib.get_close_matches`);
- reusable value-list formatting and truncation for error output;
- index-range validation using header metadata (`n_rows=0`) before full file reads.

Data flow remains unchanged for successful paths. Only error paths are enriched:
- block/chrom filters are still resolved via `list_blocks()` and `_filter_blocks`;
- phenotype/covariate loading still goes through `_read_pheno_or_covar`;
- `run_cli()` still wraps `_main()`, but now includes command context in emitted runtime error strings.

## Existing Patterns
This design follows current patterns in `src/linear_dag/cli.py`:
- validation errors are raised as `ValueError` from helper functions;
- `_read_pheno_or_covar` centralizes phenotype/covariate file checks;
- `run_cli()` owns stderr emission and exit-code policy.

No new module is introduced. This preserves the current single-file CLI pattern and minimizes migration cost.

## Implementation Phases
<!-- START_PHASE_1 -->
### Phase 1: Suggestion Utilities for User-Facing Errors
**Goal:** Add reusable suggestion and message-format helpers.

**Components:**
- `src/linear_dag/cli.py`:
  - helper to compute closest matches for a requested token against available values;
  - helper to render “Did you mean …?” fragments with bounded length.

**Dependencies:** None.

**Done when:** Helpers exist with deterministic behavior and unit tests cover close-match and no-match cases.
<!-- END_PHASE_1 -->

<!-- START_PHASE_2 -->
### Phase 2: Selection Error Enrichment (Block/Chrom/Named Columns)
**Goal:** Attach closest-match suggestions to invalid block/chromosome/column-name errors.

**Components:**
- `src/linear_dag/cli.py`:
  - extend `_filter_blocks` invalid-block and invalid-chromosome errors;
  - extend `_read_pheno_or_covar` missing-column-name errors.
- `tests/test_cli.py`:
  - tests for suggestion-present and suggestion-absent error outputs.

**Dependencies:** Phase 1 helpers.

**Done when:** Invalid selection errors include requested values, available values, and suggestions where appropriate.
<!-- END_PHASE_2 -->

<!-- START_PHASE_3 -->
### Phase 3: Column-Index Bounds Pre-Validation
**Goal:** Validate `--pheno-col-nums` and `--covar-col-nums` bounds before Polars raises low-level errors.

**Components:**
- `src/linear_dag/cli.py`:
  - preflight header-width check for integer `columns` in `_read_pheno_or_covar`;
  - explicit error contract with observed index and valid range.
- `tests/test_cli.py`:
  - tests for out-of-range positive indices and existing negative-index rejection.

**Dependencies:** Existing `_read_pheno_or_covar` validation flow.

**Done when:** Out-of-range index failures produce explicit, user-facing messages without leaking raw backend exceptions.
<!-- END_PHASE_3 -->

<!-- START_PHASE_4 -->
### Phase 4: Subcommand-Context Runtime Errors
**Goal:** Include command context in `run_cli()` runtime error output while preserving exit-code behavior.

**Components:**
- `src/linear_dag/cli.py`:
  - infer selected subcommand from `sys.argv`;
  - prefix runtime errors as `error: <subcommand>: <message>` when available.
- `tests/test_cli.py`:
  - `run_cli` runtime-error tests with and without inferred subcommand context.

**Dependencies:** Existing `run_cli()` wrapper semantics.

**Done when:** Runtime error messages include command context where possible and all exit-code contracts remain stable.
<!-- END_PHASE_4 -->

## Additional Considerations
- Keep error-message additions concise to avoid log noise in automation contexts.
- Preserve backward-compatible exception classes (`ValueError`) for downstream callers/tests.
- Use stable substring assertions in tests to avoid brittleness from minor formatting changes.
