# CLI Error UX Hardening Test Requirements

Source design: `/Users/nicholas/Projects/linear-dag/docs/design-plans/2026-02-13-cli-error-ux-hardening.md`

Source implementation phases:
- `/Users/nicholas/Projects/linear-dag/docs/implementation-plans/2026-02-13-cli-error-ux-hardening/phase_01.md`
- `/Users/nicholas/Projects/linear-dag/docs/implementation-plans/2026-02-13-cli-error-ux-hardening/phase_02.md`
- `/Users/nicholas/Projects/linear-dag/docs/implementation-plans/2026-02-13-cli-error-ux-hardening/phase_03.md`
- `/Users/nicholas/Projects/linear-dag/docs/implementation-plans/2026-02-13-cli-error-ux-hardening/phase_04.md`

## Automated Test Mapping

### cli-error-ux-hardening.AC1: Invalid selection inputs provide actionable suggestions

- **cli-error-ux-hardening.AC1.1 Success:** Invalid `--block-names` errors include nearest valid block-name suggestions when close matches exist.
  - Test type: Unit + runtime contract
  - File: `tests/test_cli.py`
  - Required tests:
    - `test_load_required_block_metadata_invalid_block_name_includes_suggestion`
    - `test_run_cli_invalid_block_name_includes_suggestion_in_stderr`

- **cli-error-ux-hardening.AC1.2 Success:** Invalid `--chromosomes` errors include nearest valid chromosome suggestions when close matches exist, while preserving accepted forms such as `21` and `chr21`.
  - Test type: Unit + runtime contract
  - File: `tests/test_cli.py`
  - Required tests:
    - `test_load_required_block_metadata_invalid_chromosome_includes_suggestion`
    - `test_load_required_block_metadata_accepts_chr_prefix_for_numeric_blocks`
    - `test_run_cli_invalid_chromosome_includes_suggestion_in_stderr`

- **cli-error-ux-hardening.AC1.3 Success:** Missing column-name selection errors (for phenotype/covariate files) include nearest valid column-name suggestions when close matches exist.
  - Test type: Unit + integration + runtime contract
  - File: `tests/test_cli.py`
  - Required tests:
    - `test_read_pheno_or_covar_missing_named_column_includes_suggestion`
    - `test_prep_data_missing_covar_name_includes_suggestion`
    - `test_run_cli_missing_covar_column_includes_suggestion_in_stderr`

- **cli-error-ux-hardening.AC1.4 Failure:** When no close matches exist, errors still include requested invalid values and a bounded list of available values.
  - Test type: Unit
  - File: `tests/test_cli.py`
  - Required tests:
    - `test_load_required_block_metadata_invalid_block_name_without_close_match_still_lists_available`
    - `test_load_required_block_metadata_invalid_chromosome_without_close_match_still_lists_available`
    - `test_read_pheno_or_covar_missing_named_column_without_close_match_still_lists_available`

### cli-error-ux-hardening.AC2: Column-index selection errors fail early with explicit bounds

- **cli-error-ux-hardening.AC2.1 Success:** Out-of-range `--pheno-col-nums` values fail before full CSV load with a clear message including observed index, valid range, and column count.
  - Test type: Unit + integration + runtime contract
  - File: `tests/test_cli.py`
  - Required tests:
    - `test_read_pheno_or_covar_out_of_bounds_col_index_reports_bounds`
    - `test_prep_data_out_of_bounds_pheno_col_nums_reports_bounds`
    - `test_run_cli_out_of_bounds_pheno_col_nums_returns_bounds_error`

- **cli-error-ux-hardening.AC2.2 Success:** Out-of-range `--covar-col-nums` values fail with the same explicit bounds format.
  - Test type: Integration + runtime contract
  - File: `tests/test_cli.py`
  - Required tests:
    - `test_prep_data_out_of_bounds_covar_col_nums_reports_bounds`
    - `test_run_cli_out_of_bounds_covar_col_nums_returns_bounds_error`

- **cli-error-ux-hardening.AC2.3 Failure:** Negative indices remain rejected with an explicit validation error.
  - Test type: Unit
  - File: `tests/test_cli.py`
  - Required tests:
    - `test_read_pheno_or_covar_negative_col_index_rejected`

### cli-error-ux-hardening.AC3: Runtime CLI failures include subcommand context

- **cli-error-ux-hardening.AC3.1 Success:** `run_cli()` runtime failures print `error: <subcommand>: <message>` when a subcommand can be inferred from argv.
  - Test type: Runtime contract
  - File: `tests/test_cli.py`
  - Required tests:
    - `test_run_cli_runtime_error_prefixes_subcommand_context`
    - `test_run_cli_invalid_block_name_includes_subcommand_prefix`

- **cli-error-ux-hardening.AC3.2 Success:** Existing exit-code behavior remains unchanged (`1` runtime failures, `2` parsing errors).
  - Test type: Runtime contract
  - File: `tests/test_cli.py`
  - Required tests:
    - `test_run_cli_maps_system_exit_to_explicit_code`
    - `test_run_cli_runtime_error_returns_one_and_stderr`

- **cli-error-ux-hardening.AC3.3 Edge:** If subcommand cannot be inferred, fallback remains `error: <message>`.
  - Test type: Runtime contract
  - File: `tests/test_cli.py`
  - Required tests:
    - `test_run_cli_runtime_error_without_subcommand_falls_back_to_plain_error`

## Human Verification Mapping

No acceptance criteria require manual-only verification. All criteria are directly automatable through unit/integration/runtime contract tests in `tests/test_cli.py`.

## Execution Verification Commands

Run during implementation and before merge:

- `ruff format src/linear_dag/cli.py tests/test_cli.py`
- `ruff check src/linear_dag/cli.py tests/test_cli.py`
- `pytest -q tests/test_cli.py`
