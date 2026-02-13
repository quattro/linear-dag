# CLI Invalid-Input Unit Tests Design

## Summary
Add focused CLI unit tests for invalid selection and column inputs in `assoc`/`rhe`/`score` workflows. The plan targets three concrete user-facing failure classes: invalid chromosome filters, invalid block-name filters, and missing covariate columns. Tests will lock current behavior first, including one currently low-level error path (`Polars InvalidOperationError`), so follow-up validation improvements can be made safely.

## Definition of Done
Produce an investigation-backed test proposal for CLI invalid-input handling in `src/linear_dag/cli.py`, specifically covering invalid chromosome filters, invalid block-name filters, and missing covariate columns. The proposal must document current observed behavior, identify coverage gaps in existing `tests/test_cli.py`, and define concrete unit tests (names, setup, assertions) that can be implemented without ambiguity. No production code changes are included in this phase.

## Acceptance Criteria
- `cli-invalid-input-tests.AC1.1`: There is explicit test coverage that an invalid block name selection fails deterministically in CLI code paths and does not silently proceed.
- `cli-invalid-input-tests.AC1.2`: There is explicit test coverage that invalid chromosome selection is handled as an error; current observed behavior (`InvalidOperationError` for string/int mismatch) is captured in tests.
- `cli-invalid-input-tests.AC1.3`: There is explicit test coverage that missing covariate column names fail with a clear exception path.
- `cli-invalid-input-tests.AC2.1`: There is test coverage for the public CLI exit-code contract (`run_cli`) showing runtime failures return code `1` with an error message on stderr for each invalid-input class.
- `cli-invalid-input-tests.AC2.2`: Existing smoke tests and parser help tests remain untouched and still describe successful paths.
- `cli-invalid-input-tests.AC3.1`: Proposed tests include deterministic fixture usage and assertion targets (exception type/message or exit code/stderr), so implementation does not require additional design work.

## Glossary
- `block metadata`: Per-block DataFrame loaded from HDF5 by `list_blocks`, used for block/chromosome filtering before expensive operator setup.
- `invalid chromosome`: A chromosome filter value that cannot be applied to available block metadata (nonexistent value or incompatible type).
- `invalid block name`: A block-name filter not present in block metadata.
- `missing covariate column`: A `--covar-name` selection not present in the provided covariate file.
- `runtime CLI failure`: A failure occurring after argument parsing, surfaced by `run_cli()` as exit code `1`.

## Context
Current CLI selection and data-loading behavior is implemented in:
- `src/linear_dag/cli.py:517` (`_filter_blocks`)
- `src/linear_dag/cli.py:578` (`_load_required_block_metadata`)
- `src/linear_dag/cli.py:645` (`_require_block_metadata`)
- `src/linear_dag/cli.py:228` (`_read_pheno_or_covar`)
- `src/linear_dag/cli.py:1117` (`run_cli`)

Existing coverage in `tests/test_cli.py` is strong for smoke and helper behavior but does not currently include direct invalid-input tests for:
- invalid `--chromosomes` filtering behavior
- invalid `--block-names` filtering behavior through selection helpers
- missing columns passed via `--covar-name`
- real `run_cli` contract checks for these concrete runtime failures

## Investigated Current Behavior
Using `tests/testdata/test_chr21_50.h5` and `tests/testdata/phenotypes_50.tsv`:
- Invalid chromosome (`--chromosomes not_a_chrom`) currently raises `polars.exceptions.InvalidOperationError` because `_filter_blocks` compares `List(String)` against `chrom: Int64`.
- Invalid block name (`--block-names missing_block`) currently yields empty metadata and then `ValueError` from `_require_block_metadata` with "No block metadata found ...".
- Missing covariate column (`--covar-name iid,not_a_col`) currently raises `polars.exceptions.ColumnNotFoundError` from `pl.read_csv` in `_read_pheno_or_covar`.

## Design Options Considered
Option A: Only helper-level unit tests (`_filter_blocks`, `_read_pheno_or_covar`, `_load_required_block_metadata`).
- Pros: Fast, deterministic, tight failure localization.
- Cons: Does not verify public exit-code/stderr contract.

Option B: Only full CLI `_main` integration tests.
- Pros: Very realistic behavior.
- Cons: Heavy setup and less precise attribution for failure origin.

Option C: Hybrid (selected): helper-level unit tests plus minimal `run_cli` contract tests.
- Pros: Catches logic regressions early and also protects user-facing behavior.
- Cons: Slightly more tests than either extreme, but still small and maintainable.

## Proposed Tests
Primary target file: `tests/test_cli.py`.

1. `test_filter_blocks_rejects_block_names_and_chromosomes_together`
- Setup: call `_filter_blocks` with both `block_names` and `chromosomes`.
- Assert: raises `ValueError` matching "Specify either block_names or chromosomes".

2. `test_load_required_block_metadata_invalid_block_name_raises`
- Setup: use real block fixture path and pass nonexistent `block_names`.
- Assert: raises `ValueError` matching "No block metadata found".

3. `test_load_required_block_metadata_chromosome_string_vs_int_raises_invalid_operation`
- Setup: use real block fixture where `chrom` dtype is `Int64` and pass string chromosome list as CLI does.
- Assert: raises `polars.exceptions.InvalidOperationError` with `is_in` type-mismatch text.
- Note: this locks current behavior; if validation is improved later, this test should be updated alongside code.

4. `test_read_pheno_or_covar_missing_named_column_raises_column_not_found`
- Setup: call `_read_pheno_or_covar` with columns including missing name.
- Assert: raises `polars.exceptions.ColumnNotFoundError`.

5. `test_prep_data_with_missing_covar_name_bubbles_column_not_found`
- Setup: call `_prep_data` with valid `linarg_path/pheno` and invalid `covar_names`.
- Assert: raises `polars.exceptions.ColumnNotFoundError`.

6. `test_run_cli_invalid_block_name_returns_one_and_stderr`
- Setup: set `cli.sys.argv` for `assoc` with invalid `--block-names`, use `-q` to suppress masthead.
- Assert: `run_cli()` returns `1` and stderr contains "No block metadata found".

7. `test_run_cli_invalid_chromosome_returns_one_and_stderr`
- Setup: set `cli.sys.argv` for `assoc` with invalid `--chromosomes`.
- Assert: `run_cli()` returns `1` and stderr contains `is_in` type-mismatch text.

8. `test_run_cli_missing_covar_name_returns_one_and_stderr`
- Setup: set `cli.sys.argv` for `assoc` with invalid `--covar-name`.
- Assert: `run_cli()` returns `1` and stderr contains missing-column message.

## Implementation Phases
Phase 1: Add helper-level invalid-input tests.
- Add tests 1-5 in `tests/test_cli.py`.
- Verify with targeted pytest invocation.

Phase 2: Add `run_cli` contract tests for real runtime failures.
- Add tests 6-8 in `tests/test_cli.py`.
- Ensure tests isolate `cli.sys.argv` mutations.

Phase 3: Validate suite and guard against brittle string matches.
- Run `pytest tests/test_cli.py`.
- Normalize assertions to stable substrings, not full exception formatting.

## Additional Considerations
- The chromosome type mismatch is likely an actual bug (string CLI input vs `Int64` metadata column). This plan intentionally captures current behavior first to prevent untracked regressions.
- If code is later improved to coerce chromosome types and produce a friendlier `ValueError`, update tests 3 and 7 in the same change.
