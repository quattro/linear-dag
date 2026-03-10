# Tests

## Layout

- `tests/core/`: `linear_dag.core` unit tests (LinearARG, operators, BED filtering).
- `tests/association/`: `linear_dag.association` unit tests (GWAS, RHE, LD).
- `tests/structure/`: `linear_dag.structure` unit tests (PCA/SVD workflows).
- `tests/cli/`: CLI unit tests.
- `tests/testdata/`: shared real-data fixtures used by unit tests.

## Shared Fixtures

Global fixtures are defined in `tests/conftest.py` and helper utilities in `tests/helpers/linarg_fixtures.py`.

Prefer using fixture-provided paths over module-local path constants:

- `linarg_h5_path`
- `phenotypes_tsv_path`
- `linarg_block_metadata`
- `first_block_name`
- `test_data_dir`

## Real Data Policy

When possible, prefer shared real fixtures in `tests/testdata/` (for example `test_chr21_50.h5`) instead of mock-only setup for operator/genotype behavior.

## Resource Ownership

Tests that open `ParallelOperator` (or similar resources) should keep lifecycle management explicit with per-test context managers so worker processes and shared memory are always cleaned up.
