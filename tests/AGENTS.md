# tests

Last verified: 2026-02-13

## Scope
This context is intentionally scoped to `tests/`.

## Purpose
`tests/` validates `linear_dag` behavior with module-aligned unit suites and selected integration-leaning coverage that exercises real data paths.

## Contracts
- **Exposes**:
  - Shared fixtures in `tests/conftest.py`: `test_data_dir`, `linarg_h5_path`, `phenotypes_tsv_path`, `linarg_block_metadata`, `first_block_name`.
  - Shared helpers in `tests/helpers/linarg_fixtures.py` for loading real test fixtures.
  - Module-aligned unit layout under `tests/core/`, `tests/association/`, `tests/structure/`, and `tests/cli/`.
- **Guarantees**:
  - Applicable unit tests prefer real fixture data from `tests/testdata/` (including `test_chr21_50.h5`) over redundant mock-only setup.
  - Test discovery remains rooted at `tests/` (`testpaths = ["tests"]`).
- **Expects**:
  - Tests that allocate process/shared-memory resources (for example `ParallelOperator`) manage them via explicit context managers.
  - New tests that need canonical fixture paths use shared fixtures instead of per-module `TEST_DATA_DIR` constants.

## Dependencies
- **Uses**:
  - `pytest` and shared fixtures from `tests/conftest.py`.
  - Real fixture assets in `tests/testdata/`.
  - Runtime libs used by test flows (`numpy`, `polars`, `scipy`) where needed.
- **Boundary**:
  - Keep test-organization and fixture-policy guidance in `tests/`.
  - Production API/contracts remain documented under `src/**/AGENTS.md`.

## Key Decisions
- Unit tests are organized by module path (`tests/<module>/`), without a `tests/unit/` prefix.
- Shared real-data fixture primitives are centralized in `tests/conftest.py` and `tests/helpers/linarg_fixtures.py`.
- Integration-leaning follow-ups are tracked in `tests/INTEGRATION_REHOMING_NOTES.md` and `tests/cli/DEFERRED_INTEGRATION_CASES.md`.

## Invariants
- `linarg_h5_path` resolves to `tests/testdata/test_chr21_50.h5` and is used as the canonical LinearARG fixture path when applicable.
- CLI and logging tests should remain runnable in environments that use `pytest -p no:capture`.
- Module test packages (`tests/core`, `tests/association`, `tests/structure`, `tests/cli`) remain importable and discoverable by pytest.

## Key Files
- `tests/conftest.py` - global shared fixtures and fixture primitives.
- `tests/helpers/linarg_fixtures.py` - shared helper loader utilities.
- `tests/README.md` - contributor-facing test layout and fixture conventions.
- `tests/INTEGRATION_REHOMING_NOTES.md` - follow-up plan for integration-test re-homing.
- `tests/cli/DEFERRED_INTEGRATION_CASES.md` - deferred CLI integration-like coverage notes.
