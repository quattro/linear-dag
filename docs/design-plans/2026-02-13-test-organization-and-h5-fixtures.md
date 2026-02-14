# Test Organization and H5 Fixture Consolidation Design

## Summary
This design restructures unit tests into module-aligned directories under `tests/<module>/` (`tests/core/`, `tests/association/`, `tests/structure/`, `tests/cli/`) while preserving test behavior. It also introduces shared pytest fixtures and helpers so tests that rely on `tests/testdata/test_chr21_50.h5` use one consistent setup path based on real test data rather than repeated local boilerplate.

Implementation is phased to minimize risk: establish shared fixture primitives first, then migrate module groups incrementally, and finish with stabilization/documentation. The plan keeps expensive resource ownership explicit (especially `ParallelOperator` context lifetimes), avoids production code changes, and preserves discovery/collection behavior from the repository root.

## Definition of Done
Produce an investigation-backed test reorganization design that moves unit tests into module-aligned subfolders under `tests/<module>/` and introduces global helpers/fixtures for tests that rely on `tests/testdata/test_chr21_50.h5` (preferring real test data over mock data when practical). The design must preserve current test behavior, identify migration phases with exact file paths, and define verification steps that keep discovery and execution stable.

## Acceptance Criteria
### test-organization-and-h5-fixtures.AC1: Unit tests are organized by module under `tests/<module>/`
- **test-organization-and-h5-fixtures.AC1.1 Success:** Core unit tests are located under `tests/core/` with paths that map to their prior core-focused coverage.
- **test-organization-and-h5-fixtures.AC1.2 Success:** Association unit tests are located under `tests/association/` and structure unit tests are located under `tests/structure/`.
- **test-organization-and-h5-fixtures.AC1.3 Success:** CLI unit tests are located under `tests/cli/` without introducing a `tests/unit/` prefix.
- **test-organization-and-h5-fixtures.AC1.4 Guardrail:** `pytest` collection still succeeds from repository root with `testpaths = ["tests"]`.
- **test-organization-and-h5-fixtures.AC1.5 Failure/Guardrail:** If any moved test path breaks discovery or imports, the migration is considered incomplete until fixed.

### test-organization-and-h5-fixtures.AC2: Shared HDF5 fixture primitives replace repeated local setup
- **test-organization-and-h5-fixtures.AC2.1 Success:** `tests/conftest.py` provides global fixtures for `test_data_dir`, `linarg_h5_path`, `phenotypes_tsv_path`, and `linarg_block_metadata`.
- **test-organization-and-h5-fixtures.AC2.2 Success:** Shared helpers in `tests/helpers/` provide reusable loading utilities for block selection, `LinearARG` loading, and `ParallelOperator` context construction.
- **test-organization-and-h5-fixtures.AC2.3 Success:** Migrated tests that currently duplicate `TEST_DATA_DIR` and `test_chr21_50.h5` setup consume shared fixtures/helpers instead of redefining equivalent local setup.
- **test-organization-and-h5-fixtures.AC2.4 Guardrail:** Fixture design remains primitive-first; long-lived shared `ParallelOperator` instances are not introduced.

### test-organization-and-h5-fixtures.AC3: Real test data remains primary for applicable unit tests
- **test-organization-and-h5-fixtures.AC3.1 Success:** Applicable unit tests continue to use `tests/testdata/test_chr21_50.h5` via shared fixtures rather than replacing coverage with mock-only data.
- **test-organization-and-h5-fixtures.AC3.2 Success:** Existing real-data-based logic paths (`LinearARG.read`, `list_blocks`, `ParallelOperator.from_hdf5`) remain exercised in migrated tests.
- **test-organization-and-h5-fixtures.AC3.3 Failure/Guardrail:** Any migration that drops real-data coverage for previously real-data unit tests without explicit justification is considered out of scope.

### test-organization-and-h5-fixtures.AC4: Migration is behavior-preserving and reviewable
- **test-organization-and-h5-fixtures.AC4.1 Success:** Reorganization changes are primarily path and fixture-consumption updates, not semantic rewrites of test intent.
- **test-organization-and-h5-fixtures.AC4.2 Success:** Targeted test runs for moved modules pass after each migration phase.
- **test-organization-and-h5-fixtures.AC4.3 Success:** Contributor guidance documents the new folder and fixture conventions.
- **test-organization-and-h5-fixtures.AC4.4 Guardrail:** Before implementation commits based on this design, `requesting-code-review` is run and all findings are resolved.

## Glossary
- **Module-Aligned Test Layout:** A test directory structure that maps tests to source domains (for example `tests/core/` for `src/linear_dag/core/`).
- **Shared Fixture Primitives:** Global pytest fixtures that provide stable paths/metadata instead of preloaded heavy runtime objects.
- **`test_chr21_50.h5`:** The canonical real LinearARG-backed test dataset used across multiple unit tests.
- **Block Metadata:** Per-block information (for example block names and variant counts) loaded from HDF5 and reused for block-scoped tests.
- **Behavior-Preserving Refactor:** A change that restructures organization and setup without changing test semantics or expected outcomes.
- **Context-Manager Ownership:** The rule that tests explicitly open/close resources like `ParallelOperator` inside `with` blocks.
- **Discovery Stability:** Maintaining successful `pytest` collection and path resolution after file moves.
- **Deferred Integration Scope:** Integration-leaning tests intentionally left for a follow-up migration to keep the current pass low-risk.

## Architecture
This design reorganizes unit tests from a flat `tests/` layout into module-aligned folders directly under `tests/`, without introducing a `tests/unit/` prefix. The target structure is `tests/core/`, `tests/association/`, `tests/structure/`, and `tests/cli/` for unit-scope coverage that maps to source ownership in `src/linear_dag/`. Integration-leaning tests are intentionally left for later follow-up to keep this pass low-risk and incremental.

A shared fixture layer is added through `tests/conftest.py` and helper utilities in `tests/helpers/`. The fixture contract is primitive-first: shared fixtures expose stable test-data paths and metadata (`test_data_dir`, `linarg_h5_path`, `phenotypes_tsv_path`, `linarg_block_metadata`) plus helper callables for loading block-scoped resources (`LinearARG` by block and `ParallelOperator` context access). This reduces repeated setup while preserving explicit context-manager ownership inside tests for worker cleanup and resource safety.

The primary objective is behavior-preserving consolidation. Existing tests that currently duplicate `TEST_DATA_DIR` and first-block lookup logic migrate to common fixtures using the same real artifacts in `tests/testdata/`, especially `tests/testdata/test_chr21_50.h5`. No production API, HDF5 schema, or CLI semantics are changed.

## Existing Patterns
Investigation found a currently flat test tree with eleven top-level `test_*.py` modules in `tests/` and no `tests/conftest.py`. At least seven modules independently define `TEST_DATA_DIR = Path(__file__).parent / "testdata"` and repeatedly reference `tests/testdata/test_chr21_50.h5`, with heavy repetition in `tests/test_cli.py`, `tests/test_parallel_processing.py`, `tests/test_bed_filtering.py`, `tests/test_association.py`, `tests/test_rhe.py`, and `tests/test_ld.py`.

Current tests already prefer real fixture data and production loaders (`LinearARG.read`, `list_blocks`, `ParallelOperator.from_hdf5`) over broad mocking for genotype workflows. This design follows that existing pattern and only centralizes setup to remove duplication. The design also preserves explicit `with ...` context-manager usage for parallel operators, which matches current cleanup expectations.

## Implementation Phases
<!-- START_PHASE_1 -->
### Phase 1: Shared Test Fixture Foundation
**Goal:** Introduce global fixture primitives and helper utilities for shared HDF5 test-data access.

**Components:**
- `tests/conftest.py` with global path/metadata fixtures:
  - `test_data_dir`
  - `linarg_h5_path` (points to `tests/testdata/test_chr21_50.h5`)
  - `phenotypes_tsv_path`
  - `linarg_block_metadata`
- `tests/helpers/linarg_fixtures.py` with helper functions/callables for:
  - loading first block name from metadata
  - loading `LinearARG` by block
  - opening `ParallelOperator.from_hdf5(...)` contexts

**Dependencies:** None.

**Done when:**
- Fixtures/helpers are importable and deterministic.
- Existing tests can adopt fixtures incrementally without behavior changes.
- Discovery command (`pytest --collect-only -q`) succeeds.
<!-- END_PHASE_1 -->

<!-- START_PHASE_2 -->
### Phase 2: Core Test Folder Migration
**Goal:** Move core unit tests into `tests/core/` and replace duplicated local HDF5 setup with shared fixtures.

**Components:**
- Move/adjust core-focused tests from:
  - `tests/test_lineararg.py`
  - `tests/test_parallel_processing.py`
  - `tests/test_bed_filtering.py`
- Target paths:
  - `tests/core/test_lineararg.py`
  - `tests/core/test_parallel_processing.py`
  - `tests/core/test_bed_filtering.py`
- Refactor moved files to consume global fixtures/helpers instead of local `TEST_DATA_DIR` and repeated block lookup code.

**Dependencies:** Phase 1.

**Done when:**
- Core files execute from new paths with unchanged test intent.
- Migrated files no longer declare redundant local `TEST_DATA_DIR` constants.
- Targeted runs for moved files pass.
<!-- END_PHASE_2 -->

<!-- START_PHASE_3 -->
### Phase 3: Association and Structure Folder Migration
**Goal:** Move association/structure unit tests into module folders and standardize real-data fixture usage.

**Components:**
- Move/adjust association-focused tests:
  - `tests/test_association.py` -> `tests/association/test_association.py`
  - `tests/test_rhe.py` -> `tests/association/test_rhe.py`
  - `tests/test_ld.py` -> `tests/association/test_ld.py`
- Move/adjust structure-focused tests:
  - `tests/test_structure.py` -> `tests/structure/test_structure.py`
- Replace duplicated HDF5 path and block metadata setup with shared fixtures/helpers where practical.

**Dependencies:** Phases 1-2.

**Done when:**
- Moved association/structure tests pass from new module folders.
- Real-data usage through shared fixtures is consistent and explicit.
- No regression in test discovery for module folders.
<!-- END_PHASE_3 -->

<!-- START_PHASE_4 -->
### Phase 4: CLI Unit Folder Migration
**Goal:** Move CLI unit tests into `tests/cli/` while preserving behavior and keeping integration-heavy scope constrained.

**Components:**
- Move/adjust:
  - `tests/test_cli.py` -> `tests/cli/test_cli.py`
- Refactor duplicated path setup to shared fixtures for `test_chr21_50.h5` and phenotype TSV access.
- Keep integration-heavy or end-to-end style cases explicitly documented if deferred from this pass.

**Dependencies:** Phases 1-3.

**Done when:**
- CLI unit tests pass from `tests/cli/test_cli.py`.
- Shared fixtures are used for repeated data-path setup.
- Any deferred integration-style tests are explicitly called out in docs or TODO notes.
<!-- END_PHASE_4 -->

<!-- START_PHASE_5 -->
### Phase 5: Stabilization and Contributor Guidance
**Goal:** Lock in conventions and verify suite stability after migration.

**Components:**
- Add/update `tests/README.md` with folder conventions and fixture usage guidance.
- Validate collection and representative execution across module folders.
- Record follow-up plan for optional integration-test re-homing (outside this change scope).

**Dependencies:** Phases 1-4.

**Done when:**
- Documentation reflects new folder and fixture conventions.
- Test collection and selected runs pass with new structure.
- Follow-up items are explicit and scoped.
<!-- END_PHASE_5 -->

## Additional Considerations
- Preserve behavior-first migration: this effort is organizational and should avoid functional test rewrites.
- Keep expensive resource ownership explicit in tests; global fixtures should provide primitives/helpers, not long-lived shared operators.
- Prefer `session` scope for immutable path/metadata fixtures and function scope for loaded objects that require cleanup.
- Maintain compatibility with existing `pytest` discovery via `testpaths = ["tests"]` and avoid introducing brittle import assumptions.
- Before implementation commits derived from the future implementation plan, run the `requesting-code-review` workflow and resolve all findings (including minor).
