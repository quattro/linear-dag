# Human Test Plan: Core from_hdf5 Constructor Unification

## Scope
Validate constructor contract unification, filtering parity, lifecycle safety, and CLI/workflow compatibility for:
- `ParallelOperator.from_hdf5`
- `GRMOperator.from_hdf5`

## Preconditions
- Run from repository root.
- Use the project test environment.
- For pytest runs in this repo, include `-p no:capture`.

## Scenario 1: Constructor Contract Parity
1. Run:
   ```bash
   pytest -p no:capture -q tests/core/test_parallel_processing.py -k "signature or constructor_contract"
   ```
2. Expected:
- Signatures match in names/order/defaults.
- `max_num_traits == 8` for both constructors.
- `alpha` accepted by both constructors.

## Scenario 2: Parallel alpha No-Op
1. Run:
   ```bash
   pytest -p no:capture -q tests/core/test_parallel_processing.py -k "parallel and alpha"
   ```
2. Expected:
- Non-default `alpha` does not change `ParallelOperator` outputs.

## Scenario 3: GRM Filtering Parity
1. Run:
   ```bash
   pytest -p no:capture -q tests/core/test_parallel_processing.py -k "grm and filter"
   pytest -p no:capture -q tests/core/test_bed_filtering.py -k "grm or bed"
   ```
2. Expected:
- MAF-only and BED+MAF GRM paths match serial filtered baselines.
- Filtered variant count consistency checks pass.

## Scenario 4: Lifecycle and Error-Path Safety
1. Run:
   ```bash
   pytest -p no:capture -q tests/core/test_parallel_processing.py -k "num_processes and ValueError"
   pytest -p no:capture -q tests/core/test_parallel_processing.py -k "in_place or borrow_variant_data_view or shutdown"
   pytest -p no:capture -q tests/core/test_parallel_processing.py -k "worker and error"
   ```
2. Expected:
- `num_processes < 1` fails with deterministic `ValueError`.
- Context exit joins workers and tears down shared memory.
- Borrowed shared-memory views are closed on exit.
- Worker error path raises `RuntimeError` and executes cleanup.

## Scenario 5: CLI and Workflow Compatibility
1. Run:
   ```bash
   pytest -p no:capture -q tests/cli/test_cli.py -k "parallel_operator_kwargs_consistent or estimate_h2g_passes_filtered"
   pytest -p no:capture -q tests/association/test_rhe.py
   ```
2. Expected:
- CLI forwards unified constructor kwargs for parallel and GRM paths.
- Assoc/score/RHE workflows remain behaviorally compatible.

## Scenario 6: Full Targeted Regression
1. Run:
   ```bash
   pytest -p no:capture -q tests/core/test_parallel_processing.py tests/core/test_bed_filtering.py
   pytest -p no:capture -q tests/cli/test_cli.py
   pytest -p no:capture -q tests/association/test_rhe.py
   ```
2. Expected:
- All suites pass with no regressions.
