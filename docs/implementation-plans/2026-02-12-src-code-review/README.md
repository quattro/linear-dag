# src Review / Remediation Plan Status

Last updated: 2026-02-12

## Revised Goal
Execute a 4-phase `src/` review/remediation plan with immediate priority on CLI hardening and cleanup, followed by core, association, and structure remediation in order.

## Priority Change
- Original plan intent: review all modules by phase.
- Current execution intent: keep the 4-phase structure, but complete CLI review plus implementation first before moving to module remediation.

## Current Status
- `phase_01.md` (CLI): active and partially completed.
- `phase_02.md` (Core): active and partially completed.
- `phase_03.md` (Association): active and partially completed.
- `phase_04.md` (Structure + consolidation): active and partially completed.

## Work Completed So Far
- Produced severity-ranked review findings for `cli`, `core`, `association`, and `structure`.
- Implemented CLI hardening/cleanup in `src/linear_dag/cli.py`:
  - early block metadata validation for workflows that require blocks
  - de-duplication of column-selection logic
  - logging handler lifecycle cleanup
  - version fallback for non-installed/dev contexts
  - removed dead/redundant CLI code paths
- Added CLI-focused regression tests in `tests/test_cli.py`:
  - `test_cli_version_fallback`
  - `test_prep_data_requires_block_metadata`
- Implemented core fixes in `src/linear_dag/core/lineararg.py`:
  - implemented `LinearARG.copy()`
  - fixed explicit `sex` propagation in `add_individual_nodes()`
  - made `list_blocks()` chromosome sorting robust for non-numeric chromosomes (e.g., `chrX`)
- Added core-focused regression tests in `tests/test_lineararg.py`:
  - `test_lineararg_copy_independent_arrays`
  - `test_add_individual_nodes_propagates_explicit_sex`
  - `test_list_blocks_handles_non_numeric_chromosomes`
- Implemented association fixes in `src/linear_dag/association/gwas.py`:
  - hardened non-HWE guardrails to require `n_individuals`, `iids`, and `number_of_heterozygotes()`
  - removed redundant `data.select(...).collect()` calls in `run_gwas` by collecting required inputs once
  - corrected `get_gwas_beta_se` return-contract docs/types to match runtime behavior
- Added association-focused regression tests in `tests/test_association.py`:
  - `test_run_gwas_non_hwe_requires_heterozygote_counter`
  - `test_get_gwas_beta_se_returns_four_arrays`
- Implemented structure fixes in `src/linear_dag/structure/infer.py`:
  - switched PCA to `eigsh` for symmetric GRM solving
  - added explicit rank/type validation for `k` in both `pca` and `svd`
  - made returned spectral values/vectors deterministic via descending sort
- Added structure-focused regression tests in `tests/test_structure.py`:
  - `test_svd_returns_sorted_singular_values`
  - `test_pca_returns_sorted_real_eigenpairs`
  - `test_structure_rank_validation`

## Verification Notes
- Compile checks for updated files completed.
- Targeted runtime checks completed for new CLI guard and version fallback behavior.
- Targeted runtime checks completed for core regression scenarios.
- Targeted runtime checks completed for association non-HWE guard and `get_gwas_beta_se` output contract.
- Targeted runtime checks completed for structure solver/ordering and rank-validation behavior.
- Full pytest run is still pending in a non-restricted runtime (sandbox constraints cause abnormal `pytest` termination and shared-memory multiprocessing limits in this environment).

## Next Steps
1. Re-run CLI test suite in a runtime that allows multiprocessing shared memory.
2. Re-run core, association, and structure targeted pytest cases in a non-restricted runtime.
3. Run any additional deep-dive module reviews requested after phase completion.
4. Finalize remediation summary across all four phases.
