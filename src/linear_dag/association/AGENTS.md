# Association Domain

Last verified: 2026-02-12

## Purpose
Runs downstream statistical genetics workflows on LinearARG/GRM operators.
This domain owns association scans, heritability estimation, PRS scoring, LD utilities, and phenotype simulation helpers.

## Contracts
- **Exposes**: `blup`, `get_gwas_beta_se`, `run_gwas`, `randomized_haseman_elston`, `simulate_phenotype` from `src/linear_dag/association/__init__.py`.
- **Guarantees**: association and heritability paths align sample IDs before computation; GWAS output is returned as Polars frames/lazy frames with per-trait beta and SE columns.
- **Expects**: data contains IID-compatible sample IDs; first covariate column is intercept/all-ones for GWAS and RHE logic; non-HWE paths receive genotypes with individual-node information.

## Dependencies
- **Uses**: `numpy`, `scipy`, `polars`, `pyarrow.parquet` plus core operators (`LinearARG`, `ParallelOperator`, merge operators).
- **Used by**: CLI workflows in `src/linear_dag/cli.py`.
- **Boundary**: avoid changing genotype graph representation here; that belongs to `core`.

## Key Decisions
- Perform linear algebra through operator interfaces to keep memory bounded.
- Handle phenotype missingness by explicit residualization and non-missing bookkeeping.
- Keep PRS loading streaming-oriented (row-group scatter into shared memory) to avoid full dense joins.

## Invariants
- Genotype/phenotype alignment is done through merge operators; merge failure is a hard error.
- Intercept covariate validation is mandatory (`covariates[:, 0] == 1` semantics).
- `assume_hwe=False` requires explicit heterozygote accounting and individual nodes.
- PRS expects Parquet variant IDs (`ID`) to map into LinearARG variant order before scoring.
- Output columns must stay trait-grouped and correspond to the same variant ordering used in compute.

## Key Files
- `src/linear_dag/association/gwas.py` - association effect-size and SE computation
- `src/linear_dag/association/heritability.py` - randomized HE estimator
- `src/linear_dag/association/prs.py` - PRS loading and scoring pipeline
- `src/linear_dag/association/util.py` - residualization and variance helpers
- `src/linear_dag/association/ld.py` - LD matrix helpers
- `src/linear_dag/association/simulation.py` - phenotype simulation
