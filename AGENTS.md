# linear-dag

Last verified: 2026-02-12

## Scope
This context is intentionally scoped to `src/` only.

## Domain Context Map
- `src/linear_dag/core/AGENTS.md` - core LinearARG representation, graph algebra, and parallel operators.
- `src/linear_dag/association/AGENTS.md` - GWAS, heritability, PRS, LD, and phenotype simulation workflows.
- `src/linear_dag/structure/AGENTS.md` - PCA/SVD-based structure inference on normalized operators.

## Purpose
`linear_dag` builds and analyzes compressed linear ancestral recombination graph (LinearARG) representations of genotype data.
It supports association testing, heritability estimation, PRS scoring, LD utilities, and structure inference at biobank scale.

## Contracts
- **Exposes**:
  - Package API from `src/linear_dag/__init__.py`: `LinearARG`, `BrickGraph`, `ParallelOperator`, `linear_arg_from_genotypes`, `list_blocks`, `read_vcf`, `compute_af`, `flip_alleles`, `apply_maf_threshold`, `binarize`, `randomized_haseman_elston`, `pca`, `svd`.
  - CLI from `src/linear_dag/cli.py`: `assoc`, `rhe`, `score`, `compress`, `multi-step-compress` (`step0`-`step5`).
- **Guarantees**:
  - `LinearARG` behaves as a `scipy.sparse.linalg.LinearOperator` over sample-by-variant genotype space.
  - `LinearARG.write()` / `LinearARG.read()` persist and restore HDF5-backed graph state and metadata.
  - Association and heritability paths align phenotype/covariate rows to genotype IDs using merge operators.
- **Expects**:
  - IID columns in phenotype/covariate inputs (`iid`, `IID`, `#iid`, etc.).
  - First covariate column is an intercept (all ones) for GWAS and RHE code paths.
  - Compatible sparse matrix/operator shapes for all `matmat`/`rmatmat` operations.

## Dependencies
- **Uses**:
  - Numeric/data stack: `numpy`, `scipy`, `polars`, `h5py`.
  - Genotype/IO: `cyvcf2`, `pyarrow.parquet`.
  - Runtime/monitoring: `psutil`.
  - Core acceleration modules under `src/linear_dag/core/*.pyx` (Cython-backed graph/solver primitives).
- **External tools**:
  - `bcftools` is required in multi-step compression step 0 (`msc_step0`).
  - `hdf5plugin` is optional but required for Blosc-compressed HDF5 I/O.
- **Boundary**:
  - Keep edits and assumptions in `src/` unless explicitly asked to expand scope.

## Key Decisions
- Linear operators are used as the primary abstraction to avoid materializing dense genotype matrices.
- Haplotypes are the base representation; diploid semantics are derived by pairing haplotypes when needed.
- HDF5 block structure is used to support blockwise and process-parallel operations.
- GWAS/PRS pipelines favor memory-aware paths (shared memory + lazy/tabular processing) over eager full-matrix materialization.

## Invariants
- `LinearARG.A` is square CSC adjacency; sample nodes are trailing nodes and `shape == (n_samples, n_variants)` derives from sample/variant indices.
- `variant_indices` and `flip` must stay index-aligned and be filtered together.
- Non-HWE paths require individual nodes (`add_individual_nodes`) and should fail fast otherwise.
- BED coordinates are interpreted as UCSC 0-based half-open intervals `[start, end)`.
- Merge/alignment failures between genotype and phenotype identifiers are fatal (do not silently continue).
- Parallel operators must be used within context managers so worker processes and shared memory are cleaned up.

## Agent Guidelines
- Preserve the public API surface:
  - If adding/removing user-facing functions, update `src/linear_dag/__init__.py` and relevant subpackage `__init__.py`.
- Preserve CLI compatibility:
  - Do not silently rename existing subcommands/flags.
  - Keep phenotype/covariate parsing behavior (`_SplitAction`, IID normalization) stable.
- Preserve on-disk compatibility:
  - Do not change `LinearARG` HDF5 schema without an explicit migration plan.
- Respect core numerical assumptions:
  - Keep intercept checks for covariates.
  - Keep explicit shape/type validation for operator algebra paths.
- Keep docstring style consistent with current project conventions:
  - Use markdown-style sections (for example `**Arguments:**`, `**Returns:**`, `**Raises:**`) with a blank line between a section header and its first list item.
  - Class docstrings should describe the class and include `!!! Example` blocks; do not enumerate class attributes in class docstrings.
  - Document only public methods/functions; skip private (`_name`) and dunder (`__name__`) callables.
  - Place `!!! info` and `!!! Example` blocks in the main description section before any `**Arguments:**`, `**Returns:**`, or `**Raises:**` sections.
  - Use `!!! info` blocks for non-obvious behavioral constraints that callers should not miss.
  - Use LaTeX inline notation (`$x$`) for math expressions in docstrings.
  - When introducing symbols, define what each symbol means near first use.
  - Use internal references for project types/classes/functions, in the form [`module.symbol`][]. Example: [`linear_dag.core.lineararg.LinearARG`][]
- For Cython-related changes:
  - Update related `.pyx`/`.pxd` pairs and Python-callable integration points together.

## Key Files
- `src/linear_dag/__init__.py` - public Python API surface
- `src/linear_dag/cli.py` - command-line contracts and argument semantics
- `src/linear_dag/core/lineararg.py` - core graph representation, filtering, serialization
- `src/linear_dag/core/parallel_processing.py` - shared-memory parallel operator implementation
- `src/linear_dag/genotype.py` - VCF ingestion and allele transformations
- `src/linear_dag/association/gwas.py` - association scan implementation
- `src/linear_dag/association/heritability.py` - randomized HE estimator
- `src/linear_dag/association/prs.py` - PRS scoring pipeline
- `src/linear_dag/bed_io.py` - BED region parsing contract
