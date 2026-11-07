# linear-dag

Last verified: 2026-11-06

## Scope
This context covers the public package under `src/` and the JAX native-build integration in `hatch_build.py`.

## Domain Context Map
- No `src/`-local `AGENTS.md` files are present; use the key files below for source ownership.

## Purpose
`linear_dag` builds and analyzes compressed linear ancestral recombination graph (LinearARG) representations of genotype data.
It supports association testing, heritability estimation, PRS scoring, LD utilities, and structure inference at biobank scale.

## Contracts
- **Exposes**:
  - Package API from `src/linear_dag/__init__.py`: `as_lineax_operator`, `LinearARG`, `BrickGraph`, `ParallelOperator`, `GRMOperator`, `Backend`, `JaxLinearARG`, `JaxParallelOperator`, `linear_arg_from_genotypes`, `list_blocks`, `read_vcf`, `compute_af`, `flip_alleles`, `apply_maf_threshold`, `binarize`, `randomized_haseman_elston`, `pca`, `svd`.
  - CLI from `src/linear_dag/cli.py`: `assoc`, `rhe`, `score`, `compress`, `multi-step-compress` (`step0`-`step5`).
- **Guarantees**:
  - `LinearARG` behaves as a `scipy.sparse.linalg.LinearOperator` over sample-by-variant genotype space.
  - `JaxLinearARG` and `JaxParallelOperator` expose JAX-compatible products over the same sample-by-variant genotype semantics.
  - `Backend.AUTO` selects CPU FFI when its native handler is available on CPU and otherwise uses pure JAX; non-CPU platforms use pure JAX.
  - `rhe --jax-backend` is experimental, reports its resolved backend and devices, and rejects variant-filtering options that are not yet supported by the JAX GRM path.
  - `LinearARG.write()` / `LinearARG.read()` persist and restore HDF5-backed graph state and metadata.
  - Association and heritability paths align phenotype/covariate rows to genotype IDs using merge operators.
  - `ParallelOperator.from_hdf5` and `GRMOperator.from_hdf5` share their core constructor keywords; only `ParallelOperator` accepts `dtype`, and `alpha` is operational only for GRM weighting.
- **Expects**:
  - IID columns in phenotype/covariate inputs (`iid`, `IID`, `#iid`, etc.).
  - First covariate column is an intercept (all ones) for GWAS and RHE code paths.
  - Compatible sparse matrix/operator shapes for all `matmat`/`rmatmat` operations.

## Dependencies
- **Uses**:
  - Numeric/data stack: `numpy`, `scipy`, `polars`, `h5py`, `jax`, `jaxlib`, `equinox`, `lineax`, `jaxtyping`.
  - Genotype/IO: `cyvcf2`, `pyarrow.parquet`.
  - Runtime/monitoring: `psutil`.
  - Core acceleration modules under `src/linear_dag/core/*.pyx` (Cython-backed graph/solver primitives).
  - Optional JAX CPU FFI extension under `src/linear_dag/core/jaxlinarg/kernels/`; build failures leave the pure-JAX fallback available.
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
- JAX backend selection stays explicit through `Backend`. Native CPU products use one device and packed ragged graph buffers; `num_processes` controls native worker threads. Pure-JAX products retain mesh distribution.
- The native HiJAX path supports JIT, batching, JVP/VJP and higher derivatives in RHS values and dynamic GRM diagonals. Graph construction remains host-side and graph arrays remain integer operands.
- Python 3.12 and JAX/jaxlib 0.12.0.1 are required for the experimental HiJAX API. Keep pins synchronized between build and runtime dependencies.
- Native scratch defaults to 256 MiB with RHS tiles up to 8 columns; this excludes graph, input/output and transformation intermediates. The on-disk HDF5 schema is unchanged.

## Invariants
- `LinearARG.A` is square CSC adjacency; sample nodes are trailing nodes and `shape == (n_samples, n_variants)` derives from sample/variant indices.
- `variant_indices` and `flip` must stay index-aligned and be filtered together.
- Non-HWE paths require individual nodes (`add_individual_nodes`) and should fail fast otherwise.
- BED coordinates are interpreted as UCSC 0-based half-open intervals `[start, end)`.
- Merge/alignment failures between genotype and phenotype identifiers are fatal (do not silently continue).
- Parallel operators must be used within context managers so worker processes and shared memory are cleaned up.
- Constructor calls with `num_processes < 1` fail deterministically with a user-facing `ValueError`.

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
  - Keep pure-JAX and CPU FFI products numerically aligned for float32/float64, vectors/matrices, JIT, batching, and autodiff.
- Preserve the JAX boundary:
  - Keep `JaxGRMOperator` and native build diagnostics under `linear_dag.core.jaxlinarg`; they are not package-root exports.
  - Keep `rhe --jax-backend` experimental and fail clearly for unsupported variant filtering.
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
- `src/linear_dag/core/jaxlinarg/operator.py` - JAX single-block operator and backend dispatch
- `src/linear_dag/core/jaxlinarg/wrapper.py` - JAX multi-block operator and mesh/block assignment
- `src/linear_dag/core/jaxlinarg/kernels/` - pure-JAX and optional CPU FFI kernels
- `src/linear_dag/association/_heritability_jax.py` - experimental JAX RHE orchestration
- `hatch_build.py` - optional JAX CPU FFI build hook
- `src/linear_dag/genotype.py` - VCF ingestion and allele transformations
- `src/linear_dag/association/gwas.py` - association scan implementation
- `src/linear_dag/association/heritability.py` - randomized HE estimator
- `src/linear_dag/association/prs.py` - PRS scoring pipeline
- `src/linear_dag/bed_io.py` - BED region parsing contract

## Lineax interoperability

- `as_lineax_operator` adapts the existing JAX genotype/GRM modules to Lineax,
  preserving graph/weight arrays as PyTree leaves and using explicit transpose
  products. It is exported at package root and under `core.jaxlinarg`.
- GRM symmetry is known; positive semidefiniteness is not inferred because dynamic
  diagonals may be signed. Solver tags remain caller assertions.
- Iterative solves do not materialize matrices. Explicit matrix/diagonal extraction
  or direct solvers may do so. Pure-JAX backend AD limitations are unchanged.
