# Core Domain

Last verified: 2026-02-17

## Purpose
Implements the fundamental LinearARG data model and graph algebra.
This domain owns graph construction, sparse operator behavior, HDF5 persistence, and parallel block execution.

## Contracts
- **Exposes**: `BrickGraph`, `linear_arg_from_genotypes`, `LinearARG`, `list_blocks`, `ParallelOperator`, `Recombination`, triangular solve/topological utilities from `src/linear_dag/core/__init__.py`.
- **Guarantees**: `LinearARG` behaves like a sample-by-variant `LinearOperator`; `write()` and `read()` round-trip graph state; block metadata and IID loading support blockwise workflows; `ParallelOperator.from_hdf5` and `GRMOperator.from_hdf5` share a constructor-preparation pipeline (metadata/BED/filter/IID bootstrap) with class-specific compute hooks.
- **Expects**: genotype inference inputs are CSC-like and shape-consistent; HDF5 blocks contain required datasets/attrs (`indptr`, `indices`, `data`, `variant_indices`, `flip`, sample counts).

## Dependencies
- **Uses**: `numpy`, `scipy.sparse`, `scipy.sparse.linalg`, `polars`, `h5py`, multiprocessing shared memory, Cython-backed modules under `core/*.pyx`.
- **Used by**: `association`, `structure`, `pipeline`, and CLI code paths.
- **Boundary**: keep phenotype/covariate semantics out of this domain; this domain provides numeric/genotype operators only.

## Key Decisions
- Represent genotypes as sparse linear operators to avoid dense matrix materialization.
- Keep graph data in HDF5 blocks to enable partitioned and parallel processing.
- Use explicit `matmat`/`rmatmat` shared-memory worker paths for scalable compute.

## Invariants
- `LinearARG.A` is square sparse adjacency and index arrays remain in-bounds for that matrix.
- `variant_indices` and `flip` are always filtered/reordered together.
- Sample indices occupy the tail of node ordering; `shape` derives from `sample_indices` and `variant_indices`.
- Non-HWE operations require individual nodes (`n_individuals` not `None`) and must fail fast otherwise.
- Parallel operators are used in context managers so workers and shared memory are always cleaned up.
- Constructor calls with `num_processes < 1` raise deterministic `ValueError` before worker startup.
- `alpha` is accepted by both constructors for contract parity but only changes GRM weighting behavior.

## Key Files
- `src/linear_dag/core/lineararg.py` - core representation, filtering, serialization
- `src/linear_dag/core/parallel_processing.py` - process orchestration and shared-memory operators
- `src/linear_dag/core/linear_arg_inference.py` - graph inference from genotype matrices
- `src/linear_dag/core/operators.py` - merge and pairing operators
