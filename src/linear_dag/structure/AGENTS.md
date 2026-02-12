# Structure Domain

Last verified: 2026-02-12

## Purpose
Provides dimensionality-reduction entry points for population structure analysis on LinearARG-based genotype operators.

## Contracts
- **Exposes**: `pca` and `svd` from `src/linear_dag/structure/__init__.py`.
- **Guarantees**: both APIs operate on normalized LinearARG-like operators and return decomposition outputs directly from SciPy routines.
- **Expects**: callers pass operators with valid shape/normalization behavior; requested rank `k` must satisfy SciPy solver constraints.

## Dependencies
- **Uses**: `numpy`, `scipy.sparse.linalg` (`eigs`, `svds`), and package-level `LinearARG`.
- **Used by**: package public API and downstream analysis workflows that need PCs/singular vectors.
- **Boundary**: no phenotype modeling, no graph construction, no file-format ownership.

## Key Decisions
- Reuse normalized operator representations from `core` instead of materializing covariance matrices when avoidable.
- Keep API minimal (`pca`, `svd`) and delegate solver details to SciPy.

## Invariants
- `linarg.normalized` is the expected numeric basis for both PCA and SVD calls.
- `pca` returns real-valued eigenpairs from a symmetric solver, with eigenvalues sorted descending.
- `svd` returns singular values sorted descending, with vectors re-ordered to match.
- This domain should remain thin: orchestration and IO stay outside `structure`.

## Key Files
- `src/linear_dag/structure/infer.py` - PCA/SVD implementations
- `src/linear_dag/structure/__init__.py` - public exports
