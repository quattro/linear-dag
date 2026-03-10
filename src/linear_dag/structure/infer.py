import numpy as np

from scipy.sparse.linalg import (
    eigsh,
    svds,
)

from ..core.lineararg import LinearARG
from ..core.parallel_processing import GRMOperator, ParallelOperator


def _validate_rank(k: int, upper_bound: int, routine_name: str) -> None:
    if not isinstance(k, (int, np.integer)):
        raise TypeError(f"{routine_name}: k must be an integer, got {type(k).__name__}.")
    if k < 1:
        raise ValueError(f"{routine_name}: k must be >= 1, got {k}.")
    if k >= upper_bound:
        raise ValueError(f"{routine_name}: k must be < {upper_bound}, got {k}.")


def svd(linarg: LinearARG | ParallelOperator, k: int = 20) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the top-$k$ singular triplets of a normalized genotype operator.

    Let $X$ denote `linarg.normalized`, where rows index samples and columns
    index variants. This routine computes a truncated decomposition
    $X \\approx U \\Sigma V^\\top$ using SciPy's `svds`, then reorders outputs so
    singular values are descending.

    !!! info

        `k` must satisfy `1 <= k < min(X.shape)` for this solver path.

    **Arguments:**

    - `linarg`: Input [`linear_dag.core.lineararg.LinearARG`][] or
      [`linear_dag.core.parallel_processing.ParallelOperator`][].
    - `k`: Number of singular triplets to compute.

    **Returns:**

    - Tuple `(U, s, Vt)` with `U.shape == (n_samples, k)`,
      `s.shape == (k,)`, and `Vt.shape == (k, n_variants)`.

    **Raises:**

    - `TypeError`: If `k` is not an integer.
    - `ValueError`: If `k` is out of bounds for the normalized operator.
    """
    normed_arg = linarg.normalized
    _validate_rank(k, min(normed_arg.shape), "svd")
    eigvecs, svals, loadings = svds(normed_arg, k=k)
    order = np.argsort(svals)[::-1]

    return eigvecs[:, order], svals[order], loadings[order, :]


def pca(grm: GRMOperator, k: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """Compute the top-$k$ principal components from a GRM-like operator.

    Let $K$ denote `grm`, where $K$ is a sample-by-sample symmetric operator
    (for example, a genetic relatedness matrix). This routine computes the
    largest eigenpairs of $K$ with `eigsh`, then reorders them so eigenvalues
    are descending.

    !!! info

        `k` must satisfy `1 <= k < grm.shape[0]` for this solver path.

    **Arguments:**

    - `grm`: Input [`linear_dag.core.parallel_processing.GRMOperator`][].
    - `k`: Number of principal components to compute.

    **Returns:**

    - Tuple `(pcs, eigvals)` with `pcs.shape == (n_samples, k)` and
      `eigvals.shape == (k,)`.

    **Raises:**

    - `TypeError`: If `k` is not an integer.
    - `ValueError`: If `k` is out of bounds for `grm`.
    """
    _validate_rank(k, grm.shape[0], "pca")
    eigvals, eigvecs = eigsh(grm, k=k, which="LM")
    order = np.argsort(eigvals)[::-1]

    return eigvecs[:, order], eigvals[order]
