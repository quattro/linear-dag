import numpy as np

from scipy.sparse.linalg import (
    eigsh,
    svds,
)

from .. import LinearARG


def _validate_rank(k: int, upper_bound: int, routine_name: str) -> None:
    if not isinstance(k, (int, np.integer)):
        raise TypeError(f"{routine_name}: k must be an integer, got {type(k).__name__}.")
    if k < 1:
        raise ValueError(f"{routine_name}: k must be >= 1, got {k}.")
    if k >= upper_bound:
        raise ValueError(f"{routine_name}: k must be < {upper_bound}, got {k}.")


def svd(linarg: LinearARG, k: int = 20) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute a truncated SVD on the normalized genotype operator.

    Returns left singular vectors, singular values, and right singular vectors.
    """
    normed_arg = linarg.normalized
    _validate_rank(k, min(normed_arg.shape), "svd")
    eigvecs, svals, loadings = svds(normed_arg, k=k)
    order = np.argsort(svals)[::-1]

    return eigvecs[:, order], svals[order], loadings[order, :]


def pca(linarg: LinearARG, k: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """Compute top principal components from the normalized genotype operator."""
    normed_arg = linarg.normalized
    GRM = normed_arg @ normed_arg.T
    _validate_rank(k, GRM.shape[0], "pca")
    eigvals, eigvecs = eigsh(GRM, k=k, which="LM")
    order = np.argsort(eigvals)[::-1]

    return eigvecs[:, order], eigvals[order]
