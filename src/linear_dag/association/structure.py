import numpy as np

from scipy.sparse.linalg import eigs, svds

from ..core.lineararg import LinearARG


def svd(linarg: LinearARG, k: int = 20) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ """
    # TODO: sanity check input...
    normed_arg = linarg.normalized
    eigvecs, svals, loadings = svds(normed_arg, k=k)

    return eigvecs, svals, loadings


def pca(linarg: LinearARG, k: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """ """
    # TODO: sanity check input...
    normed_arg = linarg.normalized
    GRM = normed_arg @ normed_arg.T
    eigvals, eigvecs = eigs(GRM, k=k)

    return eigvecs, eigvals
