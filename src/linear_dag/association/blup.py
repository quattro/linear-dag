from dataclasses import dataclass

import numpy as np

from scipy.sparse import csc_matrix, csr_matrix, diags, eye
from scipy.sparse.linalg import aslinearoperator, cg, LinearOperator, spsolve_triangular

from ..core.lineararg import LinearARG


@dataclass
class triangular_solver(LinearOperator):
    """LinearOperator wrapper for triangular solves on $I - A$.

    !!! Example

        ```python
        solver = triangular_solver(adjacency_submatrix)
        x = solver @ b
        y = solver.T @ b
        ```
    """

    A: csr_matrix

    @property
    def dtype(self):
        """Return the scalar dtype used by the wrapped sparse matrix."""
        return self.A.dtype

    @property
    def shape(self):
        """Return the matrix shape expected by `LinearOperator` consumers."""
        return self.A.shape

    def _matvec(self, other):
        if other.ndim == 1:
            other = other.reshape(-1, 1)

        return spsolve_triangular(eye(self.A.shape[0]) - self.A, other)

    def _rmatvec(self, other):
        if other.ndim == 1:
            other = other.reshape(1, -1)

        return spsolve_triangular(eye(self.A.shape[1]) - self.A.T, other.T, lower=False).T


def blup(linarg: LinearARG, heritability: float, y: np.ndarray):
    """Compute the best linear unbiased predictor (BLUP) for a phenotype vector.

    !!! info

        `heritability` is interpreted as narrow-sense heritability in $[0, 1]$.
        This function assumes `y` is already aligned to `linarg.sample_indices`.

    **Arguments:**

    - `linarg`: [`linear_dag.core.lineararg.LinearARG`][] operator for the
      genotype matrix.
    - `heritability`: Proportion of phenotypic variance attributed to genetic effects.
    - `y`: Phenotype vector aligned to `linarg` samples.

    **Returns:**

    - BLUP estimate for `y` with the same shape as the input phenotype vector.
    """

    X = linarg
    n, m = X.shape
    k = linarg.A.shape[0]

    ## Generate M and S
    M = csc_matrix(eye(k))
    M = M[:, linarg.variant_indices]
    S = csr_matrix(eye(k))
    S = S[linarg.sample_indices, :]
    non_sample_indices = np.setdiff1d(np.arange(k), linarg.sample_indices)
    T = csr_matrix(eye(k))
    T = T[non_sample_indices, :]

    # Set scalar for error variance
    var_e = 1 - heritability
    var_g = heritability / m

    ## Generate Sigma vector
    Sigma = var_g * eye(m)
    eps = 1e-12
    SigmaTilde = eps * np.ones(k)
    SigmaTilde[linarg.variant_indices] = var_g
    SigmaTilde[linarg.sample_indices] = var_e

    ## Generate Om
    Om = aslinearoperator(diags(1 / SigmaTilde))

    ## Generate B
    I = eye(k)  # noqa: E741
    I_minus_A = aslinearoperator(I - X.A)
    B = (I_minus_A.T @ Om) @ I_minus_A

    # B_s,s * y
    first_term = S @ (B @ (S.T @ y))

    # B_s,t * (B_t,t)^-1 * B_t,s * y
    second_term = T @ (B @ (S.T @ y))

    SigmaTilde = aslinearoperator(diags(SigmaTilde[non_sample_indices]))
    adjacency_matrix = linarg.A[non_sample_indices, :]
    adjacency_matrix = adjacency_matrix[:, non_sample_indices]
    linarg_adjacency_submatrix_nonsamples = triangular_solver(adjacency_matrix)
    preconditioner = linarg_adjacency_submatrix_nonsamples @ SigmaTilde @ linarg_adjacency_submatrix_nonsamples.T

    B_tt = aslinearoperator(T) @ B @ aslinearoperator(T.T)

    second_term, info = cg(B_tt, second_term, M=preconditioner)
    second_term_now = S @ (B @ (T.T @ second_term))

    K = linarg @ aslinearoperator(Sigma) @ linarg.T

    blup = K @ (first_term - second_term_now)

    return blup
