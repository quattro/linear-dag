from networkx.linalg.graphmatrix import adjacency_matrix

# Some of these imports may be superfluous 

import sys
import linear_dag as ld
from linear_dag import LinearARG

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, cg, aslinearoperator, gmres, spsolve_triangular
from scipy.sparse import eye, diags, csc_matrix, csr_matrix
import time
from dataclasses import dataclass
import pandas as pd

@dataclass
class triangular_solver(LinearOperator):
    A: csr_matrix

    @property
    def dtype(self):
        return self.A.dtype

    @property
    def shape(self):
        return self.A.shape

    def _matvec(self, other):
        if other.ndim == 1:
            other = other.reshape(-1, 1)

        return spsolve_triangular(eye(self.A.shape[0]) - self.A, other)

    def _rmatvec(self, other):
        if other.ndim == 1:
            other = other.reshape(1, -1)

        return spsolve_triangular(eye(self.A.shape[1]) - self.A.T, other.T, lower=False).T

def method_Schur(linarg: LinearARG, heritability: float,
                       y):
    """
    
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
    h2 = heritability
    var_e = 1 - h2
    var_g = h2 / m

    ## Generate Sigma vector
    Sigma = var_g * eye(m)
    eps = 1e-12
    SigmaTilde = eps * np.ones(k)
    SigmaTilde[linarg.variant_indices] = var_g
    SigmaTilde[linarg.sample_indices] = var_e

    ## Generate Om
    Om = aslinearoperator(diags(1 / SigmaTilde))

    ## Generate B
    I = eye(k)
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
    