from functools import partial
from typing import Callable, Optional, Union

import numpy as np
import scipy as sp

from numpy.random import Generator
from scipy.sparse import diags
from scipy.sparse.linalg import aslinearoperator, LinearOperator

from ..core import LinearARG


def randomized_haseman_elston(
    linarg: LinearARG,
    phenos: np.ndarray,
    num_matvecs: int = 20,
    alpha: float = -1,
    trace_est: str = "hutchinson",
    sampler: str = "normal",
    seed: Optional[Union[int, Generator]] = None,
) -> list[float]:
    """
    Implementation of the RHE algorithm from:
        Pazokitoroudi, A. et al. Efficient variance components analysis across millions of genomes.
        Nat Commun 11, 4020 (2020). https://doi.org/10.1038/s41467-020-17576-9
    Notation below follows Methods section of this paper.

    :param linarg: linear ARG for the genotype matrix
    :param y: phenotype vector
    :param b: number of random vectors to use when estimating Tr(GRM^2)
    :return: the heritability estimate
    """

    N, M = linarg.shape
    if phenos.ndim == 1:
        phenos = phenos.reshape(-1, 1)

    heterozygosity = linarg.allele_frequencies * (1 - linarg.allele_frequencies)
    heterozygosity[heterozygosity == 0] = 1

    # Genetic relatedness matrix
    sigmasq = heterozygosity ** (1 + alpha)
    Z = linarg.normalized
    K = Z @ aslinearoperator(diags(sigmasq / np.sum(sigmasq))) @ Z.T

    generator = np.random.default_rng(seed=seed)
    sampler = _construct_sampler(sampler, generator)
    estimator = _construct_estimator(trace_est)

    # se not used atm, but for some trace estimators (eg xtrace, xnystrace) we can compute it
    grm_trace, grm_sq_trace, se = estimator(K, num_matvecs, sampler)

    # center and standardize
    phenos = phenos - np.mean(phenos, axis=0)
    phenos = phenos / np.std(phenos, axis=0)

    # compute y_j' K y_j for each y_j \in y
    C = np.sum(K.matmat(phenos) * phenos, axis=0)

    # construct linear equations to solve
    LHS = np.array([[grm_sq_trace, grm_trace], [grm_trace, N]])
    RHS = np.vstack([C, N * np.ones_like(C)])
    solution = np.linalg.solve(LHS, RHS)

    # normalize back to h2g space
    heritability = solution[0, :] / (solution[0, :] + solution[1, :])

    # SE estimates?

    return heritability


_Sampler = Callable[[int, int], np.ndarray]


def _construct_sampler(name: str, generator: Generator) -> _Sampler:
    """
    Helper function to return the correct sampling distribution function based on a string
    requires x ~ Dist are E[x] = 0 and E[x x'] = I
    """
    name = str(name).lower()
    sampler = None
    # just close over the generator
    if name in {"normal", "gaussian"}:
        sampler = partial(_normal_sampler, generator=generator)
    elif name in {"sphere", "standardized"}:
        sampler = partial(_sphere_sampler, generator=generator)
    elif name in {"rademacher", "signed"}:
        sampler = partial(_rademacher_sampler, generator=generator)
    else:
        raise ValueError(f"{name} not valid sampler (e.g., 'normal', 'sphere', 'rademacher')")

    return sampler


def _normal_sampler(n: int, k: int, generator: Generator) -> np.ndarray:
    return generator.standard_normal(size=(n, k))


def _sphere_sampler(n: int, k: int, generator: Generator) -> np.ndarray:
    samples = _normal_sampler(n, k, generator)
    return np.sqrt(n) * (samples / np.linalg.norm(samples, axis=0))


def _rademacher_sampler(n: int, k: int, generator: Generator) -> np.ndarray:
    return 2 * generator.binomial(1, 0.5, size=(n, k)) - 1


_TraceEstimator = Callable[[LinearOperator, int, _Sampler], tuple[float, float, dict]]


def _construct_estimator(tr_est: str) -> _TraceEstimator:
    """
    Helper function to return the correct estimator function based on a string
    """
    tr_est = str(tr_est).lower()
    estimator = None
    if tr_est in {"hutchinson", "hutch"}:
        estimator = _hutchinson_estimator
    elif tr_est in {"hutch++", "hutchpp"}:
        estimator = _hutch_pp_estimator
    elif tr_est == "xtrace":
        estimator = _xtrace_estimator
    elif tr_est in {"xnystrace", "xnystrom"}:
        estimator = _xnystrace_estimator
    else:
        raise ValueError(f"{tr_est} not valid estimator (e.g., 'hutchinson', 'hutch++', 'xtrace', 'xnystrace')")

    return estimator


def _hutchinson_estimator(GRM: LinearOperator, k: int, sampler: _Sampler) -> tuple[float, float, dict]:
    n, _ = GRM.shape
    samples = sampler(n, k)

    projected_grm = GRM.matmat(samples)

    # compute E[x' K x]
    trace_grm = np.sum(projected_grm * samples) / k

    # compute E[x' K' K x]
    trace_grm_sq = np.sum(projected_grm**2) / k

    return trace_grm, trace_grm_sq, {}


def _hutch_pp_estimator(GRM: LinearOperator, k: int, sampler: _Sampler) -> tuple[float, float, dict]:
    """
    Hutch++ trace estimator, but generalized to estimate tr(A) and tr(A^2) in an efficient manner.

    This improves upon the variance in the stochastic estimate compared with Hutchinson by
    recognizing that tr(A) = tr(hat(A)) - tr(A - hat(A)), due to linearity of trace, where
    hat(A) is the rank-k approximation to A. We can perform two stoachastic trace estimates,
        1) tr(hat(A)) and 2) tr(A - hat(A))

    Note that tr(A - hat(A)) = tr((I - QQ') A) = tr((I - QQ')A(I - QQ')), where hat(A) = QQ' A is the
    randomized rank-k SVD approximation of A. Since (I - QQ') is a projection matrix, (I - QQ') = (I - QQ')(I - QQ').

    To estimate the trace of 2), we define G := (I - QQ')X, where X is an independent set of rvs and compute
        tr(G' A G).

    Note that to estimate the tr(AA) we can make a change of basis from A to AA by computing a new Q from QR(AQ) as
    well as G = (I - QQ'), followed by,
        1) tr(Q A A Q) = ||A Q||_F^2 = sum(AQ ** 2) and similarly,
        2) tr(G A A G) = ||A G||_F^2 = sum(AG ** 2).
    """
    n, _ = GRM.shape
    m = k // 3

    samples = sampler(n, 2 * m)
    X1 = samples[:, :m]
    X2 = samples[:, m:]

    Y = GRM.matmat(X1)

    # compute Q, _ = QR(Y) (orthogonal matrix)
    Q, _ = np.linalg.qr(Y)

    # compute G = X2 - Q @ (Q.T @ X2)
    G = X2 - Q @ (Q.T @ X2)

    AQ = GRM.matmat(Q)
    AG = GRM.matmat(G)
    # estimate trace = tr(Q.T @ A @ Q) + tr(G.T @ A @ G) / m
    trace_grm = np.sum(AQ * Q) + np.sum(AG * G) / m

    # we can't reuse the same basis due to scaling diff in eigenvalues between A and AA
    # but we can compute new basis cheaply from AQ
    Q, _ = np.linalg.qr(AQ)
    AQ = GRM.matmat(Q)

    # update G = X2 - Q @ (Q.T @ X2)
    G = X2 - Q @ (Q.T @ X2)
    AG = GRM.matmat(G)
    # estimate trace = tr(Q.T @ A @ A @ Q) + tr(G.T @ A @ A @ G) / m
    trace_grm_sq = np.sum(AQ**2) + np.sum(AG**2) / m

    return trace_grm, trace_grm_sq, {}


def _xtrace_estimator(
    GRM: LinearOperator, k: int, sampler: _Sampler, estimate_diag: bool = False
) -> tuple[float, float, dict]:
    # WIP
    raise NotImplementedError("xtrace estimator is not yet implemented")
    n, _ = GRM.shape
    m = k // 2

    samples = sampler(n, m)

    Y = GRM.matmat(samples)

    # compute Q, _ = QR(Y) (orthogonal matrix)
    Q, R = np.linalg.qr(Y)

    # solve and rescale
    S = np.linalg.inv(R).T
    S = S / np.linalg.norm(S, axis=0)

    # working variables
    Z = GRM.matmat(Q)
    H = Q.T @ Z
    W = Q.T @ samples
    T = Z.T @ samples
    HW = H @ W

    SW_d = np.sum(S * W, axis=0)
    TW_d = np.sum(T * W, axis=0)
    SHS_d = np.sum(S * (H @ S), axis=0)
    WHW_d = np.sum(W * HW, axis=0)

    term1 = SW_d * np.sum((T - H.T @ W) * S, axis=0)
    term2 = (np.abs(SW_d) ** 2) * SHS_d
    term3 = np.conjugate(SW_d) * np.sum(S * (R - HW), axis=0)

    estimates = np.full(m, np.trace(H)) - SHS_d + (WHW_d - TW_d + term1 + term2 + term3)  # * scale
    trace_grm = np.mean(estimates)
    trace_grm_sq = np.mean(estimates)  # TODO WRONG WRONG; just placeholder
    std_err = np.std(estimates) / np.sqrt(m)

    if estimate_diag:
        return estimates

    # TODO: WIP
    """
    G = samples - Q @ W
    AG = GRM.matmat(G)
    WPA = Y.T - W.T @ Z.T
    QS = Q @ S
    np.sum(AG * G, axis=0)
    """
    return trace_grm, trace_grm_sq, {"std.err": std_err}


def _xnystrace_estimator(GRM: LinearOperator, k: int, sampler: _Sampler) -> tuple[float, float, dict]:
    n, _ = GRM.shape
    m = k // 2

    samples = sampler(n, m)

    Y = GRM.matmat(samples)

    # shift for numerical issues
    nu = np.finfo(Y.dtype).eps * np.linalg.norm(Y, "fro") / np.sqrt(n)
    Y = Y + samples * nu
    Q, R = np.linalg.qr(Y)

    # compute and symmetrize H, then take cholesky factor
    H = samples.T @ Y
    L = np.linalg.cholesky(0.5 * (H + H.T))

    # Nystrom approx is Q @ B @ B' Q'
    B = sp.linalg.solve_triangular(L, R.T, lower=True)

    W = Q.T @ samples

    # invert L here, to simplify a few downstream calculations
    invL = sp.linalg.solve_triangular(L, np.eye(m), lower=True)

    # e_i ' inv(H) e_i
    denom = np.sum(invL**2, axis=1)

    # B' = R @ inv(L) => B' @ inv(L) = R @ inv(H)
    RinvH = B.T @ invL

    # X' @ Q @ R @ inv(H)
    WtRinvH = W.T @ RinvH

    # compute tr of leave-one-out low-rank nystrom approximation
    low_rank_est = np.sum(B**2) - np.sum(RinvH**2, axis=0) / denom

    # compute hutchinson tr estimator on the residuals between A and leave-one-out nsytrom approx
    # okay this took me a while to figure out, but in hindsight is ezpz. :D
    # residuals = diag[X'(A - hat(A)_i)X] = diag[X'(A - hat(A) + rank_one_term)X]
    #  = diag[X'(A - A X inv(H) X' A)X] + diag[X' rank_one_term X ]
    # Notice that the first diag term cancels out due to,
    #  = X' A X - X ' A X inv(H) X' A X = X' A X - X ' A X inv(X' A X) X' A X
    #  = X' A X - X' A X = 0
    # the remaining diag term can be computed as,
    # WtRinvH**2 == [X' Q R inv(H) e_i e_i' inv(H) R' Q' X for e_i in I] and rescaled
    resid_est = np.sum(WtRinvH**2, axis=0) / denom

    # combine low-rank nystrom trace estimate plus hutchinson on the nystrom residuals (and epsilon noise term)
    estimates = low_rank_est + resid_est - nu * n
    trace_est = np.mean(estimates)
    trace_std_err = np.std(estimates) / np.sqrt(k)

    # compute low-rank nystrom approx of A^2 re-using existing terms...
    # Recall Q' Q = I_m
    # AA<X> = A A X inv(X' A A X) X' A A = A Q R inv(R' Q' Q R) R' Q' A
    #  = A Q R inv(R' R) R' Q' A
    #  = A Q R inv(R) inv(R') R' Q' A = A Q Q' A
    # this simplifies the rank-one update too
    invRt = sp.linalg.solve_triangular(R.T, np.eye(m), lower=True)
    denom = np.sum(invRt**2, axis=1)

    Z = GRM.matmat(Q)
    Ztilde = Z @ invRt
    low_rank_sq_est = np.sum(Z**2) - np.sum(Ztilde**2, axis=0) / denom

    # A similar argument regarding the residuals can be made as above. Namely,
    # residuals = diag[X'(AA - hat(AA)_i)X] = diag[X'(AA - hat(AA) + rank_one_term)X]
    #  = diag[X'(AA - AA X inv(X' AA X) X' AA)X] + diag[X' rank_one_term X ]
    # Notice that the first diag term cancels out due to,
    #  = X'A AX - X'A AX inv(X' AA X) X'A AX = R'Q'QR - R'Q'Q R = R'R - R'R = 0
    # interestingly however, the final rank_one diag term reduces substantially to diag(invRt).
    # Recall AX = QR, rank_one_term = AQ inv(R) e_i e_i' inv(R) Q'A / e_i' inv(R) e_i, thus (ignoring normalizing term)
    # diag[X' rank_one_term X] = diag[X'AQ inv(R) e_i e_i' inv(R)Q'AX] = diag[R'Q'Qinv(R) e_i e_i' inv(R)Q'QR]
    #  = diag[R inv(R) e_i e_i' inv(R) R] = diag[e_i e_i] = I. Bringing back the normalizing term we have
    #  I / ||inv(R) e_i||^2 = 1 / denom
    resid_sq_est = 1.0 / denom
    sq_estimates = low_rank_sq_est + resid_sq_est - nu * n  # n-sq?
    sq_trace_est = np.mean(sq_estimates)
    sq_trace_std_err = np.std(sq_estimates) / np.sqrt(k)

    return trace_est, sq_trace_est, {"tr.std.err": trace_std_err, "sq.tr.std.err": sq_trace_std_err}
