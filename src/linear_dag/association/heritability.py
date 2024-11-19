# heritability.py
from functools import partial
from typing import Callable, Optional

import numpy as np

from numpy.random import Generator
from scipy.sparse import diags
from scipy.sparse.linalg import aslinearoperator, LinearOperator

from ..core import LinearARG


def randomized_haseman_elston(
    linarg: LinearARG,
    ys: np.ndarray,
    B: int = 20,
    alpha: float = -1,
    trace_est: str = "hutchinson",
    sampler: str = "normal",
    seed: Optional[int] = None,
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
    grm_trace, grm_sq_trace, se = estimator(K, B, sampler)

    # center and standardize
    ys = ys - np.mean(ys, axis=0)
    ys = ys / np.std(ys, axis=0)

    # compute y_j' K y_j for each y_j \in y
    C = np.sum(K.matmat(ys) ** 2, axis=0)

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


_TraceEstimator = Callable[[LinearOperator, int, _Sampler], float]


def _construct_estimator(tr_est: str):
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
    elif tr_est == "xnystrace" or tr_est == "xnystrom":
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

    # estimate trace = tr(Q.T @ A @ Q) + tr(G.T @ A @ G) / k
    AQ = GRM.matmat(Q)
    AG = GRM.matmat(G)
    trace_grm = np.sum(AQ * Q) + np.sum(AG * G) / (G.shape[1])

    # we should be able to re-use the same basis Q, since (GRM @ GRM @ X1) spans the same basis
    # but with squared eigenvalues
    trace_grm_sq = np.sum(AQ**2) + np.sum(AG**2) / (G.shape[1])

    return trace_grm, trace_grm_sq, {}


def _xtrace_estimator(GRM: LinearOperator, k: int, sampler: _Sampler) -> tuple[float, float, dict]:
    # WIP
    raise NotImplementedError("xnystrace_estimator is not yet implemented")
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

    return trace_grm, trace_grm_sq, {"std.err": std_err}


def _xnystrace_estimator(GRM: LinearOperator, k: int, sampler: _Sampler) -> tuple[float, float, dict]:
    raise NotImplementedError("xnystrace_estimator is not yet implemented")
