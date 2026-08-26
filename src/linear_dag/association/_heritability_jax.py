# pattern: Mixed (unavoidable)
# Reason: This module mirrors the public RHE orchestration contract while
# keeping the estimator numerics in JAX. Polars ingress and final DataFrame
# materialization remain host-side boundary work.

from __future__ import annotations

import logging
import os
import warnings

from collections.abc import Callable
from functools import partial
from typing import Any, Optional, Union

import jax
import jax.numpy as jnp
import numpy as np
import polars as pl

from jax.scipy.linalg import solve_triangular
from jaxtyping import Array
from numpy.random import Generator

from linear_dag.core.alignment import get_iid_alignment, IidAlignment
from linear_dag.core.jaxlinarg import JaxGRMOperator

from .util import impute_missing_with_mean, residualize_phenotypes

_BLOCKWISE_GRM_ENV = "LINEAR_DAG_JAX_RHE_BLOCKWISE_GRM"
_FALSE_ENV_VALUES = {"0", "false", "no", "off"}


def randomized_haseman_elston(
    grm: JaxGRMOperator,
    data: pl.LazyFrame,
    pheno_cols: list[str],
    covar_cols: list[str],
    num_matvecs: int = 20,
    trace_est: str = "hutchinson",
    sampler: str = "normal",
    seed: Optional[Union[int, Generator]] = None,
    logger: Optional[logging.Logger] = None,
) -> pl.DataFrame:
    """Estimate SNP heritability with a JAX-backed GRM operator."""

    def _info(msg, *args):
        if logger is not None:
            logger.info(msg, *args)

    _info(
        "randomized_haseman_elston_jax: starting with %d pheno, %d covar, num_matvecs=%d, trace_est=%s, sampler=%s",
        len(pheno_cols),
        len(covar_cols),
        num_matvecs,
        trace_est,
        sampler,
    )

    if not np.allclose(data.select(covar_cols[0]).collect().to_numpy(), 1.0):
        raise ValueError("First column of covar_cols should be '1'")

    prepared = _prepare_jax_rhe_inputs(grm, data, pheno_cols, covar_cols)
    _validate_num_matvecs(num_matvecs, prepared.yresid.shape[0], trace_est)

    operator = _build_aligned_operator(grm, prepared.alignment)
    estimator = _construct_estimator(trace_est)
    sample = _construct_sampler(sampler, _coerce_generator(seed))
    sample = _residualize_sampler(sample, prepared.covariates, dtype=prepared.yresid.dtype)

    grm_trace, grm_sq_trace, _se = estimator(operator.matmat, operator.shape[0], num_matvecs, sample)

    yresid = prepared.yresid
    C = jnp.sum(operator.matmat(yresid) * yresid, axis=0)
    N_j = jnp.sum(yresid * yresid, axis=0)

    identity_trace = jnp.asarray(yresid.shape[0], dtype=yresid.dtype)
    lhs = jnp.array([[grm_sq_trace, grm_trace], [grm_trace, identity_trace]], dtype=yresid.dtype)
    rhs = jnp.vstack([C, N_j])
    solution_host = np.linalg.solve(
        np.asarray(lhs, dtype=np.float64),
        np.asarray(rhs, dtype=np.float64),
    )
    if not np.all(np.isfinite(solution_host)):
        raise np.linalg.LinAlgError("Singular matrix")
    solution = jnp.asarray(solution_host, dtype=yresid.dtype)

    var_s2g, var_s2e, covariances = _compute_err_variance_vectorized(
        operator.matmat,
        yresid,
        solution,
        grm_sq_trace,
        grm_trace,
        num_matvecs,
        identity_trace,
    )

    s2g = solution[0, :] * grm_trace
    s2e = solution[1, :] * N_j
    heritability = s2g / (s2g + s2e)
    var_s2g = (grm_trace**2) * var_s2g
    var_s2e = (N_j**2) * var_s2e
    covariances = (grm_trace * N_j) * covariances

    numer = (s2e**2) * var_s2g + (s2g**2) * var_s2e - 2 * s2g * s2e * covariances
    denom = (s2g + s2e) ** 4
    var_h2g = numer / denom

    return pl.DataFrame(
        {
            "phenotype": pheno_cols,
            "s2g": np.asarray(s2g),
            "s2g.se": np.asarray(jnp.sqrt(var_s2g)),
            "s2e": np.asarray(s2e),
            "s2e.se": np.asarray(jnp.sqrt(var_s2e)),
            "h2g": np.asarray(heritability),
            "h2g.se": np.asarray(jnp.sqrt(var_h2g)),
        }
    )


class _PreparedRHEInputs(tuple):
    __slots__ = ()

    def __new__(cls, alignment: IidAlignment, yresid: Array, covariates: np.ndarray):
        return tuple.__new__(cls, (alignment, yresid, covariates))

    @property
    def alignment(self) -> IidAlignment:
        return self[0]

    @property
    def yresid(self) -> Array:
        return self[1]

    @property
    def covariates(self) -> np.ndarray:
        return self[2]


class _AlignedJaxGRM:
    def __init__(self, base_matmat: Callable[[Array], Array], sample_count: int, *, jit_matmat: bool = True):
        self.shape = (sample_count, sample_count)
        self.matmat = jax.jit(base_matmat) if jit_matmat else base_matmat


def _prepare_jax_rhe_inputs(
    grm: JaxGRMOperator,
    data: pl.LazyFrame,
    pheno_cols: list[str],
    covar_cols: list[str],
) -> _PreparedRHEInputs:
    grm_iids = _require_grm_iids(grm)
    alignment = get_iid_alignment(data.select("iid").cast(pl.Utf8).collect().to_series(), grm_iids.cast(pl.Utf8))
    _validate_diploid_alignment(alignment)

    phenotypes = data.select(pheno_cols).collect().to_numpy(writable=True)
    covariates = data.select(covar_cols).collect().to_numpy(writable=True)

    yresid, covariates = _prep_for_h2_estimation_host(phenotypes, covariates)
    return _PreparedRHEInputs(alignment, jnp.asarray(yresid, dtype=grm.dtype), covariates)


def _require_grm_iids(grm: JaxGRMOperator) -> pl.Series:
    iids = getattr(grm, "iids", None)
    if iids is None:
        iids = getattr(getattr(grm, "operator", None), "iids", None)
    if iids is None:
        raise ValueError("JAX GRM operator must expose iids for RHE alignment")
    if not isinstance(iids, pl.Series):
        iids = pl.Series("iids", iids)
    return iids


def _validate_diploid_alignment(alignment: IidAlignment) -> None:
    left_counts = np.bincount(alignment.left_indices, minlength=alignment.n_left)
    matched_left_counts = left_counts[left_counts != 0]
    if not np.all(matched_left_counts == 2):
        raise ValueError("Each row of the phenotype matrix should match zero or two rows of the genotype operator")

    right_counts = np.bincount(alignment.right_indices, minlength=alignment.n_right)
    matched_right_counts = right_counts[right_counts != 0]
    if not np.all(matched_right_counts == 1):
        raise ValueError("Each row of the genotype operator should match at most one row of the phenotype matrix")

    two_n = np.sum(right_counts > 0)
    if two_n != 2 * np.sum(left_counts > 0):
        raise ValueError("Diploid IID alignment is inconsistent")


def _prep_for_h2_estimation_host(
    phenotypes: np.ndarray,
    covariates: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if not np.allclose(covariates[:, 0], 1.0):
        raise ValueError("First column of covariates should be all-ones")

    covariates = impute_missing_with_mean(covariates)
    is_missing = np.isnan(phenotypes)
    num_nonmissing = np.sum(~is_missing, axis=0)
    if np.any(num_nonmissing == 0):
        raise ValueError("Each phenotype must have at least one non-missing value")

    phenotypes.ravel()[is_missing.ravel()] = 0.0
    yresid = residualize_phenotypes(phenotypes, covariates, is_missing)
    yresid /= np.sqrt(np.sum(yresid**2, axis=0) / num_nonmissing)
    return yresid, covariates


def _build_aligned_operator(
    grm: JaxGRMOperator,
    alignment: IidAlignment,
) -> _AlignedJaxGRM:
    use_blockwise_grm = _should_use_blockwise_grm(grm)
    grm_matmat = grm.matmat_blockwise if use_blockwise_grm else grm.matmat

    def base_matmat(values: Array) -> Array:
        merged = alignment.gather_left_jax(values)
        haplotype_values = alignment.scatter_right_jax(merged)
        haplotype_result = grm_matmat(haplotype_values)
        merged_result = alignment.gather_right_jax(haplotype_result)
        return 0.5 * alignment.scatter_left_jax(merged_result)

    return _AlignedJaxGRM(base_matmat, alignment.n_left, jit_matmat=not use_blockwise_grm)


def _should_use_blockwise_grm(grm: JaxGRMOperator) -> bool:
    setting = os.environ.get(_BLOCKWISE_GRM_ENV, "1").strip().lower()
    return setting not in _FALSE_ENV_VALUES and hasattr(grm, "matmat_blockwise")


_HostSampler = Callable[[int, int], np.ndarray]
_Sampler = Callable[[int, int], Array]
_TraceEstimator = Callable[[Callable[[Array], Array], int, int, _Sampler], tuple[Array, Array, dict]]


def _coerce_generator(seed: Optional[Union[int, Generator]]) -> Generator:
    if isinstance(seed, Generator):
        return seed
    return np.random.default_rng(seed=seed)


def _residualize_sampler(sample: _HostSampler, covariates: np.ndarray, *, dtype: Any) -> _Sampler:
    def residualized_sample(n: int, k: int) -> Array:
        values = sample(n, k)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            coefficients = np.linalg.lstsq(covariates, values)[0]
        return jnp.asarray(values - covariates @ coefficients, dtype=dtype)

    return residualized_sample


def _construct_sampler(name: str, generator: Generator) -> _HostSampler:
    name = str(name).lower()
    if name in {"normal", "gaussian"}:
        return partial(_normal_sampler, generator=generator)
    if name in {"sphere", "standardized"}:
        return partial(_sphere_sampler, generator=generator)
    if name in {"rademacher", "signed"}:
        return partial(_rademacher_sampler, generator=generator)
    raise ValueError(f"{name} not valid sampler (e.g., 'normal', 'sphere', 'rademacher')")


def _normal_sampler(n: int, k: int, generator: Generator) -> np.ndarray:
    return generator.standard_normal(size=(n, k))


def _sphere_sampler(n: int, k: int, generator: Generator) -> np.ndarray:
    samples = _normal_sampler(n, k, generator)
    return np.sqrt(n) * (samples / np.linalg.norm(samples, axis=0))


def _rademacher_sampler(n: int, k: int, generator: Generator) -> np.ndarray:
    return 2 * generator.binomial(1, 0.5, size=(n, k)) - 1


def _construct_estimator(tr_est: str) -> _TraceEstimator:
    tr_est = str(tr_est).lower()
    if tr_est in {"hutchinson", "hutch"}:
        return _hutchinson_estimator
    if tr_est in {"hutch++", "hutchpp"}:
        return _hutch_pp_estimator
    if tr_est in {"xnystrace", "xnystrom"}:
        return _xnystrace_estimator
    raise ValueError(f"{tr_est} not valid estimator (e.g., 'hutchinson', 'hutch++', 'xnystrace')")


def _validate_num_matvecs(num_matvecs: int, sample_count: int, trace_est: str) -> None:
    if not isinstance(num_matvecs, (int, np.integer)):
        raise TypeError(f"num_matvecs must be an integer, got {type(num_matvecs).__name__}")
    if num_matvecs < 1:
        raise ValueError(f"num_matvecs={num_matvecs} must be >= 1")
    if num_matvecs > sample_count:
        raise ValueError(f"num_matvecs={num_matvecs} should be <= N={sample_count}")
    name = str(trace_est).lower()
    if name in {"hutch++", "hutchpp"} and num_matvecs < 3:
        raise ValueError("hutch++ requires num_matvecs >= 3")
    if name in {"xnystrace", "xnystrom"} and num_matvecs < 2:
        raise ValueError("xnystrace requires num_matvecs >= 2")


def _hutchinson_estimator(
    grm_matmat: Callable[[Array], Array],
    n: int,
    k: int,
    sampler: _Sampler,
) -> tuple[Array, Array, dict]:
    samples = sampler(n, k)
    projected_grm = grm_matmat(samples)
    trace_grm = jnp.sum(projected_grm * samples) / k
    trace_grm_sq = jnp.sum(projected_grm**2) / k
    return trace_grm, trace_grm_sq, {}


def _hutch_pp_estimator(
    grm_matmat: Callable[[Array], Array],
    n: int,
    k: int,
    sampler: _Sampler,
) -> tuple[Array, Array, dict]:
    m = k // 3
    if m < 1:
        raise ValueError("hutch++ requires k >= 3")

    samples = sampler(n, 2 * m)
    X1 = samples[:, :m]
    X2 = samples[:, m:]

    Y = grm_matmat(X1)
    Q, _ = jnp.linalg.qr(Y)
    G = X2 - Q @ (Q.T @ X2)

    AQ = grm_matmat(Q)
    AG = grm_matmat(G)
    trace_grm = jnp.sum(AQ * Q) + jnp.sum(AG * G) / m

    Q, _ = jnp.linalg.qr(AQ)
    AQ = grm_matmat(Q)
    G = X2 - Q @ (Q.T @ X2)
    AG = grm_matmat(G)
    trace_grm_sq = jnp.sum(AQ**2) + jnp.sum(AG**2) / m
    return trace_grm, trace_grm_sq, {}


def _xnystrace_estimator(
    grm_matmat: Callable[[Array], Array],
    n: int,
    k: int,
    sampler: _Sampler,
) -> tuple[Array, Array, dict]:
    m = k // 2
    samples = sampler(n, m)

    Y = grm_matmat(samples)
    nu = jnp.finfo(Y.dtype).eps * jnp.linalg.norm(Y, "fro") / jnp.sqrt(jnp.asarray(n, dtype=Y.dtype))
    Y = Y + samples * nu
    Q, R = jnp.linalg.qr(Y)

    H = samples.T @ Y
    L = jnp.linalg.cholesky(0.5 * (H + H.T))
    if not bool(jnp.all(jnp.isfinite(L))):
        raise RuntimeError("Stochastic low-rank GRM is not PSD. Try changing number of mat-vecs")

    B = solve_triangular(L, R.T, lower=True)
    W = Q.T @ samples
    invL = solve_triangular(L, jnp.eye(m, dtype=samples.dtype), lower=True)

    denom = jnp.sum(invL**2, axis=1)
    RinvH = B.T @ invL
    WtRinvH = W.T @ RinvH

    low_rank_est = jnp.sum(B**2) - jnp.sum(RinvH**2, axis=0) / denom
    resid_est = jnp.sum(WtRinvH**2, axis=0) / denom
    estimates = low_rank_est + resid_est - nu * n
    trace_est = jnp.mean(estimates)
    trace_std_err = jnp.std(estimates) / jnp.sqrt(jnp.asarray(k, dtype=samples.dtype))

    invRt = solve_triangular(R.T, jnp.eye(m, dtype=samples.dtype), lower=True)
    denom = jnp.sum(invRt**2, axis=1)

    Z = grm_matmat(Q)
    Ztilde = Z @ invRt
    low_rank_sq_est = jnp.sum(Z**2) - jnp.sum(Ztilde**2, axis=0) / denom
    resid_sq_est = 1.0 / denom
    sq_estimates = low_rank_sq_est + resid_sq_est - nu * n
    sq_trace_est = jnp.mean(sq_estimates)
    sq_trace_std_err = jnp.std(sq_estimates) / jnp.sqrt(jnp.asarray(k, dtype=samples.dtype))

    return trace_est, sq_trace_est, {"tr.std.err": trace_std_err, "sq.tr.std.err": sq_trace_std_err}


def _compute_err_variance_vectorized(
    grm_matmat: Callable[[Array], Array],
    yresid: Array,
    solutions: Array,
    grm_sq_trace: Array,
    grm_trace: Array,
    num_matvecs: int,
    identity_trace: Array,
) -> tuple[Array, Array, Array]:
    Y = yresid
    dtype = Y.dtype

    denom = identity_trace * grm_sq_trace - grm_trace**2
    denom2 = denom * denom
    identity_trace2 = identity_trace * identity_trace

    s2g = solutions[0].astype(dtype)
    s2e = solutions[1].astype(dtype)
    s2g_row = s2g[None, :]
    s2e_row = s2e[None, :]

    KY = grm_matmat(Y)
    Proj = identity_trace * KY - grm_trace * Y
    KProj = grm_matmat(Proj)

    VProj = s2g_row * KProj + s2e_row * Proj
    term1 = 2.0 * jnp.einsum("ij,ij->j", Proj, VProj)

    Tmp = s2g_row * KY + s2e_row * Y
    KTmp = grm_matmat(Tmp)
    Rtmp = identity_trace * KTmp - grm_trace * Tmp
    covUW_hat = 2.0 * jnp.einsum("ij,ij->j", Y, Rtmp)

    term2 = (s2g**2) * (grm_sq_trace / num_matvecs)
    trV2 = (s2g**2) * grm_sq_trace + 2.0 * s2g * s2e * grm_trace + (s2e**2) * identity_trace
    varW_hat = 2.0 * trV2

    var_s2g = (term1 + term2) / denom2
    var_s2e = (
        (grm_trace**2 / (identity_trace2 * denom2)) * term1
        + varW_hat / identity_trace2
        - (2.0 * grm_trace / (identity_trace2 * denom)) * covUW_hat
    )
    cov_ge = covUW_hat / (denom * identity_trace) - (grm_trace / identity_trace) * term1 / denom2
    return var_s2g, var_s2e, cov_ge
