# pattern: Mixed (unavoidable)
# Reason: This module mirrors the public RHE orchestration contract while
# keeping the estimator numerics in JAX. Polars ingress and final DataFrame
# materialization remain host-side boundary work.

from __future__ import annotations

import logging
import os

from collections.abc import Callable
from functools import partial
from typing import Any, cast, Optional, Union

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import polars as pl

from jax.scipy.linalg import solve_triangular
from jaxtyping import Array
from numpy.random import Generator

from linear_dag.core.alignment import get_iid_alignment, IidAlignment
from linear_dag.core.jaxlinarg import JaxGRMOperator, JaxParallelOperator
from linear_dag.core.jaxlinarg.grm import _is_packed_operator

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

    _validate_rhe_columns(data, pheno_cols, covar_cols)
    prepared = _prepare_jax_rhe_inputs(grm, data, pheno_cols, covar_cols)
    _validate_num_matvecs(num_matvecs, prepared.yresid.shape[0], trace_est)

    operator = _build_residualized_operator(
        grm,
        prepared.alignment,
        prepared.covariates,
    )
    estimator = _construct_estimator(trace_est)
    sample = _construct_sampler(sampler, _coerce_generator(seed), dtype=prepared.yresid.dtype)

    grm_trace, grm_sq_trace, _se = estimator(operator.matmat, operator.shape[0], num_matvecs, sample)

    yresid = prepared.yresid
    C = jnp.sum(operator.matmat(yresid) * yresid, axis=0)
    N_j = jnp.sum(yresid * yresid, axis=0)

    identity_trace = jnp.asarray(operator.residual_rank, dtype=yresid.dtype)
    lhs = jnp.array([[grm_sq_trace, grm_trace], [grm_trace, identity_trace]], dtype=yresid.dtype)
    rhs = jnp.vstack([C, N_j])
    solution = jnp.linalg.solve(lhs, rhs)

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
    s2e = solution[1, :] * identity_trace
    heritability = s2g / (s2g + s2e)
    var_s2g = (grm_trace**2) * var_s2g
    var_s2e = (identity_trace**2) * var_s2e
    covariances = (grm_trace * identity_trace) * covariances

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

    def __new__(cls, alignment: IidAlignment, yresid: Array, covariates: Array):
        return tuple.__new__(cls, (alignment, yresid, covariates))

    @property
    def alignment(self) -> IidAlignment:
        return self[0]

    @property
    def yresid(self) -> Array:
        return self[1]

    @property
    def covariates(self) -> Array:
        return self[2]


class _ResidualizedJaxGRM(eqx.Module):
    """Private explicit carrier for one IID-aligned, residualized JAX GRM."""

    grm: Any
    left_indices: Array
    right_indices: Array
    basis: Array
    shape: tuple[int, int] = eqx.field(static=True)
    residual_rank: int = eqx.field(static=True)
    use_blockwise_grm: bool = eqx.field(static=True)
    use_explicit_packed_grm: bool = eqx.field(static=True)

    def __check_init__(self) -> None:
        if self.left_indices.ndim != 1 or self.right_indices.ndim != 1:
            raise ValueError("IID alignment indices must be rank-1 arrays")
        if self.left_indices.shape != self.right_indices.shape:
            raise ValueError("IID alignment index arrays must have the same shape")
        if self.basis.ndim != 2 or self.basis.shape[0] != self.shape[0]:
            raise ValueError("Covariate basis rows must match residualized GRM dimensions")
        if self.shape[0] != self.shape[1]:
            raise ValueError("Residualized JAX GRM must be square")
        if self.residual_rank != self.shape[0] - self.basis.shape[1]:
            raise ValueError("Residual rank must equal sample count minus covariate rank")
        if self.grm.shape[0] != self.grm.shape[1]:
            raise ValueError("Residualized JAX GRM requires a square base operator")
        if self.use_blockwise_grm and self.use_explicit_packed_grm:
            raise ValueError("Packed GRM execution cannot use the exact-ragged blockwise fallback")

    def matmat(self, values: Array) -> Array:
        """Apply the residualized GRM while keeping packed graph state explicit."""
        if self.use_explicit_packed_grm:
            return _residualized_jax_grm_matmat(self, values)
        return _residualized_jax_grm_matmat_impl(self, values)


@jax.jit
def _residualized_jax_grm_matmat(operator: _ResidualizedJaxGRM, values: Array) -> Array:
    """JIT one packed residualized GRM call with every array as an operand."""
    return _residualized_jax_grm_matmat_impl(operator, values)


def _residualized_jax_grm_matmat_impl(operator: _ResidualizedJaxGRM, values: Array) -> Array:
    matrix = jnp.asarray(values, dtype=operator.grm.dtype)
    if matrix.ndim == 1:
        matrix = matrix.reshape(-1, 1)
        was_vector = True
    elif matrix.ndim == 2:
        was_vector = False
    else:
        raise ValueError(f"expected rank 1 or 2 input, got rank {matrix.ndim}")
    if matrix.shape[0] != operator.shape[0]:
        raise ValueError(f"expected leading dimension {operator.shape[0]}, got {matrix.shape[0]}")

    projected = _project_off_covariates(matrix, operator.basis)
    merged = projected[operator.left_indices]
    haplotype_values = (
        jnp.zeros(
            (operator.grm.shape[0], matrix.shape[1]),
            dtype=matrix.dtype,
        )
        .at[operator.right_indices]
        .add(merged)
    )
    if operator.use_blockwise_grm:
        haplotype_result = operator.grm.matmat_blockwise(haplotype_values)
    else:
        haplotype_result = operator.grm.matmat(haplotype_values)
    merged_result = haplotype_result[operator.right_indices]
    result = jnp.zeros_like(matrix).at[operator.left_indices].add(merged_result)
    result = _project_off_covariates(0.5 * result, operator.basis)
    return result[:, 0] if was_vector else result


def _project_off_covariates(values: Array, basis: Array) -> Array:
    if basis.shape[1] == 0:
        return values
    return values - basis @ (basis.T @ values)


def _validate_rhe_columns(data: pl.LazyFrame, pheno_cols: list[str], covar_cols: list[str]) -> None:
    if not pheno_cols:
        raise ValueError("pheno_cols must contain at least one phenotype column")
    if not covar_cols:
        raise ValueError("covar_cols must contain at least one covariate column")
    schema_names = set(data.collect_schema().names())
    missing = [name for name in ("iid", *pheno_cols, *covar_cols) if name not in schema_names]
    if missing:
        missing_names = ", ".join(dict.fromkeys(missing))
        raise ValueError(f"RHE data is missing required column(s): {missing_names}")


def _validate_rhe_values(data: pl.LazyFrame, pheno_cols: list[str], covar_cols: list[str]) -> None:
    intercept_is_valid = (
        data.select((pl.col(covar_cols[0]).cast(pl.Float64, strict=False) == 1.0).fill_null(False).all())
        .collect()
        .item()
    )
    if not intercept_is_valid:
        raise ValueError("First column of covariates should be all-ones")
    phenotype_has_value = (
        data.select(
            [
                (
                    pl.col(name).cast(pl.Float64, strict=False).is_not_null()
                    & pl.col(name).cast(pl.Float64, strict=False).is_not_nan()
                )
                .any()
                .alias(name)
                for name in pheno_cols
            ]
        )
        .collect()
        .row(0)
    )
    if not all(phenotype_has_value):
        raise ValueError("Each phenotype must have at least one non-missing value")


def _prepare_jax_rhe_inputs(
    grm: JaxGRMOperator,
    data: pl.LazyFrame,
    pheno_cols: list[str],
    covar_cols: list[str],
) -> _PreparedRHEInputs:
    grm_iids = _require_grm_iids(grm)
    alignment = get_iid_alignment(data.select("iid").cast(pl.Utf8).collect().to_series(), grm_iids.cast(pl.Utf8))
    _validate_diploid_alignment(alignment)
    _validate_rhe_values(data, pheno_cols, covar_cols)

    phenotypes = data.select(pheno_cols).collect().to_numpy(writable=True)
    covariates = data.select(covar_cols).collect().to_numpy(writable=True)

    grm_dtype = cast(Any, grm.dtype)
    phenotypes_jax = jnp.asarray(phenotypes, dtype=grm_dtype)
    covariates_jax = jnp.asarray(covariates, dtype=grm_dtype)
    yresid, covariates_jax = _prep_for_h2_estimation_jax(phenotypes_jax, covariates_jax)
    return _PreparedRHEInputs(alignment, yresid, covariates_jax)


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


def _prep_for_h2_estimation_jax(phenotypes: Array, covariates: Array) -> tuple[Array, Array]:
    if not bool(jnp.allclose(covariates[:, 0], 1.0)):
        raise ValueError("First column of covariates should be all-ones")

    covariates = _impute_missing_with_mean_jax(covariates)
    is_missing = jnp.isnan(phenotypes)
    num_nonmissing = jnp.sum(~is_missing, axis=0)
    if bool(jnp.any(num_nonmissing == 0)):
        raise ValueError("Each phenotype must have at least one non-missing value")

    phenotypes = jnp.where(is_missing, 0.0, phenotypes)
    yresid = _residualize_phenotypes_jax(phenotypes, covariates, is_missing)
    yresid = yresid / jnp.sqrt(jnp.sum(yresid**2, axis=0) / num_nonmissing)
    return yresid, covariates


def _impute_missing_with_mean_jax(values: Array) -> Array:
    col_means = jnp.nanmean(values, axis=0, keepdims=True)
    return jnp.where(jnp.isnan(values), col_means, values)


def _residualize_phenotypes_jax(phenotypes: Array, covariates: Array, phenotypes_missing: Array) -> Array:
    beta = _backslash_jax(covariates, phenotypes)
    missing_fraction = jnp.mean(phenotypes_missing, axis=0, keepdims=True)
    residuals = phenotypes - (covariates @ beta) / (1 - missing_fraction)
    residuals = jnp.where(phenotypes_missing, 0.0, residuals)

    for i in range(phenotypes.shape[1]):
        nonmissing = np.asarray(~phenotypes_missing[:, i])
        if np.mean(nonmissing) >= 1.0:
            continue
        beta_i = _backslash_jax(covariates[nonmissing, :], phenotypes[nonmissing, i])
        residual_i = phenotypes[nonmissing, i] - covariates[nonmissing, :] @ beta_i
        residuals = residuals.at[nonmissing, i].set(residual_i)
    return residuals


def _backslash_jax(A: Array, b: Array, lam: float = 1e-5) -> Array:
    return jnp.linalg.pinv(A.T @ A, rtol=lam) @ (A.T @ b)


def _orthonormal_covariate_basis(covariates: Array) -> Array:
    q, r = jnp.linalg.qr(covariates, mode="reduced")
    diag = jnp.abs(jnp.diag(r))
    if diag.size == 0:
        return q[:, :0]
    tol = jnp.finfo(r.dtype).eps * max(covariates.shape) * jnp.max(diag)
    rank = int(jnp.sum(diag > tol))
    return q[:, :rank]


def _build_residualized_operator(
    grm: JaxGRMOperator,
    alignment: IidAlignment,
    covariates: Array,
) -> _ResidualizedJaxGRM:
    use_blockwise_grm = _should_use_blockwise_grm(grm)
    use_explicit_packed_grm = isinstance(grm, JaxGRMOperator) and _is_packed_operator(grm.operator)
    numerical_grm = (
        JaxGRMOperator(grm.operator, alpha=grm.alpha, center=grm.center) if isinstance(grm, JaxGRMOperator) else grm
    )
    basis = _orthonormal_covariate_basis(covariates)
    shape = (alignment.n_left, alignment.n_left)
    return _ResidualizedJaxGRM(
        grm=numerical_grm,
        left_indices=jnp.asarray(alignment.left_indices, dtype=jnp.int32),
        right_indices=jnp.asarray(alignment.right_indices, dtype=jnp.int32),
        basis=basis,
        shape=shape,
        residual_rank=shape[0] - basis.shape[1],
        use_blockwise_grm=use_blockwise_grm,
        use_explicit_packed_grm=use_explicit_packed_grm,
    )


def _should_use_blockwise_grm(grm: JaxGRMOperator) -> bool:
    setting = os.environ.get(_BLOCKWISE_GRM_ENV, "1").strip().lower()
    return (
        setting not in _FALSE_ENV_VALUES
        and isinstance(grm, JaxGRMOperator)
        and isinstance(grm.operator, JaxParallelOperator)
    )


_Sampler = Callable[[int, int], Array]
_TraceEstimator = Callable[[Callable[[Array], Array], int, int, _Sampler], tuple[Array, Array, dict]]


def _coerce_generator(seed: Optional[Union[int, Generator]]) -> Generator:
    if isinstance(seed, Generator):
        return seed
    return np.random.default_rng(seed=seed)


def _construct_sampler(name: str, generator: Generator, *, dtype) -> _Sampler:
    name = str(name).lower()
    if name in {"normal", "gaussian"}:
        return partial(_normal_sampler, generator=generator, dtype=dtype)
    if name in {"sphere", "standardized"}:
        return partial(_sphere_sampler, generator=generator, dtype=dtype)
    if name in {"rademacher", "signed"}:
        return partial(_rademacher_sampler, generator=generator, dtype=dtype)
    raise ValueError(f"{name} not valid sampler (e.g., 'normal', 'sphere', 'rademacher')")


def _normal_sampler(n: int, k: int, generator: Generator, dtype) -> Array:
    return jnp.asarray(generator.standard_normal(size=(n, k)), dtype=dtype)


def _sphere_sampler(n: int, k: int, generator: Generator, dtype) -> Array:
    samples = _normal_sampler(n, k, generator, dtype)
    return jnp.sqrt(jnp.asarray(n, dtype=samples.dtype)) * (samples / jnp.linalg.norm(samples, axis=0))


def _rademacher_sampler(n: int, k: int, generator: Generator, dtype) -> Array:
    return jnp.asarray(2 * generator.binomial(1, 0.5, size=(n, k)) - 1, dtype=dtype)


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
