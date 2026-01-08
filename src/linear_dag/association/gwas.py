import logging
from operator import concat
import time

from typing import Optional

import numpy as np
import polars as pl

from scipy.sparse.linalg import LinearOperator

from linear_dag.core.parallel_processing import ParallelOperator

from ..core.operators import get_inner_merge_operators
from .util import (
    get_genotype_variance_explained,
    get_genotype_variance_explained_recompute_AC,
    impute_missing_with_mean,
    residualize_phenotypes,
)


def get_gwas_beta_se(
    genotypes: LinearOperator,
    right_op: LinearOperator,
    y_resid: np.ndarray,
    covariates: np.ndarray,
    num_nonmissing: np.ndarray | None = None,
    y_nonmissing: np.ndarray | None = None,
    num_heterozygotes: np.ndarray | None = None,
    in_place_op: bool = False,
    logger: Optional[logging.Logger] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute GWAS effect sizes and SEs in per-allele units.
    - Assumes the first column of `covariates` is all-ones.
    - If `num_heterozygotes` is None, assumes HWE for the denominator.

    Returns (beta, se, allele_counts).
    """
    if logger:
        logger.info(
            f"get_gwas_beta_se: in_place_op={in_place_op}; y_resid={y_resid.shape} covariates={covariates.shape}",
        )
    if not np.allclose(covariates[:, 0], 1):
        raise ValueError("First column of covariates should be all-ones")

    handle_missingness = y_nonmissing is not None
    if handle_missingness:
        if logger:
            logger.info(f"y_nonmissing shape={y_nonmissing.shape}")
        num_nonmissing = np.sum(y_nonmissing, axis=0)
    else:
        y_nonmissing = np.zeros((y_resid.shape[0], 0), dtype=y_resid.dtype)

    if num_nonmissing is None:
        num_nonmissing = y_resid.shape[0] * np.ones(y_resid.shape[1])

    num_covariates = covariates.shape[1]
    num_traits = y_resid.shape[1]

    y_concat = np.concatenate((y_resid, covariates, y_nonmissing), axis=1, dtype=np.float32)
    if in_place_op:
        if not isinstance(genotypes, ParallelOperator):
            raise ValueError("in_place_op=True requires genotypes to be a ParallelOperator")
        Xty = genotypes._rmatmat(right_op.T @ y_concat, in_place=True)
    else:
        genotypes = right_op @ genotypes
        Xty = genotypes.T @ y_concat

    if logger:
        logger.info(f"Xty shape={Xty.shape}")

    # Operate in place on possibly large array
    beta = Xty[:, :num_traits]
    
    print('Xty:')
    print(f' 4134685: {beta[4134685+85278662]}')
    print(f' 5487797: {beta[5487797+85278662]}')
    print(f' 7811929: {beta[7811929+85278662]}')
    
    # beta *= (right_op.shape[0]/num_nonmissing)
    assert np.shares_memory(beta, Xty)
    if logger:
        logger.info(f"beta shape={beta.shape}")

    # Denominator, equal across traits despite different missingness
    if handle_missingness:
        denominator, allele_counts = get_genotype_variance_explained_recompute_AC(
            Xty[:, num_traits:],
            covariates,
            num_heterozygotes,
            num_nonmissing,
        )
    else:
        denominator, allele_counts = get_genotype_variance_explained(
            Xty[:, num_traits:],
            covariates,
            num_heterozygotes,
        )
        
    print('denominator:')
    print(f' 4134685: {denominator[4134685+85278662]}')
    print(f' 5487797: {denominator[5487797+85278662]}')
    print(f' 7811929: {denominator[7811929+85278662]}')
    
    print('allele_counts:')
    print(f' 4134685: {allele_counts[4134685+85278662]}')
    print(f' 5487797: {allele_counts[5487797+85278662]}')
    print(f' 7811929: {allele_counts[7811929+85278662]}')
    
    print('beta:')
    print(f' 4134685: {beta[4134685+85278662]}')
    print(f' 5487797: {beta[5487797+85278662]}')
    print(f' 7811929: {beta[7811929+85278662]}')

    # avoid numerical issues for variants with no variance
    numerically_zero = 1e-4 # typical nonzero values > 1
    denominator[denominator < numerically_zero] = np.nan
    beta /= denominator

    if logger:
        logger.info(f"got beta")

    var_numerator = np.sum(y_concat[:, :num_traits]**2, axis=0) \
        / (num_nonmissing - 2*num_covariates).astype(np.float32)

    assert y_concat.dtype == np.float32
    assert var_numerator.dtype == np.float32
    if logger:
        logger.info(f"got var numerator")
    return beta, var_numerator, denominator, allele_counts


def _format_sumstats(
    beta: np.ndarray,
    var_numerator: np.ndarray,
    var_denominator: np.ndarray,
    variant_info: pl.DataFrame,
    pheno_cols: list[str],
    *,
    detach_arrays: bool = False,
    recompute_AC: bool = False,
    logger: Optional[logging.Logger] = None,
) -> pl.LazyFrame:
    """
    Format summary statistics into a LazyFrame in a memory-aware way:
    - store BETA and VAR_DENOMINATOR as columns
    - compute SE lazily from expressions
    """

    # Optionally detach from any shared/in-place buffers to avoid cross-iteration aliasing
    # in repeat-covar mode. Keep Fortran order to match expected layout.
    if detach_arrays:
        beta = np.array(beta, dtype=np.float32, order="F", copy=True)
        var_denominator = np.array(var_denominator, dtype=np.float32, copy=True)

    # Start from variant_info and add columns lazily; avoid concatenation
    df = pl.LazyFrame(
        beta,
        schema={f"{name}_BETA": pl.Float32 for name in pheno_cols},
        orient="row",
    )

    # Handle var_denominator: if recompute_AC, create per-trait columns; otherwise single column
    if recompute_AC:
        for k, name in enumerate(pheno_cols):
            df = df.with_columns(pl.Series(f"{name}_VAR_DENOM", var_denominator[:, k]))
    else:
        df = df.with_columns(VAR_DENOMINATOR=pl.lit(var_denominator.ravel()))

    df = pl.concat([variant_info, df], how="horizontal")

    if logger:
        logger.info("Finished creating lazy frame")

    # Add SE lazily; uses beta and var terms, no materialization
    exprs = []
    for k, name in enumerate(pheno_cols):
        se_name = f"{name}_SE"
        denom_col = f"{name}_VAR_DENOM" if recompute_AC else "VAR_DENOMINATOR"
        exprs.append(
            (pl.lit(float(var_numerator[k])) / pl.col(denom_col)).sqrt().cast(pl.Float32).alias(se_name)
        )
    df = df.with_columns(exprs)

    # Order columns grouped by phenotype (variant_info first). Do not include LOG10P by default.
    names_in_order = variant_info.collect_schema().names()
    for name in pheno_cols:
        names_in_order.append(f"{name}_BETA")
        names_in_order.append(f"{name}_SE")
    df = df.select(names_in_order)
    if logger:
        logger.info("Finished formatting sumstats")

    return df


def run_gwas(
    genotypes: LinearOperator,
    data: pl.LazyFrame,
    pheno_cols: list[str],
    covar_cols: list[str],
    variant_info: Optional[pl.LazyFrame] = None,
    assume_hwe: bool = True,
    recompute_AC: bool = False,
    in_place_op: bool = False,
    detach_arrays: bool = False,
    logger: Optional[logging.Logger] = None,
) -> pl.LazyFrame:
    """
    Linear regression association scan with covariates.
    - `data` must contain `iid`, `pheno_cols`, and `covar_cols` (first covariate is all-ones).
    - If `assume_hwe` is False, requires a LinearARG with individual nodes.
    Returns a LazyFrame of per-variant summary statistics.
    """
    if logger:
        logger.info(f"run_gwas: assume_hwe={assume_hwe}; n_pheno={len(pheno_cols)} n_covar={len(covar_cols)}")

    if not assume_hwe and not hasattr(genotypes, "n_individuals"):
        raise ValueError("If assume_hwe is False, genotypes must be a linear ARG with individual nodes.")

    if not np.allclose(data.select(covar_cols[0]).collect().to_numpy(), 1.0):
        raise ValueError("First column of covar_cols should be '1'")

    left_op, right_op = get_inner_merge_operators(
        data.select("iid").cast(pl.Utf8).collect().to_series(), genotypes.iids
    )  # data iids to shared iids, shared iids to genotypes iids
    if left_op.shape[1] == 0:
        raise ValueError("Merge failed between genotype and phenotype data")

    if assume_hwe:
        num_heterozygotes = None
    else:
        # assumes diploid``
        individuals_to_include = np.isin(genotypes.iids[::2], data.select("iid").collect().to_numpy().astype(str))
        num_heterozygotes = genotypes.number_of_heterozygotes(individuals_to_include)
    if logger:
        logger.info(f"carrier handling: {'HWE assumed' if assume_hwe else 'using explicit num_heterozygotes'}")

    phenotypes = data.select(pheno_cols).cast(pl.Float32).collect().to_numpy()
    covariates = data.select(covar_cols).cast(pl.Float32).collect().to_numpy()
    if logger:
        logger.info(f"loaded arrays: phenotypes={phenotypes.shape} covariates={covariates.shape}")

    # Merge to shared sample space and residualize phenotypes
    covariates = left_op.T @ covariates
    phenotypes = left_op.T @ phenotypes
    is_missing = np.isnan(phenotypes)
    num_nonmissing = np.sum(~is_missing, axis=0)
    phenotypes.ravel()[is_missing.ravel()] = 0
    covariates = impute_missing_with_mean(covariates)
    y_resid = residualize_phenotypes(phenotypes, covariates, is_missing)

    t = time.time()
    if logger:
        logger.info(f"starting GWAS computation; in_place_op={in_place_op}")

    beta, var_numerator, var_denominator, allele_counts = get_gwas_beta_se(
        genotypes,
        right_op,
        y_resid,
        covariates,
        num_nonmissing,
        y_nonmissing=~is_missing if recompute_AC else None,
        num_heterozygotes=num_heterozygotes,
        in_place_op=in_place_op,
        logger=logger,
    )
    if logger:
        logger.info(f"GWAS computation finished in {time.time() - t:.3f}s")
    assert beta.dtype == np.float32
    assert var_numerator.dtype == np.float32

    if variant_info is None:
        variant_info = pl.LazyFrame()

    variant_info = variant_info.with_columns(
            pl.Series("A1FREQ", allele_counts.ravel() / genotypes.shape[0]).cast(pl.Float32))

    if num_heterozygotes is not None:
        variant_info = variant_info.with_columns(
            pl.Series("A1_CARRIER_FREQ", num_heterozygotes.astype(np.float32).ravel() * 2 / genotypes.shape[0]).cast(
                pl.Float32
            )
        )

    if logger:
        logger.info(f"variant_info columns={len(variant_info.collect_schema().names())} pheno_cols={len(pheno_cols)}")

    result = _format_sumstats(
        beta,
        var_numerator,
        var_denominator,
        variant_info,
        pheno_cols,
        detach_arrays=detach_arrays,
        recompute_AC=recompute_AC,
        logger=logger,
    )

    if logger:
        logger.info("Finished formatting GWAS results")
    return result


def simple_gwas(genotypes: np.ndarray,
                phenotypes: np.ndarray,
                covariates: np.ndarray,
                ploidy: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """
    Simple GWAS implementation for testing purposes.
    Regresses phenotypes on genotypes residualized on covariates,
    and calculates beta and standard error for each variant. Genotypes
    can be either diploid or haploid. If genotypes are haploid but the
    chromosome is diploid, ploidy should be 2, and d.f. for the standard
    error will be n - p, where p is the number of covariates, instead of
    (2n - p) / 2.
    """

    nan_rows = np.isnan(phenotypes[:,0]).ravel()
    genotypes = genotypes[~nan_rows, :]
    phenotypes = phenotypes[~nan_rows]
    covariates = covariates[~nan_rows, :]

    C = covariates
    CtC_inv = np.linalg.inv(C.T @ C)
    P = C @ CtC_inv @ C.T
    M = np.eye(C.shape[0], dtype=C.dtype) - P

    Xr = M @ genotypes
    yr = M @ phenotypes

    XtY = Xr.T @ yr
    XtX = np.sum(Xr * Xr, axis=0)
    XtX[XtX < 1e-6] = np.nan
    beta = XtY / XtX[:, None]

    s_yy = np.sum(yr * yr, axis=0) / (yr.shape[0] - ploidy*C.shape[1])
    se = np.sqrt(s_yy / XtX[:, None])

    return beta, se
