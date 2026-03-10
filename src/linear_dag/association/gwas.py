import logging
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


def _validate_non_hwe_genotypes(genotypes: LinearOperator) -> None:
    n_individuals = getattr(genotypes, "n_individuals", None)
    has_iids = getattr(genotypes, "iids", None) is not None
    has_heterozygote_counter = callable(getattr(genotypes, "number_of_heterozygotes", None))

    if n_individuals is None or not has_iids or not has_heterozygote_counter:
        raise ValueError(
            "If assume_hwe is False, genotypes must expose n_individuals, iids, and number_of_heterozygotes()."
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute GWAS effect sizes and standard errors in per-allele units.

    Let $X$ denote the genotype operator in shared sample space, $Y$ denote
    residualized phenotypes, and $C$ denote covariates.

    !!! info

        The first column of `covariates` must be an all-ones intercept.
        If `num_heterozygotes` is `None`, denominator terms are computed under
        Hardy-Weinberg assumptions. If `in_place_op=True`, `genotypes` must be
        [`linear_dag.core.parallel_processing.ParallelOperator`][].

    **Arguments:**

    - `genotypes`: Genotype operator over samples-by-variants.
    - `right_op`: Merge operator mapping shared IID space into genotype IID space.
    - `y_resid`: Residualized phenotype matrix.
    - `covariates`: Covariate matrix with intercept in first column.
    - `num_nonmissing`: Optional non-missing sample counts per phenotype.
    - `y_nonmissing`: Optional indicator matrix for phenotype missingness.
    - `num_heterozygotes`: Optional per-variant heterozygote counts for non-HWE paths.
    - `in_place_op`: Whether to use in-place reverse matmul on parallel operators.
    - `logger`: Optional logger for diagnostics.

    **Returns:**

    - Tuple `(beta, var_numerator, var_denominator, allele_counts)` where:
      - `beta` is per-variant per-trait effect size.
      - `var_numerator` is per-trait residual variance numerator.
      - `var_denominator` is per-variant denominator term.
      - `allele_counts` is per-variant alternate-allele count.

    **Raises:**

    - `ValueError`: If covariates are missing an intercept or if
      `in_place_op=True` is requested without a
      [`linear_dag.core.parallel_processing.ParallelOperator`][].
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

    # avoid numerical issues for variants with no variance
    numerically_zero = 1e-4  # typical nonzero values > 1
    denominator[denominator < numerically_zero] = np.nan
    beta /= denominator

    if logger:
        logger.info("got beta")

    var_numerator = np.sum(y_concat[:, :num_traits] ** 2, axis=0) / (num_nonmissing - 2 * num_covariates).astype(
        np.float32
    )

    assert y_concat.dtype == np.float32
    assert var_numerator.dtype == np.float32
    if logger:
        logger.info("got var numerator")
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
        exprs.append((pl.lit(float(var_numerator[k])) / pl.col(denom_col)).sqrt().cast(pl.Float32).alias(se_name))
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
    """Run a covariate-adjusted linear-regression GWAS association scan.

    Let $Y$ denote phenotype columns and $C$ denote covariates. This function
    aligns IIDs between `data` and `genotypes`, residualizes $Y$ on $C$, and
    computes per-variant summary statistics.

    !!! info

        `data` must include `iid`, phenotype columns, and covariate columns.
        The first covariate column must be an all-ones intercept. If
        `assume_hwe=False`, `genotypes` must provide explicit heterozygote
        counts (for example via individual-node-enabled LinearARG paths).

    **Arguments:**

    - `genotypes`: Genotype operator supporting matmul/rmatmul operations.
    - `data`: Phenotype/covariate table keyed by `iid`.
    - `pheno_cols`: Phenotype column names to scan.
    - `covar_cols`: Covariate column names (first column is intercept).
    - `variant_info`: Optional variant metadata lazy frame to augment outputs.
    - `assume_hwe`: Whether to use HWE-based denominator handling.
    - `recompute_AC`: Whether to recompute allele-count terms under missingness.
    - `in_place_op`: Whether to use in-place operator paths when available.
    - `detach_arrays`: Whether to copy result arrays before LazyFrame assembly.
    - `logger`: Optional logger for diagnostics.

    **Returns:**

    - `polars.LazyFrame` with per-variant summary statistics grouped by trait
      (`<trait>_BETA`, `<trait>_SE`) plus selected variant metadata columns.

    **Raises:**

    - `ValueError`: If intercept validation fails, IID merge has no overlap, or
      non-HWE mode is requested without required genotype capabilities.
    """
    if logger:
        logger.info(f"run_gwas: assume_hwe={assume_hwe}; n_pheno={len(pheno_cols)} n_covar={len(covar_cols)}")

    if not assume_hwe:
        _validate_non_hwe_genotypes(genotypes)

    selected_data = data.select(["iid", *pheno_cols, *covar_cols]).collect()
    data_iids = selected_data.select("iid").cast(pl.Utf8).to_series()

    if not np.allclose(selected_data.select(covar_cols[0]).to_numpy(), 1.0):
        raise ValueError("First column of covar_cols should be '1'")

    left_op, right_op = get_inner_merge_operators(
        data_iids, genotypes.iids
    )  # data iids to shared iids, shared iids to genotypes iids
    if left_op.shape[1] == 0:
        raise ValueError("Merge failed between genotype and phenotype data")

    if assume_hwe:
        num_heterozygotes = None
    else:
        # assumes diploid
        data_iid_array = data_iids.to_numpy().astype(str, copy=False)
        genotype_iids = np.asarray(genotypes.iids[::2]).astype(str, copy=False)
        individuals_to_include = np.isin(genotype_iids, data_iid_array)
        num_heterozygotes = genotypes.number_of_heterozygotes(individuals_to_include)
    if logger:
        logger.info(f"carrier handling: {'HWE assumed' if assume_hwe else 'using explicit num_heterozygotes'}")

    phenotypes = selected_data.select(pheno_cols).cast(pl.Float32).to_numpy()
    covariates = selected_data.select(covar_cols).cast(pl.Float32).to_numpy()
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
        pl.Series("A1FREQ", allele_counts.ravel() / genotypes.shape[0]).cast(pl.Float32)
    )

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


def simple_gwas(
    genotypes: np.ndarray, phenotypes: np.ndarray, covariates: np.ndarray, ploidy: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    """Compute a simple dense GWAS baseline for testing.

    Let $X$ denote genotype columns, $y$ denote phenotype values, and $C$
    denote covariates. The function residualizes both $X$ and $y$ on $C$ and
    computes ordinary least-squares effect sizes and standard errors per
    variant.

    !!! info

        This helper is intended for test/reference comparisons. `ploidy`
        controls effective degrees-of-freedom scaling for the standard-error
        denominator in haploid-vs-diploid interpretations.

    **Arguments:**

    - `genotypes`: Dense genotype matrix.
    - `phenotypes`: Dense phenotype matrix.
    - `covariates`: Dense covariate matrix.
    - `ploidy`: Ploidy scaling factor for standard-error degrees of freedom.

    **Returns:**

    - Tuple `(beta, se)` where each array is shaped
      `(n_variants, n_traits)`.
    """

    nan_rows = np.isnan(phenotypes[:, 0]).ravel()
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

    s_yy = np.sum(yr * yr, axis=0) / (yr.shape[0] - ploidy * C.shape[1])
    se = np.sqrt(s_yy / XtX[:, None])

    return beta, se
