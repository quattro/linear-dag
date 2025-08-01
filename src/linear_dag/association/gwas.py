from typing import Optional

import numpy as np
import polars as pl

from scipy.sparse.linalg import LinearOperator
from scipy.stats import chi2

from ..core.operators import get_inner_merge_operators
from .util import (
    _get_genotype_variance,
    _get_genotype_variance_explained,
    _impute_missing_with_mean,
    residualize_phenotypes,
)


def get_gwas_beta_se(
    genotypes: LinearOperator,
    left_op: LinearOperator,
    right_op: LinearOperator,
    phenotypes: np.ndarray,
    covariates: np.ndarray,
    assume_hwe: bool,
) -> np.ndarray:
    """
    Gets GWAS effect-size estimates and standard errors in per-allele units.
    The first column of covariates should be all-ones.

    Args:
        genotypes: Unnormalized, phased genotypes as a linear operator (e.g. ParallelOperator or LinearARG)
        left_op: Left merge operator for matching genotypes and phenotypes
        right_op: Right merge operator for matching genotypes and phenotypes
        phenotypes: Phenotypes matrix
        covariates: Covariates matrix, which should include the all-ones annotation
        assume_hwe: Whether or not to assume HWE. If not, genotypes must be the ploidy linear ARG

    Returns:
        tuple: (beta, se, sample_size, allele_counts, carrier_counts)
            beta: GWAS effect-size estimates in per-allele units
            se: Standard errors assuming Hardy-Weinberg equilibrium
            sample_size: Number of non-missing samples per trait
            allele_counts: Number of haplotypes carrying each allele
            carrier_counts: Number of individuals carrying each allele
    """
    if not np.allclose(covariates[:, 0], 1):
        raise ValueError("First column of covariates should be all-ones")

    cols_matched_per_row = left_op @ np.ones(left_op.shape[1])
    # Check if all *non-zero* counts are exactly 2
    if not np.all(cols_matched_per_row[cols_matched_per_row != 0] == 2):
        raise ValueError("Each row of the phenotype matrix should match zero or two rows of the genotype operator")

    rows_matched_per_col = np.ones(right_op.shape[0]) @ right_op
    # Check if all *non-zero* counts are exactly 1
    if not np.all(rows_matched_per_col[rows_matched_per_col != 0] == 1):
        raise ValueError("Each row of the genotype operator should match at most one row of the phenotype matrix")

    two_n = np.sum(rows_matched_per_col > 0)
    assert two_n == 2 * np.sum(cols_matched_per_row > 0)

    covariates = left_op.T @ covariates
    phenotypes = left_op.T @ phenotypes

    # Handle missingness
    covariates = _impute_missing_with_mean(covariates)

    is_missing = np.isnan(phenotypes)
    num_nonmissing = np.sum(~is_missing, axis=0)
    phenotypes.ravel()[is_missing.ravel()] = 0

    # Residualize phenotypes on covariates
    y_resid = residualize_phenotypes(phenotypes, covariates, is_missing)
    numerator = (right_op @ genotypes).T @ y_resid / num_nonmissing

    # Get denominator, which is assumed equal across traits despite different missingness
    var_explained, allele_counts = _get_genotype_variance_explained(right_op @ genotypes, covariates)
    if assume_hwe:
        denominator = np.maximum(allele_counts - var_explained, 1e-6) / two_n
        carrier_counts = None
    else:
        if genotypes.sex is not None:
            raise NotImplementedError
        # non-missing individuals to include in carrier count
        individuals_to_include = np.where(rows_matched_per_col[::2] == 1)[0]
        var_genotypes, carrier_counts = _get_genotype_variance(genotypes, allele_counts, individuals_to_include)
        denominator = np.maximum(var_genotypes - var_explained, 1e-6) / two_n
    assert np.all(denominator > 0), min(denominator)

    var_resid = np.sum(y_resid**2, axis=0) / num_nonmissing
    se = np.sqrt(var_resid.reshape(1, -1) / (denominator * num_nonmissing.reshape(1, -1)))

    return (numerator / denominator, se, num_nonmissing // 2, allele_counts, carrier_counts)

def _format_sumstats(
    beta: np.ndarray,
    se: np.ndarray,
    variant_info: pl.LazyFrame,
    pheno_cols: list[str],
    ) -> pl.LazyFrame:

    def log_chisq_pval(z: np.ndarray) -> np.ndarray:
        return -chi2(1).logsf(z**2) / np.log(10)

    data = np.hstack((beta, se, log_chisq_pval(beta / se)))
    names = [f"{name}_{suffix}"  for suffix in ["BETA", "SE", "LOG10P"] for name in pheno_cols]
    
    names_in_order = variant_info.collect_schema().names() + \
        [f"{name}_{suffix}" for name in pheno_cols for suffix in ["BETA", "SE", "LOG10P"]]

    df = pl.LazyFrame(data, schema=names)
    df = pl.concat([variant_info, df], how="horizontal").select(names_in_order)
    return df


def run_gwas(
    genotypes: LinearOperator,
    data: pl.LazyFrame,
    pheno_cols: list[str],
    covar_cols: list[str],
    variant_info: Optional[pl.LazyFrame] = None,
    assume_hwe: bool = True,
) -> pl.LazyFrame:
    """
    Runs a linear regression association scan with covariates.

    Args:
        genotypes: Unnormalized, phased genotypes as ParallelOperator or LinearARG.
            Must have iids for merging with data
        data: Polars LazyFrame containing genotypes and phenotypes, and
            a column named `iid` for merging with genotypes
        pheno_cols: List of columns in data containing phenotypes
        covar_cols: List of columns in data containing covariates
        variant_info: Optional variant information to include in results,
            as a Polars LazyFrame of length equal to genotypes.shape[1]
        assume_hwe: Whether or not to assume HWE. If not, the ploidy ARG
            must be provided so number of carriers per variant can be computed.

    Returns:
        Polars LazyFrame containing GWAS results
    """
    if not assume_hwe and not hasattr(genotypes, "n_individuals"):
        raise ValueError("If assume_hwe is False, genotypes must be a linear ARG with individual nodes.")

    if not np.allclose(data.select(covar_cols[0]).collect().to_numpy(), 1.0):
        raise ValueError("First column of covar_cols should be '1'")
    
    left_op, right_op = get_inner_merge_operators(
        data.select("iid").cast(pl.Utf8).collect().to_series(), genotypes.iids
    )  # data iids to shared iids, shared iids to genotypes iids
    if left_op.shape[1] == 0:
        raise ValueError("Merge failed between genotype and phenotype data")

    phenotypes = data.select(pheno_cols).collect().to_numpy()
    covariates = data.select(covar_cols).collect().to_numpy()

    beta, se, sample_size, allele_counts, carrier_counts = get_gwas_beta_se(
        genotypes, left_op, right_op, phenotypes, covariates, assume_hwe
    )

    if variant_info is None:
        variant_info = pl.LazyFrame()
    variant_info = variant_info.with_columns(
        pl.Series("A1FREQ", allele_counts.ravel() / genotypes.shape[0]).cast(pl.Int64),
    )

    return _format_sumstats(beta, se, variant_info, pheno_cols)
