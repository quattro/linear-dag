from linear_dag.core.operators import get_merge_operator
from scipy.sparse.linalg import LinearOperator
from scipy.stats import norm
import numpy as np
from typing import Optional, Union, Tuple
import polars as pl

def _backslash(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """MATLAB-style backslash"""
    return np.linalg.solve(A.T @ A, A.T @ b)

def _residualize_phenotypes_mar(phenotypes: np.ndarray, covariates: np.ndarray, phenotypes_missing: np.ndarray) -> np.ndarray:
    """ Residualize phenotypes on covariates assuming missing at random (MAR),
    i.e., covariance matrix of the covariates is the same for individuals with and without missing data.
    Input phenotypes should be zero for missing values."""
    beta = _backslash(covariates, phenotypes)
    residuals = phenotypes - (covariates @ beta) / (1 - np.mean(phenotypes_missing, axis=0))
    residuals.ravel()[phenotypes_missing.ravel()] = 0
    return residuals

def residualize_phenotypes(phenotypes: np.ndarray, 
                            covariates: np.ndarray, 
                            phenotypes_missing: np.ndarray,
                            missingness_threshold_mar: float = 0) -> np.ndarray:
    """ Residualize phenotypes on covariates. For phenotypes with missingness <=
    `missingness_threshold_mar, assumes those phenotypes are missing at random
    with respect to covariates."""
    residuals = _residualize_phenotypes_mar(phenotypes, covariates, phenotypes_missing)
    for i in range(phenotypes.shape[1]):
        nonmissing = ~phenotypes_missing[:, i]
        if np.mean(nonmissing) >= 1 - missingness_threshold_mar:
            continue
        beta = _backslash(covariates[nonmissing, :], phenotypes[nonmissing, i])
        residuals[nonmissing, i] = phenotypes[nonmissing, i] - (covariates[nonmissing, :] @ beta)
    return residuals

def _get_genotype_variance_explained(
                                    genotypes: LinearOperator, 
                                    covariates: np.ndarray
                                    ) -> Tuple[np.ndarray, np.ndarray]:
    """ Get variance of genotypes explained by covariates: 
                diag(X'C(C'C)^-1C'X) / n
        where X is the genotype operator, C is the covariate matrix, and diag() extracts the diagonal.
        
        Args:
            genotypes: Unnormalized, phased genotypes as a linear operator (e.g. ParallelOperator or LinearARG)
            covariates: Covariates matrix, which should include the all-ones annotation except for missing values

        Returns:
            tuple: (total_var_explained, allele_fallele_countrequency)
                total_var_explained: Total variance of genotypes explained by covariates
                allele_count: Allele count of the genotypes, assuming first column of covariates is all-ones
                                except for missing values
    """
    covariate_inner = covariates.T @ covariates
    between_product = covariates.T @ genotypes
    covariates_backslash_genotypes = np.linalg.solve(covariate_inner, between_product)

    # diag(A.T @ B) == sum(A * B)
    total_var_explained = np.sum(between_product * covariates_backslash_genotypes, axis=0).reshape(-1,1)

    allele_count = between_product[0,:].reshape(-1,1)
    return total_var_explained, allele_count
    
def _impute_missing_with_mean(data: np.ndarray) -> np.ndarray:
    """Impute missing values with the mean of the column in place."""
    data = data.copy()
    for col in range(data.shape[1]):
        is_missing = np.isnan(data[:, col])
        col_mean = np.mean(data[~is_missing, col])
        data[is_missing, col] = col_mean
    return data


def get_gwas_beta_se(
                genotypes: LinearOperator, 
                merge_operator: LinearOperator, 
                phenotypes: np.ndarray, 
                covariates: np.ndarray,
                variant_info: Optional[pl.LazyFrame] = None,
                ) -> np.ndarray:
    """
    Gets GWAS effect-size estimates and standard errors in per-allele units.
    The first column of covariates should be all-ones.

    Args:
        genotypes: Unnormalized, phased genotypes as a linear operator (e.g. ParallelOperator or LinearARG)
        merge_operator: Merge operator for matching genotypes and phenotypes; shape is
            (phenotypes.shape[0], genotypes.shape[0])
        phenotypes: Phenotypes matrix
        covariates: Covariates matrix, which should include the all-ones annotation

    Returns:
        tuple: (beta, se, sample_size)
            beta: GWAS effect-size estimates in per-allele units
            se: Standard errors assuming Hardy-Weinberg equilibrium
            sample_size: Number of non-missing samples per trait
    """
    if not np.allclose(covariates[:,0], 1):
        raise ValueError("First column of covariates should be all-ones")

    cols_matched_per_row = merge_operator @ np.ones(merge_operator.shape[1])
    # Check if all *non-zero* counts are exactly 2
    if not np.all(cols_matched_per_row[cols_matched_per_row != 0] == 2):
        raise ValueError("Each row of the phenotype matrix should match zero or two rows of the genotype operator")
    
    rows_matched_per_col = np.ones(merge_operator.shape[0]) @ merge_operator
    # Check if all *non-zero* counts are exactly 1
    if not np.all(rows_matched_per_col[rows_matched_per_col != 0] == 1):
        raise ValueError("Each row of the genotype operator should match at most one row of the phenotype matrix")

    two_n = np.sum(rows_matched_per_col>0)
    assert two_n == 2 * np.sum(cols_matched_per_row>0)

    covariates = merge_operator.T @ covariates
    phenotypes = merge_operator.T @ phenotypes
    
    # Handle missingness
    covariates = _impute_missing_with_mean(covariates)
    is_missing = np.isnan(phenotypes)
    num_nonmissing = np.sum(~is_missing, axis=0)
    print(f"num_nonmissing: {num_nonmissing}")
    phenotypes.ravel()[is_missing.ravel()] = 0

    # Residualize phenotypes on covariates
    y_resid = residualize_phenotypes(phenotypes, covariates, is_missing)
    y_resid /= np.sqrt(np.sum(y_resid**2, axis=0) / num_nonmissing) # ||y_resid||^2 == num_nonmissing
    assert np.allclose(np.sum(y_resid**2, axis=0), num_nonmissing), "Non-unit mean squared residuals, indicating a numerical issue; check for collinearity"

    print(f"mean y_resid: {np.mean(y_resid, axis=0)}")
    assert np.allclose(np.mean(y_resid, axis=0), 0, rtol=1e-3), "Non-zero mean residuals, indicating a numerical issue; check for collinearity"
    numerator = genotypes.T @ y_resid / num_nonmissing
    
     # Get denominator, which is assumed equal across traits despite different missingness
    var_explained, allele_counts = _get_genotype_variance_explained(genotypes, covariates)
    denominator = (allele_counts - var_explained + 1e-6) / two_n
    assert np.all(denominator > 0)

    return (numerator / denominator, 
            1 / (np.sqrt(denominator * num_nonmissing.reshape(1,-1))),
            num_nonmissing // 2)

def run_gwas(
            genotypes: LinearOperator, 
            data: pl.LazyFrame, 
            pheno_cols: list[str], 
            covar_cols: list[str],
            variant_info: Optional[pl.LazyFrame] = None,
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

    Returns:
        Polars LazyFrame containing GWAS results
    """
    if not np.allclose(data.select(covar_cols[0]).collect().to_numpy(), 1.0):
        raise ValueError("First column of covar_cols should be '1'")

    merge_operator = get_merge_operator(data.select('iid').collect().to_series(), genotypes.iids)
    print(merge_operator.shape)
    phenotypes = data.select(pheno_cols).collect().to_numpy()
    covariates = data.select(covar_cols).collect().to_numpy()

    beta, se, sample_size = get_gwas_beta_se(genotypes, merge_operator, phenotypes, covariates)

    m, num_traits = beta.shape
    if len(pheno_cols) != num_traits:
        raise ValueError("Mismatch between number of pheno_cols and calculated traits.")
    if sample_size.shape != (num_traits,):
         # Adjust if sample_size comes out as (1, num_traits)
         if sample_size.shape == (1, num_traits):
             sample_size = sample_size.flatten()
         else:
             raise ValueError(f"Unexpected shape for sample_size: {sample_size.shape}")

    # Create repeated series for traits and sample sizes using extend_constant
    Phenotype = pl.Enum(pheno_cols)
    trait_series = pl.Series(dtype=Phenotype)
    n_series = pl.Series(dtype=pl.Float32)

    for i, pheno_name in enumerate(pheno_cols):
        trait_series = trait_series.extend_constant(pheno_name, m)
        n_series = n_series.extend_constant(sample_size[i], m)

    def chisq_pval(z: pl.Series) -> pl.Series:
        return pl.Series(2 * (1 - norm.cdf(z.abs().to_numpy())))

    results_df = pl.LazyFrame({
        "beta": beta.T.ravel(),
        "se": se.T.ravel(),
        "n": n_series,
        "trait": trait_series
        })\
        .with_row_index('variant_index')\
        .with_columns(
            pl.col('variant_index') % m,
            (pl.col('beta') / pl.col('se')).alias('z'))\
        .with_columns(
            pl.col('z').map_batches(chisq_pval).alias('pval')
        )

    if variant_info is not None:
        results_df = results_df.join(variant_info.with_row_index('variant_index'), on='variant_index')

    return results_df
