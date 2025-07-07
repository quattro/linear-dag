from linear_dag.core.operators import get_merge_operator, get_inner_merge_operators
from scipy.sparse.linalg import LinearOperator
from scipy.stats import chi2
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
                                    covariates: np.ndarray,
                                    ) -> Tuple[np.ndarray, np.ndarray]:
    """ Get variance of genotypes explained by covariates: 
                diag(X'C(C'C)^-1C'X) / n
        where X is the genotype operator, C is the covariate matrix, and diag() extracts the diagonal.
        
        Args:
            genotypes: Unnormalized, phased genotypes as a linear operator (e.g. ParallelOperator or LinearARG)
            covariates: Covariates matrix, which should include the all-ones annotation except for missing values

        Returns:
            tuple: (total_var_explained, allele_count)
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


def _get_genotype_variance(
                          genotypes: LinearOperator,
                          allele_counts: np.ndarray,
                          individuals_to_include: np.ndarray,
                        ) -> Tuple[np.int64, np.ndarray]:
    """ Get variance of genotypes without assuming HWE: diag(X^TX)
    
    Args:
        genotypes: Unnormalized, phased genotypes as a linear ARG with ploidy (i.e. individual nodes)
        allele_counts: Counts of each allele
        individuals_to_keep: Non-missing individuals to include in carrier count
        
    Returns:
        tuple: (var_genotypes, num_homozygotes)
            var_genotypes: variance of genotypes
            carrier_counts: number of carriers per allele
    """
    carrier_counts = genotypes.number_of_carriers(individuals_to_include).reshape(-1, 1)
    assert np.all(allele_counts - carrier_counts >= 0)
    var_genotypes = 3 * allele_counts - 2 * carrier_counts # 4 * num_homozygotes + num_heterozygotes
    return var_genotypes, carrier_counts
    
    
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
                left_op: LinearOperator,
                right_op: LinearOperator,
                phenotypes: np.ndarray, 
                covariates: np.ndarray,
                assume_hwe: bool,
                variant_info: Optional[pl.LazyFrame] = None,
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
    if not np.allclose(covariates[:,0], 1):
        raise ValueError("First column of covariates should be all-ones")

    cols_matched_per_row = left_op @ np.ones(left_op.shape[1])
    # Check if all *non-zero* counts are exactly 2
    if not np.all(cols_matched_per_row[cols_matched_per_row != 0] == 2):
        raise ValueError("Each row of the phenotype matrix should match zero or two rows of the genotype operator")
    
    rows_matched_per_col = np.ones(right_op.shape[0]) @ right_op
    # Check if all *non-zero* counts are exactly 1
    if not np.all(rows_matched_per_col[rows_matched_per_col != 0] == 1):
        raise ValueError("Each row of the genotype operator should match at most one row of the phenotype matrix")

    two_n = np.sum(rows_matched_per_col>0)
    assert two_n == 2 * np.sum(cols_matched_per_row>0)
    
    covariates = left_op.T @ covariates
    phenotypes = left_op.T @ phenotypes
        
    # Handle missingness
    covariates = _impute_missing_with_mean(covariates)
    is_missing = np.isnan(phenotypes)
    num_nonmissing = np.sum(~is_missing, axis=0)
    print(f"num_nonmissing: {num_nonmissing}")
    phenotypes.ravel()[is_missing.ravel()] = 0

    # Residualize phenotypes on covariates
    y_resid = residualize_phenotypes(phenotypes, covariates, is_missing)
    # y_resid /= np.sqrt(np.sum(y_resid**2, axis=0) / num_nonmissing) # ||y_resid||^2 == num_nonmissing
    # assert np.allclose(np.sum(y_resid**2, axis=0), num_nonmissing), "Non-unit mean squared residuals, indicating a numerical issue; check for collinearity"
    # assert np.allclose(np.mean(y_resid, axis=0), 0, rtol=1e-3), "Non-zero mean residuals, indicating a numerical issue; check for collinearity"
    numerator = (right_op @ genotypes).T @ y_resid / num_nonmissing
    
    # Get denominator, which is assumed equal across traits despite different missingness
    
    var_explained, allele_counts = _get_genotype_variance_explained(right_op @ genotypes, covariates)
    if assume_hwe:
        denominator = np.maximum(allele_counts - var_explained, 1e-6) / two_n
        carrier_counts = None
    else:        
        if genotypes.sex is not None:
            raise NotImplementedError
        individuals_to_include = np.where(rows_matched_per_col[::2]==1)[0] # non-missing individuals to include in carrier count
        var_genotypes, carrier_counts = _get_genotype_variance(genotypes, allele_counts, individuals_to_include)
        denominator = np.maximum(var_genotypes - var_explained, 1e-6) / two_n
    assert np.all(denominator > 0), min(denominator)
    
    var_resid = np.sum(y_resid ** 2, axis=0) / num_nonmissing
    se = np.sqrt(var_resid.reshape(1, -1) / (denominator * num_nonmissing.reshape(1, -1)))

    return (numerator / denominator, 
            se,
            num_nonmissing // 2,
            allele_counts,
            carrier_counts)

def run_gwas(
            genotypes: LinearOperator, 
            data: pl.LazyFrame, 
            pheno_cols: list[str], 
            covar_cols: list[str],
            variant_info: Optional[pl.LazyFrame] = None,
            assume_hwe: bool = True,
            ) -> list:
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
    if not assume_hwe and not hasattr(genotypes, 'n_individuals'):
        raise ValueError('If assume_hwe is False, genotypes must be a linear ARG with individual nodes.')
    
    if not np.allclose(data.select(covar_cols[0]).collect().to_numpy(), 1.0):
        raise ValueError("First column of covar_cols should be '1'")

    left_op, right_op = get_inner_merge_operators(data.select('iid').collect().to_series(), genotypes.iids) # data iids to shared iids, shared iids to genotypes iids
    phenotypes = data.select(pheno_cols).collect().to_numpy()
    covariates = data.select(covar_cols).collect().to_numpy()

    beta, se, sample_size, allele_counts, carrier_counts = get_gwas_beta_se(genotypes, left_op, right_op, phenotypes, covariates, assume_hwe)

    m, num_traits = beta.shape
    if len(pheno_cols) != num_traits:
        raise ValueError("Mismatch between number of pheno_cols and calculated traits.")
    if sample_size.shape != (num_traits,):
         # Adjust if sample_size comes out as (1, num_traits)
         if sample_size.shape == (1, num_traits):
             sample_size = sample_size.flatten()
         else:
             raise ValueError(f"Unexpected shape for sample_size: {sample_size.shape}")

    def log_chisq_pval(z: np.ndarray) -> np.ndarray:
        # return -np.log10(1 - chi2(1).cdf(z ** 2))
        return -chi2(1).logsf(z ** 2)
    
    results = []
    
    cols = ["A1FREQ", "BETA", "SE", "CHISQ", "LOG10P", "N"]
    if variant_info is not None:
        cols = ["CHROM", "POS", "ID", "ALLELE0", "ALLELE1"] + cols 
        variant_info = variant_info.rename({
                "REF": "ALLELE0",
                "ALT": "ALLELE1",
            })
        variant_info = variant_info.with_row_index("variant_index").with_columns(
                pl.col("variant_index").cast(pl.Int32)
            )
    if carrier_counts is not None:
        cols = cols + ["CARRIER_COUNTS"]

    for i in range(len(pheno_cols)):
        z_scores = beta[:, i]/se[:, i]
        frame_dict = {
            "variant_index": pl.Series("variant_index", np.arange(m, dtype=np.int32), dtype=pl.Int32),
            "BETA": beta[:, i],
            "SE": se[:, i],
            "CHISQ": z_scores**2,
            "LOG10P": log_chisq_pval(z_scores),
            "A1FREQ": allele_counts.reshape(-1)/genotypes.shape[0],
            "N": m*[genotypes.shape[0]//2]
        }
        if carrier_counts is not None:
            frame_dict["CARRIER_COUNTS"] = carrier_counts.reshape(-1)

        df = pl.DataFrame(frame_dict)
        
        if variant_info is not None:
            df = df.join(
                variant_info.collect(),
                on="variant_index",
                how="left"
            )
            
        df = df.select(cols)
        results.append(df)

    return results
