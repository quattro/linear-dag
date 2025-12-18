import numpy as np
import time
from scipy.sparse.linalg import LinearOperator


def _backslash(A: np.ndarray, b: np.ndarray, lam: float = 1e-5) -> np.ndarray:
    """MATLAB-style backslash"""
    # return np.linalg.solve(A.T @ A, A.T @ b)
    return np.linalg.pinv(A.T @ A, rcond=lam) @ (A.T @ b)

def _residualize_phenotypes_mar(
    phenotypes: np.ndarray, covariates: np.ndarray, phenotypes_missing: np.ndarray
) -> np.ndarray:
    """Residualize phenotypes on covariates assuming missing at random (MAR),
    i.e., covariance matrix of the covariates is the same for individuals with and without missing data.
    Input phenotypes should be zero for missing values."""
    beta = _backslash(covariates, phenotypes)
    residuals = phenotypes - (covariates @ beta) / (1 - np.mean(phenotypes_missing, axis=0))
    residuals.ravel()[phenotypes_missing.ravel()] = 0
    return residuals


def residualize_phenotypes(
    phenotypes: np.ndarray,
    covariates: np.ndarray,
    phenotypes_missing: np.ndarray,
    missingness_threshold_mar: float = 0,
) -> np.ndarray:
    """Residualize phenotypes on covariates. For phenotypes with missingness <=
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



def get_genotype_variance_explained(
    XtC: np.ndarray,
    C: np.ndarray,
    batch_size: int = 100_000,
    lam: float = 1e-5,
) -> tuple[np.ndarray, np.ndarray]:
    """Get variance of genotypes explained by covariates:
            diag(X'C(C'C)^-1C'X) / n
    where X is the genotype operator, C is the covariate matrix, and diag() extracts the diagonal.

    Args:
        XtC: X'C
        C: Covariates matrix, which should include the all-ones annotation except for missing values
        batch_size: Number of SNPs to process at once to reduce memory usage
        lam: Regularization parameter for the pseudoinverse

    Returns:
        tuple: (total_var_explained, allele_count)
            total_var_explained: Total variance of genotypes explained by covariates
            allele_count: Allele count of the genotypes, assuming first column of covariates is all-ones
                            except for missing values
    """
    num_covar = C.shape[1]
    covariate_inner = C.T @ C

    num_snps = XtC.shape[0]
    total_var_explained = np.zeros((num_snps, 1), dtype=np.float32)

    for start_idx in range(0, num_snps, batch_size):
        end_idx = min(start_idx + batch_size, num_snps)
        XtC_batch = XtC[start_idx:end_idx, :num_covar]
        # C_backslash_XtC_batch = np.linalg.solve(covariate_inner, XtC_batch.T).astype(np.float32)
        C_backslash_XtC_batch = (np.linalg.pinv(covariate_inner, rcond=lam) @ XtC_batch.T).astype(np.float32)
        total_var_explained[start_idx:end_idx] = np.sum(
            XtC_batch.T * C_backslash_XtC_batch, axis=0
        ).reshape(-1, 1)

    allele_count = XtC[:, 0:1]

    return total_var_explained, allele_count

def get_genotype_variance_explained_recompute_AC(
    XtCD: np.ndarray,
    C: np.ndarray,
    num_nonmissing: np.ndarray|None = None,
    batch_size: int = 100_000,
    lam: float = 1e-5,
) -> tuple[np.ndarray, np.ndarray]:
    """Get variance of genotypes explained by covariates for each phenotype, using recomputed allele counts
    for each phenotype.

    Args:
        XtCD: X'[C D], where C is the covariates matrix and D is the matrix of indicators for non-missing phenotypes
        C: Covariates matrix, which should include the all-ones annotation except for missing values
        num_nonmissing: Number of non-missing observations for each phenotype, equal to the number of nonzeros in
        each column of D
        batch_size: Number of SNPs to process at once to reduce memory usage
        lam: Regularization parameter for the pseudoinverse

    Returns:
        tuple: (total_var_explained, allele_count)
            total_var_explained: Total variance of genotypes explained by covariates
            allele_count: Allele count of the genotypes, assuming first column of covariates is all-ones
                            except for missing values
    """
    n, num_covar = C.shape
    if num_covar + len(num_nonmissing) != XtCD.shape[1]:
        raise ValueError("XtCD must have the same number of columns as C and D")

    # n * v, n * p
    total_var_explained, total_allele_count = get_genotype_variance_explained(XtCD[:,:num_covar], C, batch_size, lam)

    # n' * p'
    allele_count = XtCD[:, num_covar:]

    # (p'/p)^2 * n' * v
    var_explained = n / num_nonmissing * total_var_explained * \
                            (allele_count / total_allele_count.reshape(-1,1)) ** 2
    np.nan_to_num(var_explained, copy=False, nan=0.0)

    return var_explained, allele_count


def _get_genotype_variance(
    genotypes: LinearOperator,
    allele_counts: np.ndarray,
    individuals_to_include: np.ndarray,
) -> tuple[np.int64, np.ndarray]:
    """Get variance of genotypes without assuming HWE: diag(X^TX)

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
    var_genotypes = 3 * allele_counts - 2 * carrier_counts  # 4 * num_homozygotes + num_heterozygotes
    return var_genotypes, carrier_counts


def impute_missing_with_mean(data: np.ndarray) -> np.ndarray:
    """Impute missing values with the mean of the column in place."""
    data = data.copy()
    for col in range(data.shape[1]):
        is_missing = np.isnan(data[:, col])
        col_mean = np.mean(data[~is_missing, col])
        data[is_missing, col] = col_mean
    return data
