import numpy as np
import time
from scipy.sparse.linalg import LinearOperator


def _backslash(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """MATLAB-style backslash"""
    return np.linalg.solve(A.T @ A, A.T @ b)


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
) -> tuple[np.ndarray, np.ndarray]:
    """Get variance of genotypes explained by covariates:
            diag(X'C(C'C)^-1C'X) / n
    where X is the genotype operator, C is the covariate matrix, and diag() extracts the diagonal.

    Args:
        XtC: X'C
        C: Covariates matrix, which should include the all-ones annotation except for missing values

    Returns:
        tuple: (total_var_explained, allele_count)
            total_var_explained: Total variance of genotypes explained by covariates
            allele_count: Allele count of the genotypes, assuming first column of covariates is all-ones
                            except for missing values
    """
    covariate_inner = C.T @ C
    t = time.time()
    between_product = XtC.T
    print(f"Time to compute between product: {time.time() - t}")
    t = time.time()
    C_backslash_XtC = np.linalg.solve(covariate_inner, between_product)
    print(f"Time to compute backslash: {time.time() - t}")

    # diag(A.T @ B) == sum(A * B)
    t = time.time()
    total_var_explained = np.sum(between_product * C_backslash_XtC, axis=0).reshape(-1, 1)
    print(f"Time to compute total variance explained: {time.time() - t}")

    allele_count = between_product[0, :].reshape(-1, 1)
    return total_var_explained, allele_count


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
