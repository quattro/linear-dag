from scipy.sparse.linalg import LinearOperator
import numpy as np
import polars as pl
from pathlib import Path
from typing import List

def _backslash(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """MATLAB-style backslash"""
    return np.linalg.solve(A.T @ A, A.T @ b)

def _residualize_phenotypes(phenotypes: np.ndarray, covariates: np.ndarray, phenotypes_missing: np.ndarray) -> np.ndarray:
    """ Residualize phenotypes on covariates, accounting for missingness assuming missing at random,
    i.e., covariance matrix of the covariates is the same for individuals with and without missing data.
    Input phenotypes should be zero for missing values."""
    beta = _backslash(covariates, phenotypes)
    residuals = phenotypes - (covariates @ beta) / (1 - np.mean(phenotypes_missing))
    residuals.ravel()[phenotypes_missing.ravel()] = 0
    return residuals

def _get_genotype_variance_explained(genotypes: LinearOperator, covariates: np.ndarray) -> np.ndarray:
    """ Get variance of genotypes explained by covariates: 
                diag(X'C(C'C)^-1C'X) / n
        where X is the genotype operator, C is the covariate matrix, and diag() extracts the diagonal
    """
    n = genotypes.shape[0]
    covariate_inner = covariates.T @ covariates
    between_product = covariates.T @ genotypes
    covariates_backslash_genotypes = np.linalg.solve(covariate_inner, between_product)

    # diag(A @ B) == sum(A * B)
    return 1/n * np.sum(between_product * covariates_backslash_genotypes, axis=0).reshape(-1,1)
    
def _impute_missing_with_mean(data: np.ndarray) -> np.ndarray:
    """Impute missing values with the mean of the column in place."""
    data = data.copy()
    for col in range(data.shape[1]):
        is_missing = np.isnan(data[:, col])
        col_mean = np.mean(data[~is_missing, col])
        data[is_missing, col] = col_mean
    return data

def get_gwas_betas(genotypes_normalized: LinearOperator, phenotypes: np.ndarray, covariates: np.ndarray):
    """
    Get GWAS effect-size estimates in per-s.d. units: s.d. of residualized phenotype per s.d. of 
    residualized genotype. Recommended that `covariates` includes the all-ones annotation 
    (unless phenotypes have been pre-centered). Assumes that columns of `genotypes_normalized` 
    have x_i'x_i = n.

    Args:
        genotypes_normalized: Normalized genotypes as a linear operator (e.g. ParallelOperator or LinearARG)
        phenotypes: Phenotypes matrix
        covariates: Covariates matrix

    Returns:
        np.ndarray: GWAS effect-size estimates in per-s.d. units
    """
    
    # Handle missingness
    covariates = _impute_missing_with_mean(covariates)
    is_missing = np.isnan(phenotypes)
    num_nonmissing = np.sum(~is_missing)
    phenotypes = phenotypes.copy()
    phenotypes.ravel()[is_missing.ravel()] = 0

    # Residualize phenotypes on covariates
    y_resid = _residualize_phenotypes(phenotypes, covariates, is_missing)
    y_resid /= np.sqrt(np.sum(y_resid**2, axis=0) / num_nonmissing) # ||y_resid||^2 == num_nonmissing

    # Get GWAS effect-size estimates
    Xty = genotypes_normalized.T @ y_resid / num_nonmissing
    
    var_explained = _get_genotype_variance_explained(genotypes_normalized, covariates)
    denominator = np.sqrt(1 + 1e-12 - var_explained) # asssumed equal across traits despite missingness
    return Xty / denominator


