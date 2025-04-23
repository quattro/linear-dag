from ..core.parallel_processing import ParallelOperator
from scipy.sparse.linalg import LinearOperator
import numpy as np
import polars as pl

def _backslash(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """MATLAB-style backslash"""
    return np.linalg.solve(A.T @ A, A.T @ b)

def _residualize_phenotypes(phenotypes: np.ndarray, covariates: np.ndarray) -> np.ndarray:
    """ Residualize phenotypes on covariates """
    beta = _backslash(covariates, phenotypes)
    return phenotypes - covariates @ beta

def _get_genotype_variance_explained(genotypes: LinearOperator, covariates: np.ndarray) -> np.ndarray:
    """ Get variance of genotypes explained by covariates: 
                diag(X'C(C'C)^-1C'X) / n
        where X is the genotype operator, C is the covariate matrix, and diag() extracts the diagonal
    """
    covariate_inner = covariates.T @ covariates
    between_product = covariates.T @ genotypes
    return np.mean(between_product * np.linalg.solve(covariate_inner, between_product), axis=1)
    
def get_gwas_betas(genotypes_normalized: LinearOperator, phenotypes: np.ndarray, covariates: np.ndarray):
    """
    Get GWAS effect-size estimates in per-s.d. units: s.d. of residualized phenotype per s.d. of 
    residualized genotype. Recommended that `covariates` 
    includes the all-ones annotation (unless phenotypes have been pre-centered). Assumes that columns of
    `genotypes_normalized` have x_i'x_i = n.
    """
    y_resid = _residualize_phenotypes(phenotypes, covariates)
    y_resid /= np.std(y_resid, axis=0)
    Xty = genotypes_normalized.T @ y_resid / genotypes_normalized.shape[0]
    print(np.std(Xty, axis=0))
    
    print("Xty:")
    print(Xty[:5, :5])
    denominator = np.sqrt(1 - _get_genotype_variance_explained(genotypes_normalized, covariates))
    print("denominator:")
    print(denominator[:5])
    return Xty / denominator
