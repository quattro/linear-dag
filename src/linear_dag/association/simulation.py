import numpy as np
from typing import Optional
from scipy.sparse.linalg import LinearOperator
from numpy.random import Generator

def simulate_phenotype(genotypes: LinearOperator, 
                        heritability: float, 
                        fraction_causal: float = 1, 
                        num_traits: int = 1,
                        return_genetic_component: bool = False,
                        return_beta: bool = False,
                        variant_effect_variance: Optional[np.ndarray] = None,
                        seed: Optional[Generator] = None):
    """
    Simulates quantitative phenotypes
        y = X * beta + epsilon
    with specified heritability, AF-dependent architecture, and polygenicity
    :param genotypes: linear operator for the genotype matrix, which 
        can be normalized (cols have unit variance) or not (0-1-2 or 0-1 valued).
    :param heritability: between 0 and 1
    :param fraction_causal: between 0 and 1
    :param num_traits: number of phenotypes to simulate
    :param return_genetic_component: whether to return the genetic component of the phenotype
    :param return_beta: whether to return the beta vector
    :param variant_effect_variance: relative effect-size variance for each marker, in units
        of variance-in-y per variance-in-columns-of-genotypes
    """
    N, M = genotypes.shape
    if seed is None:
        seed = np.random.default_rng()
    beta = seed.standard_normal((M, num_traits), dtype=np.float32)
    is_causal = np.random.rand(M) < fraction_causal
    beta *= is_causal.reshape(-1, 1)
    if variant_effect_variance is not None:
        beta *= np.sqrt(variant_effect_variance).reshape(-1, 1)

    y_bar = genotypes @ beta
    y_bar -= np.mean(y_bar, axis=0)
    
    # Standardize y_bar
    multiplier = np.sqrt(heritability) / np.std(y_bar, axis=0)
    if np.any(np.isinf(multiplier)):
        raise ValueError("Divide by zero error occurred, perhaps because no causal variants were sampled.")
    multiplier[np.isnan(multiplier)] = 0 # heritability == 0
    y_bar *= multiplier
    beta *= multiplier

    y = y_bar + np.random.randn(N, num_traits) * np.sqrt(1 - heritability)
    y = y.astype(np.float32)

    if return_beta:
        return y, beta
    if return_genetic_component:
        return y, y_bar
    return y
