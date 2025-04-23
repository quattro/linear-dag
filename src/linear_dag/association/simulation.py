import numpy as np
from ..core import LinearARG
from ..core.parallel_processing import ParallelOperator
from typing import Optional, Union
from numpy.random import Generator

def simulate_phenotype(linarg: Union[LinearARG, ParallelOperator], heritability: float, 
                        alpha: float = 0, fraction_causal: float = 1, 
                        num_traits: int = 1,
                        return_genetic_component: bool = False,
                        return_beta: bool = False,
                        seed: Optional[Generator] = None):
    """
    Simulates quantitative phenotypes
        y = X * beta + epsilon
    with specified heritability, AF-dependent architecture, and polygenicity
    :param linarg: genotypes
    :param heritability: between 0 and 1
    :param num_traits: number of phenotypes to simulate
    :param alpha: between -1 and 0; cf. Schoech et al 2019 Nat Commun
    :param fraction_causal: between 0 and 1
    :return: vector of phenotype values
    """

    N, M = linarg.shape
    if seed is None:
        seed = np.random.default_rng()
    beta = seed.standard_normal((M, num_traits), dtype=np.float32)
    heterozygosity = 2 * linarg.allele_frequencies * (1 - linarg.allele_frequencies)
    beta[heterozygosity == 0] = 0
    heterozygosity[heterozygosity == 0] = 1
    beta = beta * np.sqrt(heterozygosity**alpha).reshape(-1, 1)
    is_causal = np.random.rand(M) < fraction_causal
    beta = beta * is_causal.reshape(-1, 1)

    y_bar = linarg @ beta
    y_bar -= np.mean(y_bar, axis=0)
    y_bar /= np.std(y_bar, axis=0)
    y_bar *= np.sqrt(heritability)
    y = y_bar + np.random.randn(N, num_traits) * np.sqrt(1 - heritability)
    y = y.astype(np.float32)

    if return_beta:
        return y, beta
    if return_genetic_component:
        return y, y_bar
    return y
