import numpy as np

from linear_dag import LinearARG


def simulate_phenotype(linarg: LinearARG, heritability: float, alpha: float = 0, fraction_causal: float = 1):
    """
    Simulates quantitative phenotypes
        y = X * beta + epsilon
    with specified heritability, AF-dependent architecture, and polygenicity
    :param linarg: genotypes
    :param heritability: between 0 and 1
    :param alpha: between -1 and 0; cf. Schoech et al 2019 Nat Commun
    :param fraction_causal: between 0 and 1
    :return: vector of phenotype values
    """

    N, M = linarg.shape

    beta = np.random.randn(M)
    heterozygosity = 2 * linarg.allele_frequencies * (1 - linarg.allele_frequencies)
    beta[heterozygosity == 0] = 0
    heterozygosity[heterozygosity == 0] = 1
    beta = beta * np.sqrt(heterozygosity**alpha)
    is_causal = np.random.rand(M) < fraction_causal
    beta = beta * is_causal

    y_bar = linarg @ beta
    y_bar -= np.mean(y_bar)
    y_bar /= np.std(y_bar)
    y_bar *= np.sqrt(heritability)
    y = y_bar + np.random.randn(N) * np.sqrt(1 - heritability)

    return y
