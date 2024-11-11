# heritability.py
import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import aslinearoperator
from linear_dag import LinearARG


def randomized_haseman_elston(linarg: LinearARG, ys: np.ndarray, B: int = 20, alpha: float = -1) -> list[float]:
    """
    Implementation of the RHE algorithm from:
        Pazokitoroudi, A. et al. Efficient variance components analysis across millions of genomes.
        Nat Commun 11, 4020 (2020). https://doi.org/10.1038/s41467-020-17576-9
    Notation below follows Methods section of this paper.

    :param linarg: linear ARG for the genotype matrix
    :param y: phenotype vector
    :param b: number of random vectors to use when estimating Tr(GRM^2)
    :return: the heritability estimate
    """

    N, M = linarg.shape

    heterozygosity = linarg.allele_frequencies * (1 - linarg.allele_frequencies)
    heterozygosity[heterozygosity == 0] = 1

    # Genetic relatedness matrix
    K = linarg.mean_centered @ aslinearoperator(diags(heterozygosity ** alpha / np.sum(heterozygosity ** (1+alpha)))) @ linarg.mean_centered.T

    # Tr(K@K)
    T = 0
    b = 0
    for i in range(B):
        z = np.random.randn(N)
        Kz = K @ z
        b += np.dot(z,Kz)
        T += np.dot(Kz,Kz)
    T /= B
    b /= B
    print(T, b)

    heritability = []
    for i in range(ys.shape[1]):
        y = ys[:, i]
        y = y.copy() - np.mean(y)
        y = y / np.std(y)  # y is mean 0, variance 1
        c = y.T @ K @ y
        solution = np.linalg.solve(np.array([[T, b], [b, N]]), np.array([c, N]).T)
        heritability.append(solution[0])

    return heritability





