import numpy as np

from numpy.typing import NDArray
from scipy.sparse import csc_matrix


def compute_af(genotypes: csc_matrix, ploidy: int = 1) -> NDArray:
    n, p = genotypes.shape

    column_sums = genotypes.sum(axis=0)

    # Convert column sums to a flat array (necessary for sparse matrices)
    column_sums = np.ravel(column_sums)
    af = column_sums / n / ploidy

    return af


def flip_alleles(genotypes: csc_matrix, ploidy: int = 1) -> tuple[csc_matrix, NDArray]:
    n, p = genotypes.shape

    # Calculate allele frequencies
    af = compute_af(genotypes, ploidy)
    flip = af > 0.5

    # list-of-columns format
    genotypes_lil = genotypes.T.tolil()

    for i in range(genotypes_lil.shape[0]):
        if flip[i]:
            af[i] = 1 - af[i]

            # Convert the row to dense, flip the alleles, and assign it back
            row_dense = genotypes_lil[i, :].toarray()
            flipped_row_dense = ploidy - row_dense
            genotypes_lil[i, :] = flipped_row_dense

    f_idxs = np.where(flip)[0]
    return genotypes_lil.T.tocsc(), f_idxs


def apply_maf_threshold(genotypes: csc_matrix, ploidy: int = 1, threshold: float = 0.0) -> csc_matrix:
    # Calculate allele frequencies
    af = compute_af(genotypes, ploidy)

    # Calculate MAF (ensure p is a flat array for element-wise operations)
    maf = np.minimum(af, 1 - af)

    # Find indices where MAF is above the threshold
    maf_above_threshold_indices = np.where(maf > threshold)[0]

    # Keep only the columns of self.genotypes where MAF is above the threshold
    return genotypes[:, maf_above_threshold_indices]


def binarize(genotypes: csc_matrix, r2_threshold: float = 0.0) -> tuple[csc_matrix, NDArray]:
    n, p = genotypes.shape
    discretized_genotypes = np.rint(genotypes).astype(int)

    # TODO: vectorize
    # Correlations between dosages + calls
    correlations = []
    for i in range(p):
        corr_coef = np.corrcoef(genotypes[:, i].todense().T, discretized_genotypes[:, i].todense().T)[0, 1]
        correlations.append(corr_coef)

    # Thresholding
    well_imputed = np.asarray(correlations) >= r2_threshold
    r2_idxs = np.where(well_imputed)[0]

    # Update the genotypes with the discretized values
    genotypes = discretized_genotypes[:, well_imputed]

    return genotypes, r2_idxs
