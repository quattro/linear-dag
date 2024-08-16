import os

from typing import Optional

import numpy as np

from cyvcf2 import VCF
from numpy.typing import NDArray
from scipy.io import mmread
from scipy.sparse import csc_matrix


def load_genotypes(
    input_file_prefix: str,
    flip_minor_alleles: bool = False,
    maf_threshold: Optional[float] = None,
    rsq_threshold: Optional[float] = None,
    skiprows: int = 0,
) -> tuple[csc_matrix, NDArray, NDArray]:
    mtx_file = f"{input_file_prefix}.mtx"
    txt_file = f"{input_file_prefix}.txt"
    if os.path.exists(mtx_file):
        genotype_file = mtx_file
        input_type = "mtx"
    elif os.path.exists(txt_file):
        genotype_file = txt_file
        input_type = "txt"
    else:
        raise FileNotFoundError(f"No genotype matrix file found with prefix: {input_file_prefix}")

    # Initialize Linarg based on input file type
    if input_type == "mtx":
        genotypes = csc_matrix(mmread(genotype_file))
    else:
        genotypes = np.loadtxt(genotype_file, skiprows=skiprows)

    if rsq_threshold is None:
        well_imputed_variants = np.arange(genotypes.shape[1])
    else:
        genotypes, well_imputed_variants = binarize(genotypes, rsq_threshold)

    ploidy = np.max(genotypes).astype(int)
    if maf_threshold is None:
        common_variants = np.arange(genotypes.shape[1])
    else:
        genotypes, common_variants = apply_maf_threshold(genotypes, ploidy, maf_threshold)

    kept_variants = well_imputed_variants[common_variants]
    print("kept_variants:", kept_variants.shape)

    if flip_minor_alleles:
        genotypes, flipped_variants = flip_alleles(genotypes, ploidy)
    else:
        flipped_variants = None

    return genotypes, kept_variants, flipped_variants


def read_vcf(
    path: str, region: Optional[str] = None, phased: bool = False, flip_minor_alleles: bool = True
) -> tuple[csc_matrix, np.ndarray, list[dict]]:
    """
    Codes unphased genotypes as 0/1/2/3, where 3 means that at least one of the two alleles is unknown.
    Codes phased genotypes as 0/1, and there are 2n rows, where rows 2*k and 2*k+1 correspond to individual k.
    """
    vcf = VCF(path, gts012=True, strict_gt=True)
    data = []
    idxs = []
    ptrs = [0]
    info = []
    flip = []

    ploidy = 1 if phased else 2

    # TODO: handle missing data
    for var in vcf(region):
        if phased:
            gts = np.ravel(np.asarray(var.genotype.array())[:, :2])
        else:
            gts = var.gt_types
        if flip_minor_alleles:
            af = np.mean(gts) / ploidy
            if af > 0.5:
                gts = ploidy - gts
                flip.append(True)
            else:
                flip.append(False)

        (idx,) = np.where(gts != 0)
        data.append(gts[idx])
        idxs.append(idx)
        ptrs.append(ptrs[-1] + len(idx))
        info.append(var.INFO)

    data = np.concatenate(data)
    idxs = np.concatenate(idxs)
    ptrs = np.array(ptrs)
    genotypes = csc_matrix((data, idxs, ptrs))
    flip = np.array(flip)

    return genotypes, flip, info


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


def apply_maf_threshold(genotypes: csc_matrix, ploidy: int = 1, threshold: float = 0.0) -> tuple[csc_matrix, NDArray]:
    # Calculate allele frequencies
    af = compute_af(genotypes, ploidy)

    # Calculate MAF (ensure p is a flat array for element-wise operations)
    maf = np.minimum(af, 1 - af)

    # Find indices where MAF is above the threshold
    maf_above_threshold_indices = np.where(maf > threshold)[0]

    # Keep only the columns of self.genotypes where MAF is above the threshold
    return genotypes[:, maf_above_threshold_indices], maf_above_threshold_indices


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
