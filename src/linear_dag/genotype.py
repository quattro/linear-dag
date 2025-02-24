import os

from collections import defaultdict
from os import PathLike
from typing import DefaultDict, Optional, Union

import cyvcf2 as cv
import numpy as np
import polars as pl

from numpy.typing import NDArray
from scipy.io import mmread
from scipy.sparse import csc_matrix


def read_vcf(
    path: Union[str, PathLike], phased: bool = True, region: Optional[str] = None, flip_minor_alleles: bool = False, whitelist: list = None, maf_filter: float = None, remove_indels: bool = False
):
    def _update_dict_from_vcf(
        var: cv.Variant, is_flipped: bool, data: DefaultDict[str, list]
    ) -> DefaultDict[str, list]:
        data["CHROM"].append(var.CHROM)
        data["POS"].append(var.POS)
        data["ID"].append(var.ID)
        data["REF"].append(var.REF)
        data["ALT"].append(",".join(var.ALT))
        data["FLIP"].append(is_flipped)

        return data

    vcf = cv.VCF(path, gts012=True, strict_gt=True, samples=whitelist)
    data = []
    idxs = []
    ptrs = [0]

    ploidy = 1 if phased else 2

    # push most of the branching up here to define functions for fewer branch conditions during loop
    if phased:
        read_gt = lambda var: np.ravel(np.asarray(var.genotype.array())[:, :2]) # noqa: E731
    else:
        read_gt = lambda var: var.gt_types # noqa: E731

    def final_read(var, flip_minor_alleles):
        gts = read_gt(var)
        if not flip_minor_alleles:
            return gts, False
        af = np.mean(gts) / ploidy
        if af > 0.5:
            return ploidy - gts, True
        else:
            return gts, False

    var_table = defaultdict(list)
    # TODO: handle missing data
    for var in vcf(region):
        
        if remove_indels:
            if any(len(alt) != 1 for alt in var.ALT) or len(var.REF) != 1:
                continue
        
        gts, is_flipped = final_read(var, flip_minor_alleles)
        
        if maf_filter is not None:
            af = np.mean(gts) / ploidy
            if (af < maf_filter) or (1-af < maf_filter):
                continue

        (idx,) = np.where(gts != 0)
        data.append(gts[idx])
        idxs.append(idx)
        ptrs.append(ptrs[-1] + len(idx))
        var_table = _update_dict_from_vcf(var, is_flipped, var_table)

    v_info = pl.DataFrame(var_table)
    
    if len(data) == 0:
        print('No variants found.')
        return None, None

    data = np.concatenate(data)
    idxs = np.concatenate(idxs)
    ptrs = np.array(ptrs)
    genotypes = csc_matrix((data, idxs, ptrs), shape=(gts.shape[0], len(ptrs) - 1)) # some samples may have no variants, so shape must be specified

    return genotypes, v_info


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
