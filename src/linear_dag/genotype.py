import logging
import os

from collections import defaultdict
from os import PathLike
from typing import Optional, Union

import cyvcf2 as cv
import h5py
import numpy as np
import polars as pl

from numpy.typing import NDArray
from scipy.io import mmread
from scipy.sparse import csc_matrix


def _iter_vcf_columns(
    path: Union[str, PathLike],
    phased: bool = True,
    region: Optional[str] = None,
    flip_minor_alleles: bool = False,
    samples: Optional[list[str]] = None,
    maf_filter: float = None,
    remove_indels: bool = False,
    remove_multiallelics: bool = False,
    sex: np.array = None,
):
    """Return sample metadata and an iterator over retained sparse VCF columns."""
    if samples is not None:
        vcf_samples = cv.VCF(path, lazy=True).samples
        samples_to_load = list(set(samples) & set(vcf_samples))
        if not samples_to_load:
            raise ValueError("Samples specified but none found in VCF")
        if len(samples_to_load) == len(vcf_samples):
            samples_to_load = None
    else:
        samples_to_load = None

    vcf = cv.VCF(path, gts012=True, strict_gt=True, samples=samples_to_load)
    iids = vcf.samples
    ploidy = 1 if phased else 2

    if phased:
        read_gt = lambda var: np.ravel(np.asarray(var.genotype.array())[:, :2])  # noqa: E731
    else:
        read_gt = lambda var: var.gt_types  # noqa: E731

    if sex is not None:
        mask = 2 * np.where(sex == 1)[0] + 1
        indices_to_keep = np.array([i for i in range(2 * len(vcf.samples)) if i not in mask])
        num_rows = len(indices_to_keep)
    else:
        indices_to_keep = None
        num_rows = len(iids) * (2 if phased else 1)

    if region:
        interval = region.split(":")[1]
        start = int(float(interval.split("-")[0]))
        end = int(float(interval.split("-")[1]))
    else:
        start = 0
        end = np.inf

    variants = vcf(region) if region else vcf

    def columns():
        for var in variants:
            if (var.POS < start) or (var.POS > end):
                continue

            if len(var.ALT) > 1:
                if remove_multiallelics:
                    continue
                variant_id = var.ID if var.ID is not None else "."
                raise ValueError(
                    "Multiallelic variant encountered in VCF/BCF input: "
                    f"{var.CHROM}:{var.POS} ID={variant_id} REF={var.REF} ALT={','.join(var.ALT)}. "
                    "Multiallelic variants are not supported by default; use "
                    "`remove_multiallelics=True` or `--remove-multiallelics` to exclude them."
                )

            if remove_indels and (any(len(alt) != 1 for alt in var.ALT) or len(var.REF) != 1):
                continue

            gts = read_gt(var)
            if indices_to_keep is not None:
                gts = gts[indices_to_keep]
                assert np.all((gts == 0) | (gts == 1)), (
                    "Haplotype vector contains non 0 or 1 values. Check genotype data or sex vector."
                )

            is_flipped = False
            if flip_minor_alleles:
                af = np.mean(gts) / ploidy
                if af > 0.5:
                    gts = ploidy - gts
                    is_flipped = True

            if maf_filter is not None:
                af = np.mean(gts) / ploidy
                if (af < maf_filter) or (1 - af < maf_filter):
                    continue

            indices = np.flatnonzero(gts)
            metadata = (var.CHROM, var.POS, var.ID, var.REF, ",".join(var.ALT))
            yield indices, gts[indices], is_flipped, metadata

    return iids, num_rows, columns()


def write_vcf_to_hdf5(
    path: Union[str, PathLike],
    output_path: Union[str, PathLike],
    phased: bool = True,
    region: Optional[str] = None,
    flip_minor_alleles: bool = False,
    samples: Optional[list[str]] = None,
    maf_filter: float = None,
    remove_indels: bool = False,
    remove_multiallelics: bool = False,
    sex: np.array = None,
    batch_nnz: int = 1_000_000,
    batch_columns: int = 100_000,
):
    """Stream retained VCF columns into an on-disk CSC representation."""
    iids, num_rows, columns = _iter_vcf_columns(
        path,
        phased=phased,
        region=region,
        flip_minor_alleles=flip_minor_alleles,
        samples=samples,
        maf_filter=maf_filter,
        remove_indels=remove_indels,
        remove_multiallelics=remove_multiallelics,
        sex=sex,
    )

    output_path = str(output_path)
    partial_path = f"{output_path}.partial"
    index_chunks = []
    data_chunks = []
    pointer_buffer = []
    buffered_nnz = 0
    total_nnz = 0
    num_variants = 0
    flip = []
    var_table = defaultdict(list)

    try:
        with h5py.File(partial_path, "w") as f:
            chunk_size = min(max(1, batch_nnz), 100_000)
            data_dtype = np.int16 if phased else np.int32
            indices_dataset = f.create_dataset(
                "indices",
                shape=(0,),
                maxshape=(None,),
                chunks=(chunk_size,),
                dtype=np.int32,
                compression="gzip",
                shuffle=True,
            )
            data_dataset = f.create_dataset(
                "data",
                shape=(0,),
                maxshape=(None,),
                chunks=(chunk_size,),
                dtype=data_dtype,
                compression="gzip",
                shuffle=True,
            )
            indptr_dataset = f.create_dataset(
                "indptr",
                shape=(1,),
                maxshape=(None,),
                chunks=(max(1, batch_columns),),
                dtype=np.int64,
                compression="gzip",
                shuffle=True,
            )
            indptr_dataset[0] = 0

            def flush():
                nonlocal buffered_nnz
                if index_chunks:
                    indices = np.concatenate(index_chunks).astype(np.int32, copy=False)
                    values = np.concatenate(data_chunks).astype(data_dtype, copy=False)
                    start = indices_dataset.shape[0]
                    stop = start + len(indices)
                    indices_dataset.resize((stop,))
                    data_dataset.resize((stop,))
                    indices_dataset[start:stop] = indices
                    data_dataset[start:stop] = values
                    index_chunks.clear()
                    data_chunks.clear()
                    buffered_nnz = 0
                if pointer_buffer:
                    start = indptr_dataset.shape[0]
                    stop = start + len(pointer_buffer)
                    indptr_dataset.resize((stop,))
                    indptr_dataset[start:stop] = pointer_buffer
                    pointer_buffer.clear()

            for indices, values, is_flipped, metadata in columns:
                index_chunks.append(indices.astype(np.int32, copy=False))
                data_chunks.append(values.astype(data_dtype, copy=False))
                buffered_nnz += len(indices)
                total_nnz += len(indices)
                pointer_buffer.append(total_nnz)
                num_variants += 1
                flip.append(is_flipped)
                chrom, pos, variant_id, ref, alt = metadata
                var_table["CHROM"].append(chrom)
                var_table["POS"].append(pos)
                var_table["ID"].append(variant_id)
                var_table["REF"].append(ref)
                var_table["ALT"].append(alt)

                if buffered_nnz >= batch_nnz or len(pointer_buffer) >= batch_columns:
                    flush()

            flush()

            if num_variants == 0:
                for key in list(f.keys()):
                    del f[key]
                f.attrs["is_empty"] = True
            else:
                f.create_dataset("shape", data=(num_rows, num_variants), compression="gzip", shuffle=True)
                f.create_dataset("flip", data=np.asarray(flip), compression="gzip", shuffle=True)
                output_iids = [iid for iid in iids for _ in range(2)] if phased else iids
                f.create_dataset("iids", data=output_iids, compression="gzip", shuffle=True)
                if sex is not None:
                    f.create_dataset("sex", data=sex, compression="gzip", shuffle=True)
                f.attrs["is_empty"] = False

        os.replace(partial_path, output_path)
    except BaseException:
        if os.path.exists(partial_path):
            os.remove(partial_path)
        raise

    if num_variants == 0:
        return None
    return pl.DataFrame(var_table)


def read_vcf(
    path: Union[str, PathLike],
    phased: bool = True,
    region: Optional[str] = None,
    flip_minor_alleles: bool = False,
    samples: Optional[list[str]] = None,
    maf_filter: float = None,
    remove_indels: bool = False,
    remove_multiallelics: bool = False,
    sex: np.array = None,
):
    """Load genotype calls from a VCF/BCF file into sparse CSC format.

    Rows correspond to haplotypes when `phased=True` and to diploid samples
    when `phased=False`. Columns correspond to retained variant records.

    !!! info

        When `sex` is provided, male haplotypes are treated as haploid on sex
        chromosomes and non-binary haplotype calls after filtering raise an
        assertion error.

        `flip_minor_alleles=True` rewrites columns so returned allele
        frequencies are at most $0.5$, and the accompanying `flip` vector
        records which columns were complemented.

    **Arguments:**

    - `path`: Input VCF/BCF path.
    - `phased`: If `True`, read phased haplotypes; otherwise read diploid genotype types.
    - `region`: Optional region string (`chrom:start-end`) for targeted loading.
    - `flip_minor_alleles`: If `True`, flips variants with allele frequency > 0.5.
    - `samples`: Optional sample IDs to include.
    - `maf_filter`: Optional MAF threshold for variant filtering.
    - `remove_indels`: If `True`, skip indel records.
    - `remove_multiallelics`: If `True`, skip multiallelic variants. If
      `False`, multiallelic variants raise a `ValueError`.
    - `sex`: Optional sex vector (`0` female / `1` male) for ploidy-aware filtering.

    **Returns:**

    - Tuple `(genotypes, flip, variant_info, iids)` where:
      - `genotypes` is a CSC matrix with samples/haplotypes on rows and variants on columns.
      - `flip` is a boolean array identifying minor-allele flips.
      - `variant_info` is a polars DataFrame with VCF metadata.
      - `iids` is the sample ID list from the VCF.
    - If no variants pass filters, returns `(None, None, None, None)`.

    **Raises:**

    - `ValueError`: If `samples` is provided but none are present in the VCF,
      or if a multiallelic variant is encountered and `remove_multiallelics=False`.
    """

    iids, num_rows, columns = _iter_vcf_columns(
        path,
        phased=phased,
        region=region,
        flip_minor_alleles=flip_minor_alleles,
        samples=samples,
        maf_filter=maf_filter,
        remove_indels=remove_indels,
        remove_multiallelics=remove_multiallelics,
        sex=sex,
    )
    data = []
    idxs = []
    ptrs = [0]
    flip = []
    var_table = defaultdict(list)
    for indices, values, is_flipped, metadata in columns:
        idxs.append(indices)
        data.append(values)
        ptrs.append(ptrs[-1] + len(indices))
        flip.append(is_flipped)
        chrom, pos, variant_id, ref, alt = metadata
        var_table["CHROM"].append(chrom)
        var_table["POS"].append(pos)
        var_table["ID"].append(variant_id)
        var_table["REF"].append(ref)
        var_table["ALT"].append(alt)

    v_info = pl.DataFrame(var_table)

    if len(data) == 0:
        return None, None, None, None

    data = np.concatenate(data)
    idxs = np.concatenate(idxs)
    ptrs = np.array(ptrs)
    genotypes = csc_matrix((data, idxs, ptrs), shape=(num_rows, len(ptrs) - 1))
    flip = np.array(flip)

    return genotypes, flip, v_info, iids


def load_genotypes(
    input_file_prefix: str,
    flip_minor_alleles: bool = False,
    maf_threshold: Optional[float] = None,
    rsq_threshold: Optional[float] = None,
    skiprows: int = 0,
) -> tuple[csc_matrix, NDArray, NDArray]:
    """Load genotype data from Matrix Market or text files and apply basic QC.

    !!! info

        The routine optionally binarizes dosages, filters by MAF, and flips
        high-frequency alternate alleles, returning both the filtered matrix and
        the retained original column indices.

    **Arguments:**

    - `input_file_prefix`: Prefix used to locate genotype files.
    - `flip_minor_alleles`: Whether to flip variants with frequency > 0.5.
    - `maf_threshold`: Optional minor-allele-frequency cutoff.
    - `rsq_threshold`: Optional dosage-to-hardcall correlation cutoff.
    - `skiprows`: Number of rows to skip when loading text genotype files.

    **Returns:**

    - Tuple `(genotypes, kept_variants, flipped_variants)` where `genotypes` is the
      filtered CSC matrix, `kept_variants` are original variant indices retained, and
      `flipped_variants` are indices flipped after filtering (or `None`).

    **Raises:**

    - `FileNotFoundError`: If neither `<prefix>.mtx` nor `<prefix>.txt` exists.
    """

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
    logging.getLogger(__name__).debug("kept_variants: %s", kept_variants.shape)

    if flip_minor_alleles:
        genotypes, flipped_variants = flip_alleles(genotypes, ploidy)
    else:
        flipped_variants = None

    return genotypes, kept_variants, flipped_variants


def compute_af(genotypes: csc_matrix, ploidy: int = 1) -> NDArray:
    """Compute per-column allele frequencies.

    Let $G \\in \\mathbb{R}^{n \\times p}$ denote the genotype matrix and let
    $c$ denote the ploidy. The returned vector is
    $f = \\mathbf{1}^\\top G / (n c)$.

    **Arguments:**

    - `genotypes`: Genotype matrix with variants on columns.
    - `ploidy`: Ploidy scaling factor used to normalize column sums.

    **Returns:**

    - NumPy array of allele frequencies for each variant column.
    """

    n, p = genotypes.shape

    column_sums = genotypes.sum(axis=0)

    # Convert column sums to a flat array (necessary for sparse matrices)
    column_sums = np.ravel(column_sums)
    af = column_sums / n / ploidy

    return af


def flip_alleles(genotypes: csc_matrix, ploidy: int = 1) -> tuple[csc_matrix, NDArray]:
    """Flip columns whose alternate-allele frequency exceeds $0.5$.

    For each selected column $j$, the routine replaces genotypes $g_j$ with
    $c - g_j$, where $c$ is `ploidy`, so the returned alternate allele becomes
    the minor allele.

    **Arguments:**

    - `genotypes`: Input CSC genotype matrix.
    - `ploidy`: Ploidy scaling used for allele-frequency computation.

    **Returns:**

    - Tuple `(flipped_genotypes, flipped_indices)` where `flipped_genotypes` is a
      CSC matrix after allele flipping and `flipped_indices` are variant indices that
      were flipped.
    """

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
    """Filter genotype columns by minor-allele-frequency threshold.

    A column is kept when $\\min(f_j, 1-f_j) > \\text{threshold}$, where $f_j$
    is the allele frequency computed by [`linear_dag.genotype.compute_af`][].

    **Arguments:**

    - `genotypes`: Input CSC genotype matrix.
    - `ploidy`: Ploidy scaling used for allele-frequency computation.
    - `threshold`: Strict MAF cutoff (`maf > threshold` retained).

    **Returns:**

    - Tuple `(filtered_genotypes, kept_indices)` where `kept_indices` are the
      original variant column positions retained after filtering.
    """

    # Calculate allele frequencies
    af = compute_af(genotypes, ploidy)

    # Calculate MAF (ensure p is a flat array for element-wise operations)
    maf = np.minimum(af, 1 - af)

    # Find indices where MAF is above the threshold
    maf_above_threshold_indices = np.where(maf > threshold)[0]

    # Keep only the columns of self.genotypes where MAF is above the threshold
    return genotypes[:, maf_above_threshold_indices], maf_above_threshold_indices


def binarize(genotypes: csc_matrix, r2_threshold: float = 0.0) -> tuple[csc_matrix, NDArray]:
    """Round dosages to hard calls and filter by dosage-call agreement.

    Each column is rounded to the nearest integer genotype. The routine then
    computes the per-column Pearson correlation between the original dosage and
    rounded hard calls, retaining columns with correlation at least
    `r2_threshold`.

    **Arguments:**

    - `genotypes`: Input CSC genotype matrix containing dosages.
    - `r2_threshold`: Minimum correlation between dosage and rounded hard calls.

    **Returns:**

    - Tuple `(discretized_genotypes, kept_indices)` where `kept_indices` are
      original variant column positions retained after thresholding.
    """

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
