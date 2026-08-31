import logging
import os
import time

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

_DEFAULT_VCF_PROGRESS_SECONDS = 60.0
_DUPLICATE_EXAMPLE_LIMIT = 5
_VARIANT_KEY_COLUMNS = ["CHROM", "POS", "REF", "ALT"]


def _warn_duplicate_variant_keys(variants: pl.DataFrame, logger: logging.Logger) -> None:
    """Warn about repeated emitted variant keys while retaining every column."""
    if variants.height < 2:
        return

    duplicate_groups = (
        variants.group_by(_VARIANT_KEY_COLUMNS, maintain_order=True)
        .agg(pl.len().alias("_count"), pl.col("ID").alias("_ids"))
        .filter(pl.col("_count") > 1)
    )
    if duplicate_groups.is_empty():
        return

    extra_columns = int((duplicate_groups["_count"] - 1).sum())
    examples = []
    for row in duplicate_groups.head(_DUPLICATE_EXAMPLE_LIMIT).iter_rows(named=True):
        ids = ",".join("." if variant_id is None else str(variant_id) for variant_id in row["_ids"])
        examples.append(f"{row['CHROM']}:{row['POS']} {row['REF']}>{row['ALT']} IDs={ids}")
    logger.warning(
        "Duplicate variant audit: duplicate_keys=%s extra_columns=%s; "
        "duplicate columns were retained; examples=%s",
        duplicate_groups.height,
        extra_columns,
        "; ".join(examples),
    )


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
    split_multiallelics: bool = False,
    logger: Optional[logging.Logger] = None,
):
    """Return sample metadata and an iterator over retained sparse VCF columns."""
    if remove_multiallelics and split_multiallelics:
        raise ValueError("remove_multiallelics and split_multiallelics are mutually exclusive")

    active_logger = logger or logging.getLogger(__name__)
    started_at = time.perf_counter()
    progress_interval_raw = os.getenv("KODAMA_VCF_PROGRESS_SECONDS", str(_DEFAULT_VCF_PROGRESS_SECONDS))
    try:
        progress_interval_seconds = float(progress_interval_raw)
        if progress_interval_seconds < 0:
            raise ValueError
    except ValueError:
        active_logger.warning(
            "Invalid KODAMA_VCF_PROGRESS_SECONDS=%r; using %.0f seconds",
            progress_interval_raw,
            _DEFAULT_VCF_PROGRESS_SECONDS,
        )
        progress_interval_seconds = _DEFAULT_VCF_PROGRESS_SECONDS

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

    if sex is not None:
        if not phased:
            raise ValueError("sex/ploidy masking is supported only when phased=True")
        if len(sex) != len(vcf.samples):
            raise ValueError("sex vector length must match the number of loaded VCF samples")
        haplotypes_to_keep = np.ones(2 * len(vcf.samples), dtype=bool)
        haplotypes_to_keep[2 * np.where(sex == 1)[0] + 1] = False
        indices_to_keep = np.flatnonzero(haplotypes_to_keep)
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
        counters = {
            "input_records": 0,
            "biallelic_records": 0,
            "multiallelic_records": 0,
            "emitted_columns": 0,
            "biallelic_columns_emitted": 0,
            "multiallelic_alt_columns_emitted": 0,
            "missing_call_records_removed": 0,
            "zero_carrier_alts_removed": 0,
            "indel_alts_removed": 0,
            "maf_alts_removed": 0,
        }
        emitted_nnz = 0
        last_progress_at = started_at
        active_logger.info(
            "Starting VCF iteration: path=%s region=%s samples=%s phased=%s "
            "split_multiallelics=%s progress_interval_seconds=%s",
            path,
            region,
            len(iids),
            phased,
            split_multiallelics,
            progress_interval_seconds,
        )

        for var in variants:
            if (var.POS < start) or (var.POS > end):
                continue

            counters["input_records"] += 1
            now = time.perf_counter()
            if progress_interval_seconds and now - last_progress_at >= progress_interval_seconds:
                active_logger.info(
                    "VCF progress: records=%s emitted_columns=%s buffered_nonzeros=%s elapsed_seconds=%.1f",
                    counters["input_records"],
                    counters["emitted_columns"],
                    emitted_nnz,
                    now - started_at,
                )
                last_progress_at = now

            is_multiallelic = len(var.ALT) > 1
            counters["multiallelic_records" if is_multiallelic else "biallelic_records"] += 1
            if is_multiallelic:
                if remove_multiallelics:
                    continue
                if not split_multiallelics:
                    variant_id = var.ID if var.ID is not None else "."
                    raise ValueError(
                        "Multiallelic variant encountered in VCF/BCF input: "
                        f"{var.CHROM}:{var.POS} ID={variant_id} REF={var.REF} ALT={','.join(var.ALT)}. "
                        "Multiallelic variants are not supported by default; use "
                        "`remove_multiallelics=True`/`--remove-multiallelics` to exclude them or "
                        "`split_multiallelics=True`/`--split-multiallelics` to split them."
                    )

            alleles = np.asarray(var.genotype.array())[:, :2]
            if phased:
                alleles = np.ravel(alleles)
                if indices_to_keep is not None:
                    alleles = alleles[indices_to_keep]
            if split_multiallelics:
                if np.any(alleles == -1):
                    counters["missing_call_records_removed"] += 1
                    continue
                if phased and np.any(alleles < 0):
                    raise ValueError(
                        f"Unexpected ploidy at {var.CHROM}:{var.POS}; "
                        "provide a sex vector for haploid chromosome calls"
                    )
            elif phased and indices_to_keep is not None:
                assert np.all((alleles == 0) | (alleles == 1)), (
                    "Haplotype vector contains non 0 or 1 values. Check genotype data or sex vector."
                )

            alts = var.ALT if split_multiallelics else var.ALT[:1]
            for alt_index, alt in enumerate(alts, start=1):
                if remove_indels and (len(var.REF) != 1 or len(alt) != 1):
                    counters["indel_alts_removed"] += 1
                    continue

                if split_multiallelics:
                    raw_gts = (alleles == alt_index).astype(np.int8)
                    if not phased:
                        raw_gts = np.sum(raw_gts, axis=1, dtype=np.int8)
                elif phased:
                    raw_gts = alleles
                else:
                    raw_gts = var.gt_types

                if split_multiallelics and not np.any(raw_gts):
                    counters["zero_carrier_alts_removed"] += 1
                    continue

                af = np.mean(raw_gts) / ploidy
                if maf_filter is not None and (af < maf_filter or 1 - af < maf_filter):
                    counters["maf_alts_removed"] += 1
                    continue

                is_flipped = bool(flip_minor_alleles and af > 0.5)
                gts = ploidy - raw_gts if is_flipped else raw_gts
                indices = np.flatnonzero(gts)
                metadata = (var.CHROM, var.POS, var.ID, var.REF, alt)
                counters["emitted_columns"] += 1
                counters[
                    "multiallelic_alt_columns_emitted" if is_multiallelic else "biallelic_columns_emitted"
                ] += 1
                emitted_nnz += len(indices)
                yield indices, gts[indices], is_flipped, metadata

        elapsed = time.perf_counter() - started_at
        active_logger.info(
            "Finished VCF iteration: records=%s emitted_columns=%s buffered_nonzeros=%s elapsed_seconds=%.1f",
            counters["input_records"],
            counters["emitted_columns"],
            emitted_nnz,
            elapsed,
        )
        active_logger.info(
            "VCF read audit: %s elapsed_seconds=%.3f emitted_nonzeros=%s",
            counters,
            elapsed,
            emitted_nnz,
        )

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
    split_multiallelics: bool = False,
    logger: Optional[logging.Logger] = None,
    batch_nnz: int = 1_000_000,
    batch_columns: int = 100_000,
    _index_dtype_max: int = np.iinfo(np.int32).max,
):
    """Stream retained VCF columns into an on-disk CSC representation."""
    if batch_nnz <= 0 or batch_columns <= 0:
        raise ValueError("batch_nnz and batch_columns must be positive")

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
        split_multiallelics=split_multiallelics,
        logger=logger,
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
    variant_info = None

    try:
        with h5py.File(partial_path, "w") as f:
            chunk_size = min(max(1, batch_nnz), 100_000)
            data_dtype = np.int16 if phased else np.int32
            index_dtype = np.int64 if num_rows > _index_dtype_max else np.int32
            indices_dataset = f.create_dataset(
                "indices",
                shape=(0,),
                maxshape=(None,),
                chunks=(chunk_size,),
                dtype=index_dtype,
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
                dtype=index_dtype,
                compression="gzip",
                shuffle=True,
            )
            indptr_dataset[0] = 0

            def upgrade_index_arrays_to_int64():
                nonlocal indices_dataset, indptr_dataset
                if indices_dataset.dtype == np.dtype(np.int64):
                    return
                for name, dataset, copy_batch in (
                    ("indices", indices_dataset, batch_nnz),
                    ("indptr", indptr_dataset, batch_columns),
                ):
                    upgraded = f.create_dataset(
                        f"_{name}64",
                        shape=dataset.shape,
                        maxshape=(None,),
                        chunks=dataset.chunks,
                        dtype=np.int64,
                        compression="gzip",
                        shuffle=True,
                    )
                    for start in range(0, len(dataset), copy_batch):
                        stop = min(start + copy_batch, len(dataset))
                        upgraded[start:stop] = dataset[start:stop]
                    del f[name]
                    f.move(f"_{name}64", name)
                indices_dataset = f["indices"]
                indptr_dataset = f["indptr"]

            def flush():
                nonlocal buffered_nnz
                if index_chunks:
                    indices = np.concatenate(index_chunks).astype(indices_dataset.dtype, copy=False)
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
                index_chunks.append(indices)
                data_chunks.append(values.astype(data_dtype, copy=False))
                buffered_nnz += len(indices)
                total_nnz += len(indices)
                pointer_buffer.append(total_nnz)
                num_variants += 1
                if total_nnz > _index_dtype_max or num_variants > _index_dtype_max:
                    upgrade_index_arrays_to_int64()
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
            if max(num_rows, num_variants) > _index_dtype_max:
                upgrade_index_arrays_to_int64()

            if num_variants == 0:
                for key in list(f.keys()):
                    del f[key]
                f.attrs["is_empty"] = True
            else:
                f.create_dataset("shape", data=(num_rows, num_variants), compression="gzip", shuffle=True)
                f.create_dataset("flip", data=np.asarray(flip), compression="gzip", shuffle=True)
                if phased and sex is None:
                    output_iids = [iid for iid in iids for _ in range(2)]
                elif phased:
                    output_iids = [
                        iid
                        for iid, sample_sex in zip(iids, sex)
                        for _ in range(1 if sample_sex == 1 else 2)
                    ]
                else:
                    output_iids = iids
                f.create_dataset("iids", data=output_iids, compression="gzip", shuffle=True)
                if sex is not None:
                    f.create_dataset("sex", data=sex, compression="gzip", shuffle=True)
                f.attrs["is_empty"] = False

        if num_variants:
            variant_info = pl.DataFrame(var_table)
            _warn_duplicate_variant_keys(variant_info, logger or logging.getLogger(__name__))
        os.replace(partial_path, output_path)
    except BaseException:
        if os.path.exists(partial_path):
            os.remove(partial_path)
        raise

    if num_variants == 0:
        return None
    return variant_info


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
    split_multiallelics: bool = False,
    logger: Optional[logging.Logger] = None,
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
    - `split_multiallelics`: If `True`, emit one binary/dosage column per ALT.
      This option is mutually exclusive with `remove_multiallelics`. Repeated
      `(CHROM, POS, REF, ALT)` keys are retained and reported after parsing. For
      multi-step compression, keep the multiallelic choice fixed for the run.
    - `logger`: Optional logger for audit counters, timings, and duplicate warnings.

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
        split_multiallelics=split_multiallelics,
        logger=logger,
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

    active_logger = logger or logging.getLogger(__name__)
    started_at = time.perf_counter()
    active_logger.info("Starting array concatenation: columns=%s buffered_nonzeros=%s", len(data), ptrs[-1])
    data = np.concatenate(data)
    idxs = np.concatenate(idxs)
    ptrs = np.array(ptrs)
    active_logger.info(
        "Finished array concatenation: columns=%s nonzeros=%s elapsed_seconds=%.1f",
        len(ptrs) - 1,
        len(data),
        time.perf_counter() - started_at,
    )
    active_logger.info(
        "Starting CSC construction: rows=%s columns=%s nonzeros=%s",
        num_rows,
        len(ptrs) - 1,
        len(data),
    )
    genotypes = csc_matrix((data, idxs, ptrs), shape=(num_rows, len(ptrs) - 1))
    flip = np.array(flip)
    active_logger.info(
        "Finished CSC construction: matrix_shape=%s nnz=%s elapsed_seconds=%.1f",
        genotypes.shape,
        genotypes.nnz,
        time.perf_counter() - started_at,
    )
    _warn_duplicate_variant_keys(v_info, active_logger)

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
