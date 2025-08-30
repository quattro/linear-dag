from typing import Optional

import numpy as np
import polars as pl
import h5py
from multiprocessing import Pool, shared_memory, cpu_count
import os
import time

from scipy.sparse.linalg import LinearOperator
from scipy.stats import chi2

from linear_dag.core.lineararg import list_blocks, LinearARG

from ..core.operators import get_inner_merge_operators
from .util import (
    _get_genotype_variance,
    _get_genotype_variance_explained,
    _impute_missing_with_mean,
    residualize_phenotypes,
)


def get_gwas_beta_se(
    genotypes: LinearOperator,
    y_resid: np.ndarray,
    covariates: np.ndarray,
    num_nonmissing: np.ndarray | None = None,
    num_carriers: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute GWAS effect sizes and SEs in per-allele units.
    - Assumes the first column of `covariates` is all-ones.
    - If `num_carriers` is None, assumes HWE for the denominator.

    Returns (beta, se, allele_counts).
    """
    if not np.allclose(covariates[:, 0], 1):
        raise ValueError("First column of covariates should be all-ones")
    if num_nonmissing is None:
        num_nonmissing = y_resid.shape[0] * np.ones(y_resid.shape[1])

    # Numerator uses pre-merged genotypes and pre-residualized phenotypes
    numerator = genotypes.T @ y_resid / num_nonmissing

    # Denominator, equal across traits despite different missingness
    var_explained, allele_counts = _get_genotype_variance_explained(genotypes, covariates)
    if num_carriers is None:
        denominator = allele_counts - var_explained
    else:
        # assumes diploid
        num_homozygotes = (allele_counts - num_carriers.reshape(*allele_counts.shape)) // 2
        assert allele_counts.shape == num_homozygotes.shape
        denominator = allele_counts + 2 * num_homozygotes - 2 * var_explained
    # Clamp to avoid div-by-zero under extreme missingness/rare variants; assumes diploid in scaling.
    denominator = np.maximum(denominator, 1e-6) / (2 * genotypes.shape[0])  # assumes diploid
    
    var_resid = np.sum(y_resid**2, axis=0) / num_nonmissing
    se = np.sqrt(var_resid.reshape(1, -1) / (denominator * num_nonmissing.reshape(1, -1)))

    return numerator / denominator, se, allele_counts


def _format_sumstats(
    beta: np.ndarray,
    se: np.ndarray,
    variant_info: pl.LazyFrame,
    pheno_cols: list[str],
    add_log10p: bool = False,
) -> pl.LazyFrame:
    """Format summary statistics with optional LOG10P, preserving variant_info order.
    Columns produced per phenotype:
    - <pheno>_BETA, <pheno>_SE (and optional LOG10P). Floats are downcast to Float32.
    """
    def log_chisq_pval(z: np.ndarray) -> np.ndarray:
        return -chi2(1).logsf(z**2) / np.log(10)

    # Stack and downcast to float32 to reduce file size without logic changes
    data = np.hstack((beta, se)).astype(np.float32)
    names = [f"{name}_{suffix}" for suffix in ["BETA", "SE"] for name in pheno_cols]
    if add_log10p:
        data = np.hstack((data, log_chisq_pval(beta / se).astype(np.float32)))
        names.append("LOG10P")
    names_in_order = variant_info.collect_schema().names() + names

    df = pl.LazyFrame(data, schema=names)
    df = pl.concat([variant_info, df], how="horizontal").select(names_in_order)
    return df


def run_gwas(
    genotypes: LinearOperator,
    data: pl.LazyFrame,
    pheno_cols: list[str],
    covar_cols: list[str],
    variant_info: Optional[pl.LazyFrame] = None,
    assume_hwe: bool = True,
) -> pl.LazyFrame:
    """
    Linear regression association scan with covariates.
    - `data` must contain `iid`, `pheno_cols`, and `covar_cols` (first covariate is all-ones).
    - If `assume_hwe` is False, requires a LinearARG with individual nodes.
    Returns a LazyFrame of per-variant summary statistics.
    """
    if not assume_hwe and not hasattr(genotypes, "n_individuals"):
        raise ValueError("If assume_hwe is False, genotypes must be a linear ARG with individual nodes.")

    if not np.allclose(data.select(covar_cols[0]).collect().to_numpy(), 1.0):
        raise ValueError("First column of covar_cols should be '1'")
    
    left_op, right_op = get_inner_merge_operators(
        data.select("iid").cast(pl.Utf8).collect().to_series(), genotypes.iids
    )  # data iids to shared iids, shared iids to genotypes iids
    if left_op.shape[1] == 0:
        raise ValueError("Merge failed between genotype and phenotype data")
    
    if assume_hwe:
        num_carriers = None
    else:
        # assumes diploid
        individuals_to_include = np.where(np.isin(genotypes.iids[::2], 
            data.select("iid").collect().to_numpy().astype(str)))[0]
        num_carriers = genotypes.number_of_carriers(individuals_to_include)

    phenotypes = data.select(pheno_cols).collect().to_numpy()
    covariates = data.select(covar_cols).collect().to_numpy()

    # Merge to shared sample space and residualize phenotypes
    covariates = left_op.T @ covariates
    phenotypes = left_op.T @ phenotypes
    is_missing = np.isnan(phenotypes)
    num_nonmissing = np.sum(~is_missing, axis=0)
    phenotypes.ravel()[is_missing.ravel()] = 0
    covariates = _impute_missing_with_mean(covariates)
    y_resid = residualize_phenotypes(phenotypes, covariates, is_missing)

    # Genotypes merged to shared space for matmuls
    genotypes_merged = right_op @ genotypes

    beta, se, allele_counts = get_gwas_beta_se(
        genotypes_merged,
        y_resid,
        covariates,
        num_nonmissing,
        num_carriers=num_carriers,
    )

    if variant_info is None:
        variant_info = pl.LazyFrame()
    variant_info = variant_info.with_columns(
        pl.Series("A1FREQ", allele_counts.ravel() / genotypes.shape[0]).cast(pl.Float32),
    )

    return _format_sumstats(beta, se, variant_info, pheno_cols, add_log10p=False)


def run_gwas_parallel(
    hdf5_file: str,
    data: pl.LazyFrame,
    pheno_cols: list[str],
    covar_cols: list[str],
    output_prefix: str,
    assume_hwe: bool = True,
    num_workers: int | None = None,
    verbose: bool = False,
):
    """
    Prepare merged/residualized inputs for parallel GWAS workers.

    - Loads iids from HDF5 and joins to subset rows and add 'arg_idx'.
    - Extracts phenotype/covariate matrices to NumPy.
    - Zeros missing phenotypes and residualizes on covariates.

    Returns:
        (merged_data: pl.LazyFrame, covariates: np.ndarray, y_resid: np.ndarray,
         is_missing: np.ndarray, num_nonmissing: np.ndarray)
    """
    t0 = time.perf_counter()
    # Load iids from HDF5
    with h5py.File(hdf5_file, "r") as h5f:
        if not assume_hwe:
            has_individuals = any(
                isinstance(h5f[k], h5py.Group) and ("n_individuals" in h5f[k].attrs)
                for k in h5f.keys()
            )
            if not has_individuals:
                raise ValueError(
                    "If assume_hwe is False, the HDF5 must include individual nodes (n_individuals)."
                )
        iids = h5f["iids"][:].astype(str)

    # Get rows that are present in the genotype data
    iid_df = pl.LazyFrame({"iid": pl.Series(iids).unique(maintain_order=True)})
    merged = (
        data.with_columns(pl.col("iid").cast(pl.Utf8))
        .join(iid_df, on="iid", how="inner", maintain_order="right")
    )

    phenotypes = merged.select(pheno_cols).collect().to_numpy().copy()
    covariates = merged.select(covar_cols).collect().to_numpy()
    iids = merged.select("iid").collect().to_numpy().astype(str)

    is_missing = np.isnan(phenotypes)
    num_nonmissing = 2 * np.sum(~is_missing, axis=0) # assumes diploid
    phenotypes.ravel()[is_missing.ravel()] = 0
    covariates = _impute_missing_with_mean(covariates)
    phenotypes = residualize_phenotypes(phenotypes, covariates, is_missing)

    # Shared memory setup for phenotypes (n x p), covariates (n x c), iids (n x 1)
    ph_shape, ph_dtype = phenotypes.shape, phenotypes.dtype
    cv_shape, cv_dtype = covariates.shape, covariates.dtype
    iids_shape, iids_dtype = iids.shape, iids.dtype

    shm_ph = shared_memory.SharedMemory(create=True, size=phenotypes.nbytes)
    shm_cv = shared_memory.SharedMemory(create=True, size=covariates.nbytes)
    shm_iids = shared_memory.SharedMemory(create=True, size=iids.nbytes)

    ph_view = np.ndarray(ph_shape, dtype=ph_dtype, buffer=shm_ph.buf)
    cv_view = np.ndarray(cv_shape, dtype=cv_dtype, buffer=shm_cv.buf)
    iids_view = np.ndarray(iids_shape, dtype=iids_dtype, buffer=shm_iids.buf)
    ph_view[:] = phenotypes
    cv_view[:] = covariates
    iids_view[:] = iids

    # Descriptors to pass to workers (name, shape, dtype.str)
    ph_desc = (shm_ph.name, ph_shape, ph_dtype.str)
    cv_desc = (shm_cv.name, cv_shape, cv_dtype.str)
    iids_desc = (shm_iids.name, iids_shape, iids_dtype.str)


    blocks = list_blocks(hdf5_file)["block_name"].to_list()

    if num_workers is None:
        num_workers = cpu_count()

    # Worker pool invocation with ensured SHM cleanup
    prep_sec = time.perf_counter() - t0
    if verbose:
        print(
            f"run_gwas_parallel: blocks={len(blocks)} workers={num_workers} "
            f"prep={prep_sec:.3f}s shapes ph={ph_shape} cv={cv_shape}"
        )
    try:
        with Pool(processes=num_workers) as p:
            worker_times = p.starmap(
                _gwas_worker,
                [
                    (ph_desc, cv_desc, iids_desc, pheno_cols, num_nonmissing, hdf5_file, block, output_prefix, assume_hwe)
                    for block in blocks
                ],
            )
            if verbose:
                # Sum timing categories across workers and print a compact summary
                keys = [
                    "read", "merge_yc", "geno", "carriers", "stats", "format", "collect", "write"
                ]
                totals = {k: 0.0 for k in keys}
                for w in worker_times:
                    for k in keys:
                        totals[k] += w.get(k, 0.0)
                print(
                    "timings(sum): "
                    + " ".join([f"{k}={totals[k]:.3f}s" for k in keys])
                )
    finally:
        # Clean up shared memory
        ph_view = cv_view = iids_view = None
        shm_ph.close(); shm_ph.unlink()
        shm_cv.close(); shm_cv.unlink()
        shm_iids.close(); shm_iids.unlink()


def _gwas_worker(
    ph_desc: tuple[str, tuple[int, ...], str],
    cv_desc: tuple[str, tuple[int, ...], str],
    iids_desc: tuple[str, tuple[int, ...], str],
    pheno_cols: list[str],
    num_nonmissing: np.ndarray,
    hdf5_file: str,
    block_name: str,
    output_prefix: str,
    assume_hwe: bool,
):
    """Attach to SHM arrays and compute per-worker outputs.
    Returns a dict of runtimes for profiling (no prints here).
    """
    ph_name, ph_shape, ph_dtype_str = ph_desc
    cv_name, cv_shape, cv_dtype_str = cv_desc
    iids_name, iids_shape, iids_dtype_str = iids_desc

    shm_ph = shared_memory.SharedMemory(name=ph_name)
    shm_cv = shared_memory.SharedMemory(name=cv_name)
    shm_iids = shared_memory.SharedMemory(name=iids_name)
    try:
        t0 = time.perf_counter()
        phenotypes = np.ndarray(ph_shape, dtype=np.dtype(ph_dtype_str), buffer=shm_ph.buf)
        covariates = np.ndarray(cv_shape, dtype=np.dtype(cv_dtype_str), buffer=shm_cv.buf)
        iids = np.ndarray(iids_shape, dtype=np.dtype(iids_dtype_str), buffer=shm_iids.buf)

        # Load genotypes for this block (with variant metadata)
        linarg = LinearARG.read(hdf5_file, block=block_name, load_metadata=True)
        t_read = time.perf_counter()

        # Merge arrays to shared sample space (iids are in genotype order from run_gwas_parallel)
        left_op, right_op = get_inner_merge_operators(
            pl.Series(iids.ravel().astype(str)), linarg.iids
        )
        if left_op.shape[1] == 0:
            raise ValueError("Merge failed between genotype and phenotype data in worker")

        # Merge y/covariates to shared space
        phenotypes_m = left_op.T @ phenotypes
        covariates_m = left_op.T @ covariates
        t_merge_yc = time.perf_counter()

        # Merge genotypes to shared space
        genotypes_m = right_op @ linarg
        t_merge_g = time.perf_counter()
        
        if assume_hwe:
            num_carriers = None
        else:
            flat_iids = iids.ravel().astype(str)
            individuals_to_include = np.where(np.isin(linarg.iids[::2], flat_iids))[0] # assumes diploid
            num_carriers = linarg.number_of_carriers(individuals_to_include)
        t_car = time.perf_counter()
        
        beta, se, allele_counts = get_gwas_beta_se(
            genotypes_m, phenotypes_m, covariates_m,
            num_nonmissing=num_nonmissing,
            num_carriers=num_carriers,
        )
        t_stats = time.perf_counter()

        # Variant info and phenotype names
        variant_info = linarg.variants if linarg.variants is not None else pl.LazyFrame()
        variant_info = variant_info.with_columns(
            pl.Series("A1FREQ", allele_counts.ravel() / genotypes_m.shape[0]).cast(pl.Float32)
        )

        # Format and write per-block Parquet
        lf = _format_sumstats(beta, se, variant_info, pheno_cols, add_log10p=False)
        t_fmt = time.perf_counter()
        out_path = os.path.join(output_prefix, f"{block_name}.parquet")
        df = lf.collect()
        t_collect = time.perf_counter()
        df.write_parquet(out_path, compression="uncompressed")
        t_write = time.perf_counter()
        return {
            "read": t_read - t0,
            "merge_yc": t_merge_yc - t_read,
            "geno": t_merge_g - t_merge_yc,
            "carriers": t_car - t_merge_g,
            "stats": t_stats - t_car,
            "format": t_fmt - t_stats,
            "collect": t_collect - t_fmt,
            "write": t_write - t_collect,
        }
    finally:
        shm_ph.close()
        shm_cv.close()
        shm_iids.close()
