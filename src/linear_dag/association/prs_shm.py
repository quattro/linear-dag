import os
import numpy as np
import polars as pl
import h5py
from scipy.sparse import coo_matrix
from multiprocessing import get_context, shared_memory, cpu_count
from linear_dag.core.lineararg import list_blocks, LinearARG

def _prs_worker_shm(
    hdf5_file,
    block_name,
    beta_starting_index,
    n_variants,
    score_cols,
    shm_name,
    n_ind,
    n_scores,
    slice_index,
):
    # Force single-threaded BLAS
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

    shm = shared_memory.SharedMemory(name=shm_name)
    try:
        elems_per_slice = n_ind * n_scores
        offset_bytes = slice_index * elems_per_slice * 4  # float32

        # SHM region for this block, float32
        partial_view = np.ndarray((n_ind, n_scores), dtype=np.float32, buffer=shm.buf, offset=offset_bytes)

        # Load genotype block
        linarg = LinearARG.read(hdf5_file, block=block_name)

        # Load beta slice for this block
        beta = (
            pl.scan_parquet("beta_tmp.parquet")
            .select(score_cols)
            .slice(beta_starting_index, n_variants)
            .collect()
            .to_numpy()
        )

        # Ensure correct beta shape
        beta = beta[:linarg.shape[1], :]

        # Compute partial PRS (float32)
        partial_view[:, :] = (linarg @ beta).astype(np.float32)

    finally:
        shm.close()

def run_prs_parallel_shm(
    hdf5_file,
    beta_path,
    score_cols,
    num_workers=None,
    blocks=None,
    chromosomes=None,
    max_workers=4,  # limit concurrent workers
):
    # Convert beta to parquet
    pl.scan_csv(beta_path, separator="\t").collect().write_parquet("beta_tmp.parquet", row_group_size=50_000)

    # Get blocks metadata
    blocks_meta = list_blocks(hdf5_file)
    if blocks is not None and chromosomes is not None:
        raise ValueError("Specify either blocks or chromosomes, not both.")
    if blocks is not None:
        blocks_meta = blocks_meta.filter(pl.col("block_name").is_in(blocks))
    if chromosomes is not None:
        blocks_meta = blocks_meta.filter(pl.col("chrom").is_in(chromosomes))

    # Indexing
    block_names = blocks_meta["block_name"].to_list()
    n_variants_list = blocks_meta["n_variants"].to_list()
    starting_indices = [0]
    for nv in n_variants_list[:-1]:
        starting_indices.append(starting_indices[-1] + nv)
    B = len(block_names)

    # Prepare SHM (float32)
    with h5py.File(hdf5_file, "r") as h5f:
        iids = h5f["iids"][:].astype(str)
    n_ind = len(iids)
    n_scores = len(score_cols)
    elems_per_slice = n_ind * n_scores
    shm = shared_memory.SharedMemory(create=True, size=B * elems_per_slice * 4)  # float32

    # Tasks
    tasks = [
        (hdf5_file, bname, start, nv, score_cols, shm.name, n_ind, n_scores, idx)
        for idx, (bname, start, nv) in enumerate(zip(block_names, starting_indices, n_variants_list))
    ]

    # Pool with limited workers
    ctx = get_context("spawn")
    if num_workers is None:
        num_workers = min(max_workers, len(tasks))
    with ctx.Pool(processes=num_workers) as pool:
        pool.starmap(_prs_worker_shm, tasks)

    # Reduce partials
    prs = np.zeros((n_ind, n_scores), dtype=np.float32)
    shm_parent = shared_memory.SharedMemory(name=shm.name)
    try:
        for idx in range(B):
            offset_bytes = idx * elems_per_slice * 4
            part_view = np.ndarray((n_ind, n_scores), dtype=np.float32, buffer=shm_parent.buf, offset=offset_bytes)
            prs += part_view
    finally:
        shm_parent.close()
        shm.unlink()

    # Sum haplotypes
    unique_ids, row_indices = np.unique(iids, return_inverse=True)
    S = coo_matrix((np.ones_like(row_indices, dtype=np.int8), (row_indices, np.arange(len(iids)))), shape=(len(unique_ids), len(iids))).tocsc()
    prs_ind = S @ prs

    # Polars DataFrame
    frame_dict = {"iid": unique_ids}
    for i, score in enumerate(score_cols):
        frame_dict[score] = prs_ind[:, i]
    return pl.DataFrame(frame_dict)
