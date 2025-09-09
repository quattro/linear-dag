import time
import os
import math
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
    """
    Worker computes partial PRS for `block_name` and writes it into its dedicated
    region inside the shared memory buffer (no locking; region is unique).
    """
    # Attach to SHM
    shm = shared_memory.SharedMemory(name=shm_name)
    try:
        # compute byte offset and shape for this worker's region
        elems_per_slice = n_ind * n_scores
        offset_elems = slice_index * elems_per_slice
        offset_bytes = offset_elems * 8  # float64

        # create view into the region for this slice
        partial_view = np.ndarray(
            (n_ind, n_scores),
            dtype=np.float64,
            buffer=shm.buf,
            offset=offset_bytes
        )

        # zero the region (optional but safe)
        partial_view[:, :] = 0.0

        # Load genotype block
        print(f"{os.getpid()}: start block {block_name}", flush=True)
        linarg = LinearARG.read(hdf5_file, block=block_name)
        print(f"{os.getpid()}: finished reading {block_name}", flush=True)

        # Load beta slice matching this block
        print(f"{os.getpid()}: reading betas {block_name}", flush=True)
        beta = (
            pl.scan_parquet("beta_tmp.parquet")
            .select(score_cols)
            .slice(beta_starting_index, n_variants)
            .collect()
            .to_numpy()
        )
        print(f"{os.getpid()}: finished reading betas {block_name}", flush=True)

        # Compute partial PRS (n_ind, n_scores)
        print(f"{os.getpid()}: computing partial PRS {block_name}", flush=True)
        partial = linarg @ beta
        print(f"{os.getpid()}: finished computing partial PRS {block_name}", flush=True)

        # Write into SHM region (no lock because unique region per block)
        print(f"{os.getpid()}: writing partial to SHM slice {slice_index}", flush=True)
        partial_view[:, :] = partial  # atomic at C-level; region is unique
        print(f"{os.getpid()}: finished writing partial to SHM slice {slice_index}", flush=True)
    finally:
        shm.close()


def run_prs_parallel_shm(
    hdf5_file,
    beta_path,
    score_cols,
    num_workers=None,
    blocks=None,
    chromosomes=None,
):
    start_total = time.time()

    # Step 0: Convert beta to parquet (parent)
    t0 = time.time()
    pl.scan_csv(beta_path, separator="\t").collect().write_parquet("beta_tmp.parquet", row_group_size=50_000)
    print(f"Step 0 (convert to parquet) took {time.time() - t0:.2f}s")

    # Step 1: Get blocks metadata (parent)
    t0 = time.time()
    blocks_meta = list_blocks(hdf5_file)
    if blocks is not None and chromosomes is not None:
        raise ValueError("Specify either blocks or chromosomes, not both.")
    if blocks is not None:
        blocks_meta = blocks_meta.filter(pl.col("block_name").is_in(blocks))
    if chromosomes is not None:
        blocks_meta = blocks_meta.filter(pl.col("chrom").is_in(chromosomes))
    print(f"Step 1 (get blocks) took {time.time() - t0:.2f}s")

    # Step 2: Prepare indexing info and check beta row count
    t0 = time.time()
    block_names = blocks_meta["block_name"].to_list()
    n_variants_list = blocks_meta["n_variants"].to_list()
    # compute starting indices (cumulative)
    starting_indices = [0]
    for nv in n_variants_list[:-1]:
        starting_indices.append(starting_indices[-1] + nv)
    B = len(block_names)
    print(f"Step 2 (indexing) blocks={B} took {time.time() - t0:.2f}s")

    # Step 3: Prepare SHM for per-block partials
    t0 = time.time()
    with h5py.File(hdf5_file, "r") as h5f:
        iids = h5f["iids"][:].astype(str)
    n_ind = len(iids)
    n_scores = len(score_cols)

    # ELEMENTS = B * n_ind * n_scores
    elems_per_slice = n_ind * n_scores
    total_elems = B * elems_per_slice
    total_bytes = total_elems * 8  # float64

    # Sanity check for memory size
    gb = total_bytes / (1024**3)
    print(f"Allocating SHM for {B} partials: approx {gb:.2f} GiB", flush=True)
    if gb > 0.9 * (os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / (1024**3)):
        print("Warning: requested SHM > system memory; consider per-worker partial files instead.", flush=True)

    shm = shared_memory.SharedMemory(create=True, size=total_bytes)
    print(f"Step 3 (created SHM) took {time.time() - t0:.2f}s")

    # Step 4: Prepare tasks with slice_index per block
    t0 = time.time()
    tasks = []
    for idx, (bname, start, nv) in enumerate(zip(block_names, starting_indices, n_variants_list)):
        tasks.append((
            hdf5_file,
            bname,
            start,
            nv,
            score_cols,
            shm.name,
            n_ind,
            n_scores,
            idx,  # slice_index
        ))
    print(f"Step 4 (prepare tasks) took {time.time() - t0:.2f}s")

    # Step 5: Launch pool and run workers
    t0 = time.time()
    ctx = get_context("spawn")
    if num_workers is None:
        num_workers = min(cpu_count(), len(tasks))
    with ctx.Pool(processes=num_workers) as pool:
        pool.starmap(_prs_worker_shm, tasks)
    print(f"Step 5 (worker pool) took {time.time() - t0:.2f}s")

    # Step 6: Reduce partials into final prs (single-threaded)
    t0 = time.time()
    prs = np.zeros((n_ind, n_scores), dtype=np.float64)
    shm_parent = shared_memory.SharedMemory(name=shm.name)
    try:
        for idx in range(B):
            offset_elems = idx * elems_per_slice
            offset_bytes = offset_elems * 8
            part_view = np.ndarray((n_ind, n_scores), dtype=np.float64, buffer=shm_parent.buf, offset=offset_bytes)
            # accumulate
            prs += part_view
    finally:
        shm_parent.close()

    print(f"Step 6 (reduce partials) took {time.time() - t0:.2f}s")

    # Step 7: Build Polars DataFrame (convert haplotype scores to individuals if needed)
    t0 = time.time()
    unique_ids, row_indices = np.unique(iids, return_inverse=True)
    num_ids = len(unique_ids)
    num_cols = len(iids)
    col_indices = np.arange(num_cols)
    data = np.ones(num_cols, dtype=np.int8)
    S = coo_matrix((data, (row_indices, col_indices)), shape=(num_ids, num_cols)).tocsc()
    prs_ind = S @ prs  # sum haplotypes -> per-individual
    print(f"Step 7 (sparse matrix multiplication) took {time.time() - t0:.2f}s")

    # Step 8: DataFrame
    t0 = time.time()
    frame_dict = {"iid": unique_ids}
    for i, score in enumerate(score_cols):
        frame_dict[score] = prs_ind[:, i]
    res = pl.DataFrame(frame_dict)
    print(f"Step 8 (build DataFrame) took {time.time() - t0:.2f}s")

    # Cleanup
    shm.unlink()
    print(f"Total run_prs_parallel_shm took {time.time() - start_total:.2f}s")
    return res
