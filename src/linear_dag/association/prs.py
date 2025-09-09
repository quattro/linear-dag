import time
import os
import numpy as np
import polars as pl
import h5py
from scipy.sparse import coo_matrix
from multiprocessing import Pool, shared_memory, Lock, cpu_count
from linear_dag.core.lineararg import list_blocks, LinearARG
import csv

# global lock for workers
_global_lock = None

def _init_worker(lock):
    global _global_lock
    _global_lock = lock

def _prs_worker(hdf5_file, block_name, beta_starting_index, n_variants, score_cols, shm_name, res_shape):
    global _global_lock
    shm = shared_memory.SharedMemory(name=shm_name)
    prs = np.ndarray(res_shape, dtype=np.float64, buffer=shm.buf)

    # Load genotype block
    print(f"{os.getpid()}: start block {block_name}", flush=True)
    linarg = LinearARG.read(hdf5_file, block=block_name)
    print(f"{os.getpid()}: finished reading {block_name}", flush=True)

    # Load beta slice
    print(f"{os.getpid()}: reading betas {block_name}", flush=True)
    beta = (
        pl.scan_parquet('beta_tmp.parquet')
        .select(score_cols)
        .slice(beta_starting_index, n_variants)
        .collect()
        .to_numpy()
    )
    print(f"{os.getpid()}: finished reading betas {block_name}", flush=True)
    
    # Compute partial PRS
    print(f"{os.getpid()}: computing partial PRS {block_name}", flush=True)
    partial = linarg @ beta
    print(f"{os.getpid()}: finished computing partial PRS {block_name}", flush=True)

    # Update shared memory safely
    print(f"{os.getpid()}: updating shared memory {block_name}", flush=True)
    with _global_lock:
        prs[:, :] += partial
    print(f"{os.getpid()}: finished updating shared memory {block_name}", flush=True)

    print(f"{os.getpid()}: closing {block_name}", flush=True)
    shm.close()
    print(f"{os.getpid()}: finished closing {block_name}", flush=True)

def run_prs_parallel(hdf5_file, beta_path, score_cols, num_workers=None, blocks=None, chromosomes=None):
    start_total = time.time()
    
    # Step 0: Convert to parquet
    t0 = time.time()
    pl.scan_csv(beta_path, separator="\t").collect().write_parquet("beta_tmp.parquet", row_group_size=50_000)
    print(f"Step 0 (convert to parquet) took {time.time() - t0:.2f}s")


    # Step 1: Get blocks
    t0 = time.time()
    blocks_meta = list_blocks(hdf5_file)
    if blocks is not None and chromosomes is not None:
        raise ValueError("Specify either blocks or chromosomes, not both.")
    if blocks is not None:
        blocks_meta = blocks_meta.filter(pl.col("block_name").is_in(blocks))
    if chromosomes is not None:
        blocks_meta = blocks_meta.filter(pl.col("chrom").is_in(chromosomes))
    print(f"Step 1 (get blocks) took {time.time() - t0:.2f}s")

    # Step 2: Check beta row count
    # t0 = time.time()
    # with open(beta_path, "r") as f:
    #     n_rows = sum(1 for _ in f) - 1
    # n_variants = blocks_meta["n_variants"].sum()
    # if n_rows != n_variants:
    #     raise ValueError(f"Number of variants in ARG ({n_variants}) != number of beta rows ({n_rows})")
    # print(f"Step 2 (check beta rows) took {time.time() - t0:.2f}s")

    # Step 3: Prepare shared memory
    t0 = time.time()
    with h5py.File(hdf5_file, "r") as h5f:
        iids = h5f["iids"][:].astype(str)
    n_ind = len(iids)
    n_scores = len(score_cols)
    shm = shared_memory.SharedMemory(create=True, size=n_ind * n_scores * 8)
    res_shape = (n_ind, n_scores)
    prs = np.ndarray(res_shape, dtype=np.float64, buffer=shm.buf)
    prs[:, :] = 0.0
    lock = Lock()
    print(f"Step 3 (shared memory) took {time.time() - t0:.2f}s")

    # Step 4: Compute starting indices
    t0 = time.time()
    n_vars = blocks_meta["n_variants"].to_list()
    offsets = np.cumsum([0] + n_vars[:-1])
    blocks_meta = blocks_meta.with_columns(pl.Series("starting_index", offsets))
    print(f"Step 4 (starting indices) took {time.time() - t0:.2f}s")

    # Step 5: Prepare tasks
    t0 = time.time()
    tasks = [
        (
            hdf5_file,
            row["block_name"],
            row["starting_index"],
            row["n_variants"],
            score_cols,
            shm.name,
            res_shape,
        )
        for row in blocks_meta.iter_rows(named=True)
    ]
    if len(tasks) == 0:
        shm.close()
        shm.unlink()
        raise ValueError(
            "No tasks were generated. Check --chrom or --blocks"
        )
    print(f"Step 5 (prepare tasks) took {time.time() - t0:.2f}s")

    # Step 6: Launch worker pool
    t0 = time.time()
    
    from multiprocessing import get_context
    ctx = get_context("spawn")  # use 'spawn' to avoid fork-related issues
    
    if num_workers is None:
        num_workers = cpu_count()
    with ctx.Pool(
        min(num_workers, len(tasks)),
        initializer=_init_worker,
        initargs=(lock,),
    ) as pool:
        pool.starmap(_prs_worker, tasks)
    print(f"Step 6 (worker pool) took {time.time() - t0:.2f}s")

    # Step 7: Build sparse matrix and multiply
    t0 = time.time()
    unique_ids, row_indices = np.unique(iids, return_inverse=True)
    num_ids = len(unique_ids)
    num_cols = len(iids)
    col_indices = np.arange(num_cols)
    data = np.ones(num_cols, dtype=np.int8)
    S = coo_matrix((data, (row_indices, col_indices)), shape=(num_ids, num_cols)).tocsc()
    prs_ind = S @ prs # sum haplotypes
    print(f"Step 7 (sparse matrix multiplication) took {time.time() - t0:.2f}s")

    # Step 8: Build Polars DataFrame
    t0 = time.time()
    frame_dict = {"iid": unique_ids}
    for i, score in enumerate(score_cols):
        frame_dict[score] = prs_ind[:, i]
    res = pl.DataFrame(frame_dict)
    print(f"Step 8 (build DataFrame) took {time.time() - t0:.2f}s")

    shm.close()
    shm.unlink()
    print(f"Total run_prs_parallel took {time.time() - start_total:.2f}s")

    return res
