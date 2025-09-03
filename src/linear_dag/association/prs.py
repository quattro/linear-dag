import time
import numpy as np
import polars as pl
import h5py
from scipy.sparse import coo_matrix
from multiprocessing import Pool, shared_memory, Lock, cpu_count
from linear_dag.core.lineararg import list_blocks, LinearARG

# global lock for workers
_global_lock = None

def _init_worker(lock):
    global _global_lock
    _global_lock = lock

def _prs_worker(hdf5_file, beta_path, block_name, beta_starting_index, n_variants, score_cols, shm_name, res_shape):
    global _global_lock
    shm = shared_memory.SharedMemory(name=shm_name)
    prs = np.ndarray(res_shape, dtype=np.float64, buffer=shm.buf)

    # Load genotype block
    linarg = LinearARG.read(hdf5_file, block=block_name)

    # Load beta slice
    beta = (
        pl.scan_csv(beta_path, separator='\t')
        .select(score_cols)
        .slice(beta_starting_index, n_variants)
        .collect()
        .to_numpy()
    )

    # Compute partial PRS
    partial = linarg @ beta

    # Update shared memory safely
    with _global_lock:
        prs[:, :] += partial

    shm.close()

def run_prs_parallel(hdf5_file, beta_path, score_cols, num_workers=None, blocks=None, chromosomes=None):
    start_total = time.time()

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
    starting_index = pl.Series(
        "starting_index",
        [0] + blocks_meta["n_variants"].to_list()[:-1]
    )
    blocks_meta = blocks_meta.with_columns(starting_index)
    print(f"Step 4 (starting indices) took {time.time() - t0:.2f}s")

    # Step 5: Prepare tasks
    t0 = time.time()
    tasks = [
        (
            hdf5_file,
            beta_path,
            row["block_name"],
            row["starting_index"],
            row["n_variants"],
            score_cols,
            shm.name,
            res_shape,
        )
        for row in blocks_meta.iter_rows(named=True)
    ]
    print(f"Step 5 (prepare tasks) took {time.time() - t0:.2f}s")

    # Step 6: Launch worker pool
    t0 = time.time()
    if num_workers is None:
        num_workers = cpu_count()
    with Pool(
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
    prs_ind = S @ prs
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
