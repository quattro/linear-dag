import logging

from typing import Optional

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from scipy.sparse import coo_matrix

from linear_dag.core.lineararg import load_variant_info
from linear_dag.core.parallel_processing import ParallelOperator


def parquet_to_numpy(
    destination: np.ndarray,
    parquet_path: str,
    score_cols: list,
    linarg_variants: pl.Series,
    beta_variants: pl.Series,
    dtype=np.float32,
):
    """Stream PRS effect sizes from Parquet into shared memory in ARG variant order.

    !!! info

        The alignment is based on variant ID intersection. Variants present in one
        source but not the other are skipped; unmatched destination rows remain
        untouched.

    **Arguments:**

    - `destination`: Shared-memory NumPy view with shape `(m_arg, n_scores)`.
    - `parquet_path`: Path to the score parquet file containing `score_cols` plus `ID`.
    - `score_cols`: Score/effect columns to stream.
    - `linarg_variants`: ARG variant IDs in operator column order.
    - `beta_variants`: Parquet variant IDs in parquet row order.
    - `dtype`: Target dtype used when writing values into `destination`.

    **Returns:**

    - `None`. `destination` is updated in place.
    """
    parq = pq.ParquetFile(parquet_path)

    # Build inner-merge index arrays: ARG rows (row_idx) and Parquet global rows (col_idx)
    row_df = pl.LazyFrame({"id": linarg_variants}).with_row_index("row_idx")
    col_df = pl.LazyFrame({"id": beta_variants}).with_row_index("col_idx")
    merged = row_df.join(col_df, on="id", how="inner").collect()

    if merged.height == 0:
        # Nothing to load; leave destination as-is
        return

    row_idx = merged.get_column("row_idx").to_numpy()  # ARG space
    col_idx = merged.get_column("col_idx").to_numpy()  # Parquet global row indices

    # Precompute row-group offsets in global row coordinates
    n_rows_per_rg = [parq.metadata.row_group(i).num_rows for i in range(parq.num_row_groups)]
    offsets = np.concatenate([[0], np.cumsum(n_rows_per_rg)])  # len = num_row_groups + 1

    # Stream per row-group, take only needed rows, and scatter into destination
    for rg_idx in range(parq.num_row_groups):
        start, end = offsets[rg_idx], offsets[rg_idx + 1]
        mask = (col_idx >= start) & (col_idx < end)
        if not np.any(mask):
            continue

        # Compute in-chunk indices and destination rows
        in_chunk = (col_idx[mask] - start).astype(np.int64)
        dest_rows = row_idx[mask].astype(np.int64)

        # Read only score columns for this row-group
        table = parq.read_row_group(rg_idx, columns=score_cols)

        # Use Arrow take to avoid converting the entire row-group when sparse
        sub = table.take(pa.array(in_chunk))

        # Scatter per trait/score column
        for j, c in enumerate(score_cols):
            try:
                arr = sub[c].to_numpy(zero_copy_only=True)
            except Exception:
                arr = sub[c].to_numpy(zero_copy_only=False)
            if arr.dtype != dtype:
                arr = arr.astype(dtype, copy=False)
            destination[dest_rows, j] = arr


def run_prs(
    linarg_path: str,
    beta_path: str,
    block_metadata: pl.DataFrame,
    score_cols: list[str],
    num_processes: int,
    logger: Optional[logging.Logger] = None,
) -> np.ndarray:
    """Compute polygenic risk scores for selected traits/effect columns.

    !!! info

        Variant alignment is performed by ID so score rows are applied to the
        matching variants in the selected ARG blocks.

    **Arguments:**

    - `linarg_path`: Path to the input
      [`linear_dag.core.lineararg.LinearARG`][] HDF5 file.
    - `beta_path`: Path to parquet file containing variant IDs and score columns.
    - `block_metadata`: Block selection metadata used to restrict the operator.
    - `score_cols`: Parquet columns to treat as score vectors.
    - `num_processes`: Number of worker processes for
      [`linear_dag.core.parallel_processing.ParallelOperator`][].
    - `logger`: Optional logger for progress messages.

    **Returns:**

    - Polars DataFrame with one row per unique individual ID and one score column
      per entry in `score_cols`.
    """
    with ParallelOperator.from_hdf5(
        linarg_path, num_processes=num_processes, block_metadata=block_metadata, max_num_traits=len(score_cols)
    ) as linarg:
        if logger:
            logger.info("Reading in betas and copying to shared memory")
        # 1) Load ARG variant IDs (ID-only, in ARG column order for selected blocks)
        block_names = block_metadata.get_column("block_name").to_list()
        linarg_ids_lf = load_variant_info(linarg_path, block_names, columns="id_only")
        linarg_ids = linarg_ids_lf.collect().get_column("ID").cast(pl.String)

        # 2) Load beta variant IDs from Parquet (assume column name 'ID')
        parq = pq.ParquetFile(beta_path)
        beta_table = parq.read(columns=["ID"])  # reads only ID column
        beta_ids = pl.Series("ID", beta_table.column("ID").to_numpy()).cast(pl.String)

        # 3) Stream-scatter betas into SHM in ARG order
        shm_array = linarg.borrow_variant_data_view()
        parquet_to_numpy(shm_array, beta_path, score_cols, linarg_ids, beta_ids)

        # 4) Compute scores with in-place matmul (consumes betas already in SHM)
        if logger:
            logger.info("Performing scoring")
        k = len(score_cols)
        dummy = np.empty((linarg.shape[1], k), dtype=np.float32)
        prs = linarg._matmat(dummy, in_place=True)
        iids = linarg.iids

    if logger:
        logger.info("Summing haplotype scores to individual scores")
    unique_ids, row_indices = np.unique(iids, return_inverse=True)
    num_ids = len(unique_ids)
    num_cols = len(iids)
    col_indices = np.arange(num_cols)
    data = np.ones(num_cols, dtype=np.int8)
    S = coo_matrix((data, (row_indices, col_indices)), shape=(num_ids, num_cols)).tocsc()
    prs_ind = S @ prs

    frame_dict = {"iid": unique_ids}
    for i, score in enumerate(score_cols):
        frame_dict[score] = prs_ind[:, i]
    result = pl.DataFrame(frame_dict)

    return result
