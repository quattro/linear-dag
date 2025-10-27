import numpy as np
import polars as pl
import pyarrow.parquet as pq
import pyarrow as pa
from linear_dag.core.parallel_processing import ParallelOperator
from linear_dag.core.lineararg import load_variant_info
import logging
from typing import Optional
from scipy.sparse import coo_matrix
import time


def parquet_to_numpy(
    destination: np.ndarray,
    parquet_path: str,
    score_cols: list,
    linarg_variants: pl.Series,
    beta_variants: pl.Series,
    dtype=np.float32,
    logger: logging.Logger | None = None,
):
    """
    Stream betas from a Parquet file into the shared-memory destination in ARG-variant order.
    """

    log = logger or logging.getLogger(__name__)
    t_total = time.time()

    log.info(f"Opening Parquet file: {parquet_path}")
    t0 = time.time()
    parq = pq.ParquetFile(parquet_path)
    log.info(f"Loaded Parquet metadata in {time.time() - t0:.3f}s "
             f"({parq.num_row_groups} row group(s), {parq.metadata.num_rows} rows)")

    # ---- Join variant IDs ----
    t0 = time.time()
    log.info("Building variant ID join (linarg_variants vs beta_variants)...")
    print(linarg_variants)
    print(beta_variants)
    row_df = pl.LazyFrame({"id": linarg_variants}).with_row_index("row_idx")
    col_df = pl.LazyFrame({"id": beta_variants}).with_row_index("col_idx")
    merged = row_df.join(col_df, on="id", how="inner").collect()
    t_join = time.time() - t0
    log.info(f"Join complete in {t_join:.3f}s "
             f"({merged.height} intersecting variants)")

    if merged.height == 0:
        print("No overlapping variants found — nothing to load.")
        return

    # ---- Extract indices ----
    t0 = time.time()
    row_idx = merged.get_column("row_idx").to_numpy()
    col_idx = merged.get_column("col_idx").to_numpy()
    log.info(f"Extracted index arrays in {time.time() - t0:.3f}s")

    # ---- Precompute row group offsets ----
    t0 = time.time()
    n_rows_per_rg = [parq.metadata.row_group(i).num_rows for i in range(parq.num_row_groups)]
    offsets = np.concatenate([[0], np.cumsum(n_rows_per_rg)])
    log.info(f"Computed row group offsets in {time.time() - t0:.3f}s")

    # ---- Stream + scatter ----
    t_stream_total = 0.0
    t_take_total = 0.0
    t_scatter_total = 0.0

    for rg_idx in range(parq.num_row_groups):
        start, end = offsets[rg_idx], offsets[rg_idx + 1]
        mask = (col_idx >= start) & (col_idx < end)
        if not np.any(mask):
            continue

        in_chunk = (col_idx[mask] - start).astype(np.int64)
        dest_rows = row_idx[mask].astype(np.int64)

        # Read row group
        t_rg = time.time()
        table = parq.read_row_group(rg_idx, columns=score_cols)
        t_stream = time.time() - t_rg
        t_stream_total += t_stream
        log.info(f"Row group {rg_idx}: read in {t_stream:.3f}s")

        # Arrow take
        t_take = time.time()
        sub = table.take(pa.array(in_chunk))
        t_take_total += time.time() - t_take
        log.info(f"Row group {rg_idx}: Arrow take() in {time.time() - t_take:.3f}s")

        # Scatter to destination
        t_scat = time.time()
        for j, c in enumerate(score_cols):
            try:
                arr = sub[c].to_numpy(zero_copy_only=True)
            except Exception:
                arr = sub[c].to_numpy(zero_copy_only=False)
            if arr.dtype != dtype:
                arr = arr.astype(dtype, copy=False)
            destination[dest_rows, j] = arr
        t_scatter_total += time.time() - t_scat
        log.info(f"Row group {rg_idx}: scatter in {time.time() - t_scat:.3f}s")

    log.info(
        f"Finished parquet_to_numpy in {time.time() - t_total:.3f}s "
        f"(join={t_join:.3f}s, read={t_stream_total:.3f}s, "
        f"take={t_take_total:.3f}s, scatter={t_scatter_total:.3f}s)"
    )

def run_prs(
    linarg_path: str,
    beta_path: str,
    block_metadata: pl.DataFrame,
    score_cols: list[str],
    num_processes: int,
    logger: Optional[logging.Logger] = None
) -> pl.DataFrame:
    log = logger or logging.getLogger(__name__)

    with ParallelOperator.from_hdf5(
        linarg_path, num_processes=num_processes, block_metadata=block_metadata, max_num_traits=len(score_cols)
    ) as linarg:

        log.info("Reading in betas and copying to shared memory")

        # ---- 1) Load ARG variant IDs lazily ----
        block_names = block_metadata.get_column("block_name").to_list()
        linarg_lf = (
            load_variant_info(linarg_path, block_names, columns="id_only")
            .select([pl.col("ID").cast(pl.Utf8)])
        )
        log.info(f"Prepared ARG LazyFrame for {len(block_names)} blocks")

        # ---- 2) Load beta variant IDs lazily from Parquet ----
        parq = pq.ParquetFile(beta_path)
        beta_table = parq.read(columns=["ID"])

        # Force Arrow column to string if it came in as binary
        if beta_table.schema.field("ID").type == "binary":
            beta_table = beta_table.set_column(
                0,
                "ID",
                beta_table["ID"].cast(pa.string())
            )

        beta_lf = pl.from_arrow(beta_table).lazy().select([pl.col("ID").cast(pl.Utf8)])
        log.info(f"Prepared beta LazyFrame from Parquet: {beta_path}")

        # ---- 3) Join IDs lazily (zero-copy) ----
        merged_lf = linarg_lf.join(beta_lf, on="ID", how="inner")
        merged = merged_lf.collect()
        log.info(f"Join complete: {merged.height} intersecting variants")

        if merged.height == 0:
            log.warning("No overlapping variants found — returning empty DataFrame")
            return pl.DataFrame({"iid": [], **{c: [] for c in score_cols}})

        # ---- 4) Extract row/col indices for scatter ----
        row_idx = merged.get_column("row_idx").to_numpy() if "row_idx" in merged.columns else np.arange(merged.height)
        col_idx = merged.get_column("col_idx").to_numpy() if "col_idx" in merged.columns else np.arange(merged.height)

        # ---- 5) Stream-scatter betas into shared memory ----
        shm_array = linarg.borrow_variant_data_view()
        parquet_to_numpy(shm_array, beta_path, score_cols, row_idx, col_idx, logger=log)

        # ---- 6) Compute PRS scores ----
        log.info("Performing scoring")
        k = len(score_cols)
        dummy = np.empty((linarg.shape[1], k), dtype=np.float32)
        prs = linarg._matmat(dummy, in_place=True)
        iids = linarg.iids

    # ---- 7) Sum haplotype scores to individual scores ----
    log.info("Summing haplotype scores to individual scores")
    unique_ids, row_indices = np.unique(iids, return_inverse=True)
    num_ids = len(unique_ids)
    num_cols = len(iids)
    col_indices = np.arange(num_cols)
    data = np.ones(num_cols, dtype=np.int8)
    S = coo_matrix((data, (row_indices, col_indices)), shape=(num_ids, num_cols)).tocsc()
    prs_ind = S @ prs

    # ---- 8) Build output Polars DataFrame ----
    frame_dict = {"iid": unique_ids}
    for i, score in enumerate(score_cols):
        frame_dict[score] = prs_ind[:, i]
    result = pl.DataFrame(frame_dict)

    return result