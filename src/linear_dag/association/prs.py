import numpy as np
import polars as pl
import time
from scipy.sparse.linalg import LinearOperator


def run_prs(genotypes: LinearOperator, data: pl.DataFrame, score_cols: list[str], iids: list[str]) -> pl.DataFrame:
    t0 = time.perf_counter()

    t_start = time.perf_counter()
    beta = np.array(data[score_cols])
    t_beta = time.perf_counter() - t_start

    t_start = time.perf_counter()
    prs = genotypes @ beta
    t_matmul = time.perf_counter() - t_start

    t_start = time.perf_counter()
    frame_dict = {"iid": iids}
    for i, score in enumerate(score_cols):
        frame_dict[score] = prs[:, i]
    t_frame = time.perf_counter() - t_start

    t_start = time.perf_counter()
    schema_overrides = {"iid": pl.Utf8} | {score: pl.Float32 for score in score_cols}
    t_schema = time.perf_counter() - t_start

    t_start = time.perf_counter()
    res = pl.DataFrame(frame_dict, schema_overrides=schema_overrides)
    t_df = time.perf_counter() - t_start

    t_total = time.perf_counter() - t0

    timings = {
        "beta_extraction": t_beta,
        "matrix_multiply": t_matmul,
        "frame_dict_build": t_frame,
        "schema_overrides": t_schema,
        "dataframe_creation": t_df,
        "total": t_total,
    }

    # human-friendly printout
    print("run_prs timings (seconds):")
    for name in ("beta_extraction", "matrix_multiply", "frame_dict_build", "schema_overrides", "dataframe_creation", "total"):
        print(f"  {name:20}: {timings[name]:.6f}")
    
    
    
    
    # beta = np.array(data[score_cols])
    # prs = genotypes @ beta
    # frame_dict = {"iid": iids}
    # frame_dict.update({score: prs[:, i] for i, score in enumerate(score_cols)})
    # schema_overrides = {"iid": pl.Utf8} | {score: pl.Float64 for score in score_cols}
    # res = pl.DataFrame(frame_dict, schema_overrides=schema_overrides)
    # return res
