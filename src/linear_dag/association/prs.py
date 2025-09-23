import numpy as np
import polars as pl
from pathlib import Path
from filelock import FileLock
from linear_dag.core.lineararg import LinearARG


def run_prs(linarg_path: str,
            beta_path: str,
            block_name: str,
            starting_index: int,
            n_variants: int,
            score_cols: list[str],
            out: str,
            logger=None
        ) -> None:
    
    if logger is None:
        from linear_dag.utils.logging import MemoryLogger  # fallback
        logger = MemoryLogger(__name__)
    
    # load data
    logger.info(f"loading LinearARG...")
    linarg = LinearARG.read(linarg_path, block=block_name)
    logger.info(f"reading betas...")
    beta = (
        pl.scan_parquet(beta_path)
        .select(score_cols)
        .slice(starting_index, n_variants)
        .collect()
    )

    # compute partial PRS
    logger.info("computing partial PRS...")
    partial = linarg @ beta
        
    logger.info(f"adding result to output file...")
    add_result_to_numpy(out, partial)
    
    
def add_result_to_numpy(outfile: str, result: np.array):
    outpath = Path(outfile)
    lockfile = outpath.with_suffix(outpath.suffix + ".lock")

    with FileLock(lockfile):
        if outpath.exists():
            existing = np.load(outpath)
            updated = existing + result
        else:
            updated = result
    
        np.save(outpath, updated)