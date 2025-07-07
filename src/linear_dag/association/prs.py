from scipy.sparse.linalg import LinearOperator
import numpy as np
from typing import Optional, Union, Tuple
import polars as pl

def run_prs(genotypes: LinearOperator, 
            data: pl.LazyFrame, 
            score_cols: list[str], 
            iids: list[str]
            ) -> pl.DataFrame:
        
    beta = np.array(data.collect()[score_cols])
    prs = genotypes @ beta
    frame_dict = {'iid': iids}
    frame_dict.update({score: prs[:, i] for i, score in enumerate(score_cols)})
    res = pl.DataFrame(frame_dict)
    return res