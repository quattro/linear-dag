from typing import Optional, Tuple

import numpy as np
import polars as pl

from scipy.sparse import csr_matrix, eye
from scipy.sparse.linalg import aslinearoperator, LinearOperator


def get_inner_merge_operators(row_ids: pl.Series, col_ids: pl.Series) -> Tuple[LinearOperator, LinearOperator]:
    """
    Returns a pair of LinearOperators that merge row_ids and col_ids into a shared space.
    """
    if row_ids.dtype != col_ids.dtype:
        raise TypeError("Data types of row_ids and col_ids must match")

    row_df = pl.LazyFrame({"id": row_ids}).with_row_index("row_idx")
    col_df = pl.LazyFrame({"id": col_ids}).with_row_index("col_idx")
    merged_df = row_df.join(col_df, on="id", how="inner").with_row_index("merged_idx").collect()
    row_matrix = csr_matrix(
        (
            np.ones(merged_df.shape[0], dtype=int),
            (merged_df.select("row_idx").to_numpy().flatten(), merged_df.select("merged_idx").to_numpy().flatten()),
        ),
        shape=(row_ids.len(), merged_df.height),
    )
    col_matrix = csr_matrix(
        (
            np.ones(merged_df.shape[0], dtype=int),
            (merged_df.select("merged_idx").to_numpy().flatten(), merged_df.select("col_idx").to_numpy().flatten()),
        ),
        shape=(merged_df.height, col_ids.len()),
    )
    return aslinearoperator(row_matrix), aslinearoperator(col_matrix)


def get_merge_operator(row_ids: pl.Series, col_ids: pl.Series) -> LinearOperator:
    """
    Returns a LinearOperator representing the matrix of size (len(row_ids), len(col_ids))
    with a one in entry (i,j) if row_ids[i] == col_ids[j] and a zero otherwise.
    """
    row_matrix, col_matrix = get_inner_merge_operators(row_ids, col_ids)
    return row_matrix @ col_matrix


def get_row_filter_operator(merge_operator: LinearOperator):
    """Given a merge operator, returns a LinearOperator that filters out
    rows with zero matching columns."""
    num_matches = merge_operator @ np.ones(merge_operator.shape[1])
    return aslinearoperator(eye(num_matches > 0))

def get_pairing_matrix(two_n: int) -> LinearOperator:
    if two_n % 2 != 0:
        raise ValueError("Number of rows in haploid_operator must be even")
    data = np.ones(two_n, dtype=np.int32)
    indices = np.arange(two_n)
    indptr = np.arange(0, two_n + 1, 2)
    pairing_matrix = csr_matrix((data, indices, indptr), shape=(two_n // 2, two_n))
    return pairing_matrix

def get_diploid_operator(haploid_operator: LinearOperator) -> LinearOperator:
    """
    Returns a LinearOperator representing the diploid genotype matrix.
    Assumes that consecutive rows of the haploid_operator are for the same individual.
    If the input operator is normalized and it is desired for the output to also be
    be normalized, divide the output by sqrt(2).
    """
    two_n = haploid_operator.shape[0]
    return aslinearoperator(get_pairing_matrix(two_n)) @ haploid_operator


def estimate_column_variance(
    operator: LinearOperator, num_samples: int = 1000, seed: Optional[int] = None
) -> np.ndarray:
    """Estimates the variance of each column of a LinearOperator over
    a subset of rows of size num_samples.
    """
    n, _ = operator.shape
    np.random.seed(seed)
    rows = np.random.choice(n, num_samples, replace=False)
    id_submatrix = np.zeros((num_samples, n))
    id_submatrix.ravel()[np.arange(num_samples) * n + rows] = 1  # eye(n)[rows, :]
    return np.var(id_submatrix @ operator, axis=0)
