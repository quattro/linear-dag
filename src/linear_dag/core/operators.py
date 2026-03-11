from typing import Optional, Tuple

import numpy as np
import polars as pl

from scipy.sparse import csr_matrix, eye
from scipy.sparse.linalg import aslinearoperator, LinearOperator


def get_inner_merge_operators(row_ids: pl.Series, col_ids: pl.Series) -> Tuple[LinearOperator, LinearOperator]:
    """
    Build left/right merge operators that align two ID vectors into a shared inner-join space.

    **Arguments:**

    - `row_ids`: Row identifiers to map from.
    - `col_ids`: Column identifiers to map to.

    **Returns:**

    - Tuple `(row_operator, col_operator)` where
      `row_operator @ col_operator` yields the inner-join merge matrix.

    **Raises:**

    - `TypeError`: If `row_ids` and `col_ids` have different dtypes.
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
    Build a sparse merge operator between two identifier vectors.

    **Arguments:**

    - `row_ids`: Row identifiers to map from.
    - `col_ids`: Column identifiers to map to.

    **Returns:**

    - Linear operator `M` with `M[i, j] = 1` when `row_ids[i] == col_ids[j]`.

    **Raises:**

    - `TypeError`: If `row_ids` and `col_ids` have different dtypes.
    """
    row_matrix, col_matrix = get_inner_merge_operators(row_ids, col_ids)
    return row_matrix @ col_matrix


def get_row_filter_operator(merge_operator: LinearOperator):
    """Build a row filter that keeps rows with at least one merge match.

    **Arguments:**

    - `merge_operator`: Merge operator from `get_merge_operator`.

    **Returns:**

    - Linear operator that selects matched rows.
    """
    num_matches = merge_operator @ np.ones(merge_operator.shape[1])
    return aslinearoperator(eye(num_matches > 0))


def get_pairing_matrix(two_n: int) -> LinearOperator:
    """Build a diploid-pairing matrix that sums adjacent haplotype rows.

    **Arguments:**

    - `two_n`: Number of haplotype rows. Must be even.

    **Returns:**

    - CSR matrix with shape `(two_n // 2, two_n)` mapping adjacent haplotypes
      to diploid rows.

    **Raises:**

    - `ValueError`: If `two_n` is odd.
    """
    if two_n % 2 != 0:
        raise ValueError("Number of rows in haploid_operator must be even")
    data = np.ones(two_n, dtype=np.int32)
    indices = np.arange(two_n)
    indptr = np.arange(0, two_n + 1, 2)
    pairing_matrix = csr_matrix((data, indices, indptr), shape=(two_n // 2, two_n))
    return pairing_matrix


def get_diploid_operator(haploid_operator: LinearOperator) -> LinearOperator:
    """
    Convert a haploid genotype operator into a diploid operator by pairing adjacent rows.

    **Arguments:**

    - `haploid_operator`: Haploid operator with rows ordered as adjacent haplotype pairs.

    **Returns:**

    - Diploid operator with half as many rows.

    **Raises:**

    - `ValueError`: If the haploid row count is odd.
    """
    two_n = haploid_operator.shape[0]
    return aslinearoperator(get_pairing_matrix(two_n)) @ haploid_operator


def estimate_column_variance(
    operator: LinearOperator, num_samples: int = 1000, seed: Optional[int] = None
) -> np.ndarray:
    """Estimate per-column variance using a random subset of rows.

    **Arguments:**

    - `operator`: Linear operator to sample.
    - `num_samples`: Number of rows sampled without replacement.
    - `seed`: Optional RNG seed for reproducibility.

    **Returns:**

    - Column variance estimates.

    **Raises:**

    - `ValueError`: If `num_samples` exceeds number of rows in `operator`.
    """
    n, _ = operator.shape
    np.random.seed(seed)
    rows = np.random.choice(n, num_samples, replace=False)
    id_submatrix = np.zeros((num_samples, n))
    id_submatrix.ravel()[np.arange(num_samples) * n + rows] = 1  # eye(n)[rows, :]
    return np.var(id_submatrix @ operator, axis=0)
