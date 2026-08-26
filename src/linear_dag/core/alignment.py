# pattern: Functional Core

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import numpy as np
import polars as pl

from jaxtyping import Array


@dataclass(frozen=True)
class IidAlignment:
    """Index representation of an exact IID inner join.

    `left_indices[k]` and `right_indices[k]` identify the source rows that
    form merged row `k`. The order matches the Polars inner-join order used by
    [`linear_dag.core.operators.get_inner_merge_operators`][].

    !!! Example
        ```python
        import polars as pl

        from linear_dag.core.alignment import get_iid_alignment

        alignment = get_iid_alignment(
            pl.Series("iid", ["sample-2", "sample-1"]),
            pl.Series("iid", ["sample-1"]),
        )
        ```
    """

    left_indices: np.ndarray
    right_indices: np.ndarray
    n_left: int
    n_right: int

    def __post_init__(self) -> None:
        left_indices = np.asarray(self.left_indices, dtype=np.int64)
        right_indices = np.asarray(self.right_indices, dtype=np.int64)
        object.__setattr__(self, "left_indices", left_indices)
        object.__setattr__(self, "right_indices", right_indices)
        object.__setattr__(self, "n_left", int(self.n_left))
        object.__setattr__(self, "n_right", int(self.n_right))

        if left_indices.shape != right_indices.shape:
            raise ValueError("left_indices and right_indices must have the same shape")
        if left_indices.ndim != 1:
            raise ValueError("left_indices and right_indices must be rank-1 arrays")
        _validate_indices("left_indices", left_indices, self.n_left)
        _validate_indices("right_indices", right_indices, self.n_right)

    @property
    def n_merged(self) -> int:
        """Return the number of rows in the merged IID space."""
        return int(self.left_indices.shape[0])

    def gather_left(self, values: np.ndarray) -> np.ndarray:
        """Apply `left_op.T` semantics to NumPy values.

        **Arguments:**

        - `values`: Array in the left-side row space.

        **Returns:**

        - Array gathered into merged IID order.
        """
        values, was_vector = _as_rank2_numpy(values, expected_rows=self.n_left)
        result = values[self.left_indices]
        return result[:, 0] if was_vector else result

    def gather_right(self, values: np.ndarray) -> np.ndarray:
        """Apply `right_op` semantics to NumPy values.

        **Arguments:**

        - `values`: Array in the right-side row space.

        **Returns:**

        - Array gathered into merged IID order.
        """
        values, was_vector = _as_rank2_numpy(values, expected_rows=self.n_right)
        result = values[self.right_indices]
        return result[:, 0] if was_vector else result

    def scatter_left(self, values: np.ndarray) -> np.ndarray:
        """Apply `left_op` semantics to NumPy values.

        **Arguments:**

        - `values`: Array in merged IID order.

        **Returns:**

        - Array scattered into the left-side row space.
        """
        values, was_vector = _as_rank2_numpy(values, expected_rows=self.n_merged)
        result = np.zeros((self.n_left, values.shape[1]), dtype=values.dtype)
        np.add.at(result, self.left_indices, values)
        return result[:, 0] if was_vector else result

    def scatter_right(self, values: np.ndarray) -> np.ndarray:
        """Apply `right_op.T` semantics to NumPy values.

        **Arguments:**

        - `values`: Array in merged IID order.

        **Returns:**

        - Array scattered into the right-side row space.
        """
        values, was_vector = _as_rank2_numpy(values, expected_rows=self.n_merged)
        result = np.zeros((self.n_right, values.shape[1]), dtype=values.dtype)
        np.add.at(result, self.right_indices, values)
        return result[:, 0] if was_vector else result

    def gather_left_jax(self, values: Any) -> Array:
        """Apply `left_op.T` semantics to JAX values.

        **Arguments:**

        - `values`: Array in the left-side row space.

        **Returns:**

        - Array gathered into merged IID order.
        """
        values, was_vector = _as_rank2_jax(values, expected_rows=self.n_left)
        result = values[jnp.asarray(self.left_indices, dtype=jnp.int32)]
        return result[:, 0] if was_vector else result

    def gather_right_jax(self, values: Any) -> Array:
        """Apply `right_op` semantics to JAX values.

        **Arguments:**

        - `values`: Array in the right-side row space.

        **Returns:**

        - Array gathered into merged IID order.
        """
        values, was_vector = _as_rank2_jax(values, expected_rows=self.n_right)
        result = values[jnp.asarray(self.right_indices, dtype=jnp.int32)]
        return result[:, 0] if was_vector else result

    def scatter_left_jax(self, values: Any) -> Array:
        """Apply `left_op` semantics to JAX values.

        **Arguments:**

        - `values`: Array in merged IID order.

        **Returns:**

        - Array scattered into the left-side row space.
        """
        values, was_vector = _as_rank2_jax(values, expected_rows=self.n_merged)
        result = jnp.zeros((self.n_left, values.shape[1]), dtype=values.dtype)
        result = result.at[jnp.asarray(self.left_indices, dtype=jnp.int32)].add(values)
        return result[:, 0] if was_vector else result

    def scatter_right_jax(self, values: Any) -> Array:
        """Apply `right_op.T` semantics to JAX values.

        **Arguments:**

        - `values`: Array in merged IID order.

        **Returns:**

        - Array scattered into the right-side row space.
        """
        values, was_vector = _as_rank2_jax(values, expected_rows=self.n_merged)
        result = jnp.zeros((self.n_right, values.shape[1]), dtype=values.dtype)
        result = result.at[jnp.asarray(self.right_indices, dtype=jnp.int32)].add(values)
        return result[:, 0] if was_vector else result


def get_iid_alignment(left_ids: pl.Series, right_ids: pl.Series) -> IidAlignment:
    """Build index arrays that align two IID vectors onto their inner join.

    **Arguments:**

    - `left_ids`: Identifiers for the left-side row space.
    - `right_ids`: Identifiers for the right-side row space.

    **Returns:**

    - [`linear_dag.core.alignment.IidAlignment`][] with merged-space index arrays.

    **Raises:**

    - `TypeError`: If identifier dtypes differ.
    """
    if left_ids.dtype != right_ids.dtype:
        raise TypeError("Data types of left_ids and right_ids must match")

    left_df = pl.LazyFrame({"id": left_ids}).with_row_index("left_idx")
    right_df = pl.LazyFrame({"id": right_ids}).with_row_index("right_idx")
    merged = left_df.join(right_df, on="id", how="inner").collect()
    return IidAlignment(
        left_indices=merged.get_column("left_idx").to_numpy(),
        right_indices=merged.get_column("right_idx").to_numpy(),
        n_left=left_ids.len(),
        n_right=right_ids.len(),
    )


def _validate_indices(name: str, indices: np.ndarray, size: int) -> None:
    if size < 0:
        raise ValueError("sizes must be non-negative")
    if indices.size == 0:
        return
    if int(np.min(indices)) < 0 or int(np.max(indices)) >= size:
        raise ValueError(f"{name} entries must be in [0, {size})")


def _as_rank2_numpy(values: np.ndarray, *, expected_rows: int) -> tuple[np.ndarray, bool]:
    array = np.asarray(values)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
        was_vector = True
    elif array.ndim == 2:
        was_vector = False
    else:
        raise ValueError(f"expected rank 1 or 2 input, got rank {array.ndim}")
    if array.shape[0] != expected_rows:
        raise ValueError(f"expected leading dimension {expected_rows}, got {array.shape[0]}")
    return array, was_vector


def _as_rank2_jax(values: Any, *, expected_rows: int) -> tuple[Array, bool]:
    array = jnp.asarray(values)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
        was_vector = True
    elif array.ndim == 2:
        was_vector = False
    else:
        raise ValueError(f"expected rank 1 or 2 input, got rank {array.ndim}")
    if array.shape[0] != expected_rows:
        raise ValueError(f"expected leading dimension {expected_rows}, got {array.shape[0]}")
    return array, was_vector
