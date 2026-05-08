# pattern: Functional Core

from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import polars as pl

from .operator import Backend

_KERNELS_MESSAGE = "JaxLinearARG numerical kernels are implemented in Phase 2"


def _shape_tuple(shape: Any) -> tuple[int, int]:
    return tuple(shape)


def split_blocks_by_n_entries(metadata: pl.DataFrame, num_devices: int) -> tuple[tuple[int, int], ...]:
    """Split contiguous block ranges by cumulative `n_entries` weight."""
    if num_devices < 1:
        raise ValueError(f"num_devices must be positive. Observed {num_devices}.")

    size_array = metadata.get_column("n_entries").to_numpy()
    size_cumsum = np.insert(np.cumsum(size_array), 0, 0)
    chunk_size = size_cumsum[-1] / num_devices

    block_indices = []
    for i in range(1, num_devices):
        target_sum = i * chunk_size
        idx = np.searchsorted(size_cumsum, target_sum)
        block_indices.append(idx)
    block_indices.append(len(size_cumsum) - 1)

    block_indices = np.array([0] + block_indices)
    return tuple((int(start), int(end)) for start, end in zip(block_indices[:-1], block_indices[1:], strict=False))


def variant_offsets_from_metadata(metadata: pl.DataFrame) -> np.ndarray:
    """Return leading-zero cumulative variant offsets from block metadata."""
    n_variants = metadata.get_column("n_variants").to_numpy()
    return np.insert(np.cumsum(n_variants), 0, 0).astype(np.int64)


class JaxParallelOperator(eqx.Module):
    r"""JAX-compatible parallel operator scaffold.

    !!! info
        Numerical products are intentionally unavailable until Phase 2.

    **Arguments:**

    - `blocks`: Block operators composed by this wrapper.
    - `shape`: Matrix shape as `(n_rows, n_cols)`.
    - `backend`: Requested numerical backend.
    - `dtype`: Computation dtype.
    """

    blocks: tuple[Any, ...] = eqx.field(converter=tuple)
    shape: tuple[int, int] = eqx.field(converter=_shape_tuple, static=True)
    backend: Backend = eqx.field(default=Backend.AUTO, converter=Backend, static=True)
    dtype: Any = eqx.field(default=jnp.float32, converter=jnp.dtype, static=True)

    def __check_init__(self) -> None:
        if len(self.shape) != 2:
            raise ValueError("shape must contain exactly two dimensions")
        if self.shape[0] < 0 or self.shape[1] < 0:
            raise ValueError("shape dimensions must be nonnegative")

    def matmat(self, x: Any) -> Any:
        raise NotImplementedError(_KERNELS_MESSAGE)

    def rmatmat(self, x: Any) -> Any:
        raise NotImplementedError(_KERNELS_MESSAGE)
