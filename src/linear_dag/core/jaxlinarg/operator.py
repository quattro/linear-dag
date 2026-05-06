# pattern: Functional Core

from enum import Enum
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp

from .kernels.pure_jax import pure_jax_solve_backward, pure_jax_solve_forward

try:
    from enum import StrEnum
except ImportError:  # pragma: no cover - Python 3.10 compatibility.

    class StrEnum(str, Enum):
        def __str__(self) -> str:
            return self.value


_KERNELS_MESSAGE = "JaxLinearARG numerical kernels are implemented in Phase 2"


class Backend(StrEnum):
    AUTO = "auto"
    PURE_JAX = "pure_jax"
    FFI_CPU = "ffi_cpu"
    PALLAS_GPU = "pallas_gpu"


class JaxLinearARG(eqx.Module):
    r"""JAX-compatible LinearARG operator scaffold.

    !!! info
        Numerical products are intentionally unavailable until Phase 2.

    **Arguments:**

    - `indptr`: CSC index pointer array.
    - `indices`: CSC row index array.
    - `data`: CSC edge value array.
    - `src_of_edge`: Source node index for each edge.
    - `variant_indices`: Variant node indices.
    - `flip`: Allele flip flags aligned to `variant_indices`.
    - `sample_indices`: Sample node indices.
    - `nonunique_indices`: Nonunique sample index mapping.
    - `n_variants`: Number of variants in genotype space.
    - `n_samples`: Number of samples in genotype space.
    - `backend`: Requested numerical backend.
    - `dtype`: Computation dtype.
    - `transpose`: Whether this view is transposed.
    """

    indptr: Any = eqx.field(converter=jnp.asarray)
    indices: Any = eqx.field(converter=jnp.asarray)
    data: Any = eqx.field(converter=jnp.asarray)
    src_of_edge: Any = eqx.field(converter=jnp.asarray)
    variant_indices: Any = eqx.field(converter=jnp.asarray)
    flip: Any = eqx.field(converter=jnp.asarray)
    sample_indices: Any = eqx.field(converter=jnp.asarray)
    nonunique_indices: Any = eqx.field(converter=jnp.asarray)
    n_variants: int = eqx.field(static=True)
    n_samples: int = eqx.field(static=True)
    backend: Backend = eqx.field(default=Backend.AUTO, converter=Backend, static=True)
    dtype: Any = eqx.field(default=jnp.float32, converter=jnp.dtype, static=True)
    transpose: bool = eqx.field(default=False, converter=bool, static=True)

    @classmethod
    def from_lineararg_arrays(
        cls,
        *,
        indptr: Any,
        indices: Any,
        data: Any,
        src_of_edge: Any,
        variant_indices: Any,
        flip: Any,
        sample_indices: Any,
        nonunique_indices: Any | None,
        n_variants: int,
        n_samples: int,
        backend: Backend = Backend.AUTO,
        dtype: Any = jnp.float32,
    ) -> "JaxLinearARG":
        node_count = int(jnp.asarray(indptr).shape[0]) - 1
        if nonunique_indices is None:
            nonunique_indices = jnp.arange(node_count, dtype=jnp.int32)
        return cls(
            indptr=jnp.asarray(indptr, dtype=jnp.int32),
            indices=jnp.asarray(indices, dtype=jnp.int32),
            data=jnp.asarray(data, dtype=dtype),
            src_of_edge=jnp.asarray(src_of_edge, dtype=jnp.int32),
            variant_indices=jnp.asarray(variant_indices, dtype=jnp.int32),
            flip=jnp.asarray(flip, dtype=jnp.bool_),
            sample_indices=jnp.asarray(sample_indices, dtype=jnp.int32),
            nonunique_indices=jnp.asarray(nonunique_indices, dtype=jnp.int32),
            n_variants=int(n_variants),
            n_samples=int(n_samples),
            backend=backend,
            dtype=dtype,
        )

    def __check_init__(self) -> None:
        arrays = {
            "indptr": self.indptr,
            "indices": self.indices,
            "data": self.data,
            "src_of_edge": self.src_of_edge,
            "variant_indices": self.variant_indices,
            "flip": self.flip,
            "sample_indices": self.sample_indices,
            "nonunique_indices": self.nonunique_indices,
        }
        for name, array in arrays.items():
            if array.ndim != 1:
                raise ValueError(f"{name} must be rank 1")

        n_edges = self.indices.shape[0]
        if self.data.shape[0] != n_edges:
            raise ValueError("data must have the same length as indices")
        if self.src_of_edge.shape[0] != n_edges:
            raise ValueError("src_of_edge must have the same length as indices")
        if self.indptr.shape[0] == 0:
            raise ValueError("indptr must contain at least one entry")
        if self.nonunique_indices.shape[0] != self.indptr.shape[0] - 1:
            raise ValueError("nonunique_indices length must match the node count from indptr")
        if self.variant_indices.shape[0] != self.flip.shape[0]:
            raise ValueError("variant_indices and flip must have the same length")
        if self.variant_indices.shape[0] != self.n_variants:
            raise ValueError("variant_indices length must match n_variants")
        if self.sample_indices.shape[0] != self.n_samples:
            raise ValueError("sample_indices length must match n_samples")
        if self.n_variants < 0:
            raise ValueError("n_variants must be nonnegative")
        if self.n_samples < 0:
            raise ValueError("n_samples must be nonnegative")
        if int(self.indptr[-1]) != n_edges:
            raise ValueError("final indptr entry must match the edge count")
        if self.src_of_edge.shape[0] and int(jnp.max(self.src_of_edge)) >= self.indptr.shape[0] - 1:
            raise ValueError("src_of_edge contains an out-of-range node index")
        if self.indices.shape[0] and int(jnp.max(self.indices)) >= self.indptr.shape[0] - 1:
            raise ValueError("indices contains an out-of-range node index")

    @property
    def shape(self) -> tuple[int, int]:
        return (self.n_samples, self.n_variants)

    def matmat(self, x: Any) -> Any:
        matrix, was_vector = _as_rank2_matrix(x, expected_rows=self.n_variants, dtype=self.dtype)
        result = self._matmat(matrix)
        return result[:, 0] if was_vector else result

    def rmatmat(self, x: Any) -> Any:
        matrix, was_vector = _as_rank2_matrix(x, expected_rows=self.n_samples, dtype=self.dtype)
        result = self._rmatmat(matrix)
        return result[:, 0] if was_vector else result

    def _matmat(self, x: Any) -> Any:
        x = jnp.asarray(x, dtype=self.dtype)
        flip_sign = jnp.where(self.flip, -1, 1).astype(x.dtype)
        b = jnp.zeros((self.indptr.shape[0] - 1, x.shape[1]), dtype=x.dtype)
        b = b.at[self.variant_indices, :].add(x * flip_sign[:, None])
        solved = _solve(self.backend, "forward", self.indptr, self.indices, self.data, self.src_of_edge, b)
        flip_sum = jnp.sum(x * self.flip.astype(x.dtype)[:, None], axis=0)
        return solved[self.sample_indices, :] + flip_sum

    def _rmatmat(self, x: Any) -> Any:
        x = jnp.asarray(x, dtype=self.dtype)
        b = jnp.zeros((self.indptr.shape[0] - 1, x.shape[1]), dtype=x.dtype)
        b = b.at[self.sample_indices, :].set(x)
        solved = _solve(self.backend, "backward", self.indptr, self.indices, self.data, self.src_of_edge, b)
        values = solved[self.variant_indices, :]
        total = jnp.sum(x, axis=0)
        return jnp.where(self.flip[:, None], total[None, :] - values, values)

    def matvec(self, x: Any) -> Any:
        return self.matmat(x)

    def rmatvec(self, x: Any) -> Any:
        return self.rmatmat(x)

    @property
    def T(self) -> "_TransposeView":
        return self.transpose_view()

    def transpose_view(self) -> "_TransposeView":
        return _TransposeView(self)

    def __matmul__(self, x: Any) -> Any:
        return self.matmat(x)


class _TransposeView(eqx.Module):
    parent: JaxLinearARG

    @property
    def shape(self) -> tuple[int, int]:
        rows, cols = self.parent.shape
        return (cols, rows)

    def matmat(self, x: Any) -> Any:
        return self.parent.rmatmat(x)

    def rmatmat(self, x: Any) -> Any:
        return self.parent.matmat(x)

    def matvec(self, x: Any) -> Any:
        return self.matmat(x)

    def rmatvec(self, x: Any) -> Any:
        return self.rmatmat(x)

    def __matmul__(self, x: Any) -> Any:
        return self.matmat(x)


def _as_rank2_matrix(x: Any, *, expected_rows: int, dtype: Any) -> tuple[jax.Array, bool]:
    array = jnp.asarray(x, dtype=dtype)
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


def _solve(
    backend: Backend,
    direction: str,
    indptr: Any,
    indices: Any,
    data: Any,
    src_of_edge: Any,
    b: Any,
) -> jax.Array:
    if backend is Backend.AUTO:
        backend = Backend.PURE_JAX
    if backend is not Backend.PURE_JAX:
        raise NotImplementedError(f"{backend} backend is implemented in a later phase")
    if direction == "forward":
        return pure_jax_solve_forward(indptr, indices, data, src_of_edge, b, n_edges=int(indices.shape[0]))
    if direction == "backward":
        return pure_jax_solve_backward(indptr, indices, data, src_of_edge, b, n_edges=int(indices.shape[0]))
    raise ValueError(f"unknown solve direction: {direction}")
