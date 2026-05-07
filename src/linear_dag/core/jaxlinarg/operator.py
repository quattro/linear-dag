# pattern: Mixed (unavoidable)
# Reason: Backend dispatch is pure, but module construction must normalize the
# requested runtime backend and emit the required user-facing fallback warning.

import warnings

from enum import Enum
from functools import partial
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from .kernels import ffi_cpu
from .kernels.pure_jax import (
    pure_jax_solve_backward_compressed,
    pure_jax_solve_forward_compressed,
)

try:
    from enum import StrEnum
except ImportError:  # pragma: no cover - Python 3.10 compatibility.

    class StrEnum(str, Enum):
        def __str__(self) -> str:
            return self.value


class Backend(StrEnum):
    AUTO = "auto"
    PURE_JAX = "pure_jax"
    FFI_CPU = "ffi_cpu"
    PALLAS_GPU = "pallas_gpu"


def resolve_backend(requested: Backend, *, platform: str | None = None) -> Backend:
    """Resolve a requested backend to a concrete executable backend."""
    requested = Backend(requested)
    platform = (jax.default_backend() if platform is None else platform).lower()

    if requested is Backend.PURE_JAX:
        return Backend.PURE_JAX
    if requested is Backend.PALLAS_GPU:
        return Backend.PALLAS_GPU
    if requested is Backend.FFI_CPU:
        if ffi_cpu.is_ffi_cpu_available():
            return Backend.FFI_CPU
        warnings.warn(
            "FFI_CPU backend is unavailable; falling back to PURE_JAX.",
            UserWarning,
            stacklevel=2,
        )
        return Backend.PURE_JAX
    if requested is Backend.AUTO:
        if platform == "cpu":
            return Backend.FFI_CPU if ffi_cpu.is_ffi_cpu_available() else Backend.PURE_JAX
        if platform in {"gpu", "cuda", "rocm"}:
            return Backend.PALLAS_GPU
        if platform == "tpu":
            return Backend.PURE_JAX
    raise ValueError(f"unknown backend: {requested}")


class JaxLinearARG(eqx.Module):
    r"""JAX-compatible LinearARG operator.

    !!! info
        Single-block numerical products are available on the pure-JAX backend.
        FFI CPU and Pallas GPU backends are reserved for later phases.

    **Arguments:**

    - `indptr`: CSC index pointer array.
    - `indices`: CSC row index array.
    - `data`: CSC edge value array.
    - `src_of_edge`: Source node index for each edge.
    - `variant_indices`: Variant node indices.
    - `flip`: Allele flip flags aligned to `variant_indices`.
    - `sample_indices`: Sample node indices.
    - `nonunique_indices`: Nonunique sample index mapping.
    - `allele_counts`: Cached allele counts aligned to variants, or `-1` for
      missing entries.
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
    allele_counts: Any = eqx.field(converter=jnp.asarray)
    n_variants: int = eqx.field(static=True)
    n_samples: int = eqx.field(static=True)
    n_nonunique_indices: int = eqx.field(default=-1, static=True)
    min_index_to_keep: int = eqx.field(default=0, static=True)
    backend: Backend = eqx.field(default=Backend.AUTO, converter=resolve_backend, static=True)
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
        n_nonunique_indices: int | None = None,
        allele_counts: Any | None = None,
        backend: Backend = Backend.AUTO,
        dtype: Any = jnp.float32,
    ) -> "JaxLinearARG":
        node_count = int(jnp.asarray(indptr).shape[0]) - 1
        if nonunique_indices is None:
            nonunique_indices = jnp.arange(node_count, dtype=jnp.int32)
        nonunique_indices = jnp.asarray(nonunique_indices, dtype=jnp.int32)
        real_n_nonunique_indices = int(jnp.max(nonunique_indices)) + 1 if nonunique_indices.size else 0
        if n_nonunique_indices is None:
            n_nonunique_indices = real_n_nonunique_indices
        else:
            n_nonunique_indices = int(n_nonunique_indices)
            if n_nonunique_indices < real_n_nonunique_indices:
                raise ValueError("n_nonunique_indices cannot be smaller than the maximum nonunique index")
        sample_indices = jnp.asarray(sample_indices, dtype=jnp.int32)
        min_index_to_keep = int(sample_indices[-1]) if sample_indices.size else 0
        return cls(
            indptr=jnp.asarray(indptr, dtype=jnp.int32),
            indices=jnp.asarray(indices, dtype=jnp.int32),
            data=jnp.asarray(data, dtype=dtype),
            src_of_edge=jnp.asarray(src_of_edge, dtype=jnp.int32),
            variant_indices=jnp.asarray(variant_indices, dtype=jnp.int32),
            flip=jnp.asarray(flip, dtype=jnp.bool_),
            sample_indices=sample_indices,
            nonunique_indices=nonunique_indices,
            allele_counts=_canonical_allele_counts(allele_counts, n_variants=int(n_variants)),
            n_variants=int(n_variants),
            n_samples=int(n_samples),
            n_nonunique_indices=n_nonunique_indices,
            min_index_to_keep=min_index_to_keep,
            backend=backend,
            dtype=dtype,
        )

    @classmethod
    def from_lineararg(
        cls,
        linarg: Any,
        *,
        backend: Backend = Backend.AUTO,
        bucket: Any = None,
        dtype: Any = None,
    ) -> "JaxLinearARG":
        from .ingress import from_lineararg

        return from_lineararg(linarg, backend=backend, bucket=bucket, dtype=dtype)

    @classmethod
    def from_hdf5_block(
        cls,
        path: Any,
        block: Any,
        *,
        backend: Backend = Backend.AUTO,
        bucket: Any = None,
        load_metadata: bool = False,
        dtype: Any = None,
    ) -> "JaxLinearARG":
        from .ingress import from_hdf5_block

        return from_hdf5_block(
            path,
            block,
            backend=backend,
            bucket=bucket,
            load_metadata=load_metadata,
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
        indptr = np.asarray(self.indptr)
        if int(indptr[0]) != 0:
            raise ValueError("indptr must start at 0")
        if np.any(np.diff(indptr) < 0):
            raise ValueError("indptr must be monotonic")
        if self.nonunique_indices.shape[0] != self.indptr.shape[0] - 1:
            raise ValueError("nonunique_indices length must match the node count from indptr")
        if self.variant_indices.shape[0] != self.flip.shape[0]:
            raise ValueError("variant_indices and flip must have the same length")
        if self.variant_indices.shape[0] != self.n_variants:
            raise ValueError("variant_indices length must match n_variants")
        if self.allele_counts.ndim != 1:
            raise ValueError("allele_counts must be rank 1")
        if self.allele_counts.shape[0] != self.n_variants:
            raise ValueError("allele_counts length must match n_variants")
        if self.sample_indices.shape[0] != self.n_samples:
            raise ValueError("sample_indices length must match n_samples")
        if self.n_variants < 0:
            raise ValueError("n_variants must be nonnegative")
        if self.n_samples < 0:
            raise ValueError("n_samples must be nonnegative")
        if self.n_nonunique_indices < 0:
            raise ValueError("n_nonunique_indices must be nonnegative")
        if int(self.indptr[-1]) != n_edges:
            raise ValueError("final indptr entry must match the edge count")

        node_count = self.indptr.shape[0] - 1
        for name in (
            "indices",
            "src_of_edge",
            "variant_indices",
            "sample_indices",
            "nonunique_indices",
        ):
            _check_no_negative_index(name, arrays[name])

        if self.src_of_edge.shape[0] and int(jnp.max(self.src_of_edge)) >= node_count:
            raise ValueError("src_of_edge contains an out-of-range node index")
        if self.indices.shape[0] and int(jnp.max(self.indices)) >= node_count:
            raise ValueError("indices contains an out-of-range node index")
        expected_src_of_edge = np.repeat(np.arange(node_count, dtype=np.int32), np.diff(indptr))
        if not np.array_equal(np.asarray(self.src_of_edge), expected_src_of_edge):
            raise ValueError("src_of_edge must match the sources implied by indptr")
        indices = np.asarray(self.indices)
        src_of_edge = np.asarray(self.src_of_edge)
        data = np.asarray(self.data)
        invalid_edge_order = (indices < src_of_edge) | ((indices == src_of_edge) & (data != 0))
        if self.indices.shape[0] and np.any(invalid_edge_order):
            raise ValueError("indices must be greater than src_of_edge")
        if self.variant_indices.shape[0] and int(jnp.max(self.variant_indices)) >= node_count:
            raise ValueError("variant_indices contains an out-of-range node index")
        if self.sample_indices.shape[0] and int(jnp.max(self.sample_indices)) >= node_count:
            raise ValueError("sample_indices contains an out-of-range node index")
        if self.nonunique_indices.shape[0] and int(jnp.max(self.nonunique_indices)) >= self.n_nonunique_indices:
            raise ValueError("nonunique_indices contains an out-of-range compressed index")
        if self.min_index_to_keep < 0 or self.min_index_to_keep > node_count:
            raise ValueError("min_index_to_keep must be within the node range")

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
        b = jnp.zeros((self.n_nonunique_indices, x.shape[1]), dtype=x.dtype)
        variant_nonunique_indices = self.nonunique_indices[self.variant_indices]
        b = b.at[variant_nonunique_indices, :].add(x * flip_sign[:, None])
        solved = _solve(
            self.backend,
            "forward",
            self.indptr,
            self.indices,
            self.data,
            self.src_of_edge,
            self.nonunique_indices,
            self.min_index_to_keep,
            b,
        )
        flip_sum = jnp.sum(x * self.flip.astype(x.dtype)[:, None], axis=0)
        sample_nonunique_indices = self.nonunique_indices[self.sample_indices]
        return solved[sample_nonunique_indices, :] + flip_sum

    def _rmatmat(self, x: Any) -> Any:
        x = jnp.asarray(x, dtype=self.dtype)
        b = jnp.zeros((self.n_nonunique_indices, x.shape[1]), dtype=x.dtype)
        sample_nonunique_indices = self.nonunique_indices[self.sample_indices]
        b = b.at[sample_nonunique_indices, :].set(x)
        solved = _solve(
            self.backend,
            "backward",
            self.indptr,
            self.indices,
            self.data,
            self.src_of_edge,
            self.nonunique_indices,
            self.min_index_to_keep,
            b,
        )
        variant_nonunique_indices = self.nonunique_indices[self.variant_indices]
        values = solved[variant_nonunique_indices, :]
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

    @property
    def T(self) -> JaxLinearARG:
        return self.transpose_view()

    def transpose_view(self) -> JaxLinearARG:
        return self.parent

    def __matmul__(self, x: Any) -> Any:
        return self.matmat(x)


def _check_no_negative_index(name: str, array: Any) -> None:
    if array.shape[0] and int(jnp.min(array)) < 0:
        raise ValueError(f"{name} contains a negative index")


def _canonical_allele_counts(allele_counts: Any | None, *, n_variants: int) -> jax.Array:
    if allele_counts is None:
        return jnp.full((n_variants,), -1, dtype=jnp.int32)
    return jnp.asarray(allele_counts, dtype=jnp.int32)


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
    nonunique_indices: Any,
    min_index_to_keep: int,
    b: Any,
) -> jax.Array:
    return _solve_impl(
        backend,
        direction,
        indptr,
        indices,
        data,
        src_of_edge,
        nonunique_indices,
        min_index_to_keep,
        b,
    )


# custom_vjp disables forward-mode differentiation for this wrapped function;
# reverse-mode gradients are defined by the transpose-direction solve below.
_solve = partial(jax.custom_vjp, nondiff_argnums=(0, 1, 7))(_solve)


def _solve_fwd(
    backend: Backend,
    direction: str,
    indptr: Any,
    indices: Any,
    data: Any,
    src_of_edge: Any,
    nonunique_indices: Any,
    min_index_to_keep: int,
    b: Any,
) -> tuple[jax.Array, tuple[Any, Any, Any, Any, Any]]:
    result = _solve_impl(
        backend,
        direction,
        indptr,
        indices,
        data,
        src_of_edge,
        nonunique_indices,
        min_index_to_keep,
        b,
    )
    return result, (indptr, indices, data, src_of_edge, nonunique_indices)


def _solve_bwd(
    backend: Backend,
    direction: str,
    min_index_to_keep: int,
    residual: tuple[Any, Any, Any, Any, Any],
    grad: Any,
) -> tuple[None, None, None, None, None, jax.Array]:
    indptr, indices, data, src_of_edge, nonunique_indices = residual
    transpose_direction = "backward" if direction == "forward" else "forward"
    grad_b = _solve_impl(
        backend,
        transpose_direction,
        indptr,
        indices,
        data,
        src_of_edge,
        nonunique_indices,
        min_index_to_keep,
        grad,
    )
    return None, None, None, None, None, grad_b


_solve.defvjp(_solve_fwd, _solve_bwd)


def _solve_impl(
    backend: Backend,
    direction: str,
    indptr: Any,
    indices: Any,
    data: Any,
    src_of_edge: Any,
    nonunique_indices: Any,
    min_index_to_keep: int,
    b: Any,
) -> jax.Array:
    if backend is Backend.AUTO:
        backend = resolve_backend(backend)
    if backend is Backend.FFI_CPU:
        if ffi_cpu.is_ffi_cpu_available():
            if direction == "forward":
                return ffi_cpu.ffi_cpu_solve_forward(
                    indptr,
                    indices,
                    data,
                    src_of_edge,
                    nonunique_indices,
                    min_index_to_keep,
                    b,
                )
            if direction == "backward":
                return ffi_cpu.ffi_cpu_solve_backward(
                    indptr,
                    indices,
                    data,
                    src_of_edge,
                    nonunique_indices,
                    min_index_to_keep,
                    b,
                )
            raise ValueError(f"unknown solve direction: {direction}")
        warnings.warn(
            "FFI_CPU backend is unavailable; falling back to PURE_JAX.",
            UserWarning,
            stacklevel=2,
        )
        backend = Backend.PURE_JAX
    if backend is Backend.PURE_JAX and direction == "forward":
        return pure_jax_solve_forward_compressed(
            indptr,
            indices,
            data,
            src_of_edge,
            nonunique_indices,
            b,
            min_index_to_keep=min_index_to_keep,
            n_edges=int(indices.shape[0]),
        )
    if backend is Backend.PURE_JAX and direction == "backward":
        return pure_jax_solve_backward_compressed(
            indptr,
            indices,
            data,
            src_of_edge,
            nonunique_indices,
            b,
            min_index_to_keep=min_index_to_keep,
            n_edges=int(indices.shape[0]),
        )
    if backend is Backend.PALLAS_GPU:
        raise NotImplementedError(f"{backend} backend is implemented in a later phase")
    raise ValueError(f"unknown solve direction: {direction}")
