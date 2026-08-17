# pattern: Imperative Shell

"""Registration and invocation of the optional native CPU FFI kernels."""

from __future__ import annotations

from functools import cache
from importlib import import_module
from typing import Any

import jax
import jax.numpy as jnp

from ..packing import (
    PACKED_FFI_DESCRIPTOR_FIELDS,
    PACKED_FFI_DESCRIPTOR_VERSION as _PACKED_FFI_DESCRIPTOR_VERSION,
)

PACKED_FFI_DESCRIPTOR_VERSION = _PACKED_FFI_DESCRIPTOR_VERSION

FFI_CPU_SOLVE_FORWARD_F32 = "linear_dag_jaxlinarg_solve_forward_f32"
FFI_CPU_SOLVE_BACKWARD_F32 = "linear_dag_jaxlinarg_solve_backward_f32"
FFI_CPU_SOLVE_FORWARD_F64 = "linear_dag_jaxlinarg_solve_forward_f64"
FFI_CPU_SOLVE_BACKWARD_F64 = "linear_dag_jaxlinarg_solve_backward_f64"
FFI_CPU_PACKED_SOLVE_FORWARD_F32 = "linear_dag_jaxlinarg_packed_solve_forward_f32"
FFI_CPU_PACKED_SOLVE_BACKWARD_F32 = "linear_dag_jaxlinarg_packed_solve_backward_f32"
FFI_CPU_PACKED_SOLVE_FORWARD_F64 = "linear_dag_jaxlinarg_packed_solve_forward_f64"
FFI_CPU_PACKED_SOLVE_BACKWARD_F64 = "linear_dag_jaxlinarg_packed_solve_backward_f64"

_last_ffi_cpu_error: Exception | None = None


@cache
def is_ffi_cpu_available() -> bool:
    """Return whether the native CPU FFI handler can be imported and registered."""
    global _last_ffi_cpu_error
    try:
        _load_ffi_cpu_impl()
    except Exception as error:
        _last_ffi_cpu_error = error
        return False
    _last_ffi_cpu_error = None
    return True


def is_ffi_cpu_built() -> bool:
    """Return whether the native CPU FFI handler can be imported."""
    try:
        _import_ffi_cpu_impl()
    except Exception:
        return False
    return True


def last_ffi_cpu_error() -> Exception | None:
    """Return the last native CPU FFI import or registration error."""
    return _last_ffi_cpu_error


def is_ffi_cpu_blas_enabled() -> bool:
    """Return whether the native CPU FFI handler was built with CBLAS support."""
    try:
        return bool(_import_ffi_cpu_impl().blas_enabled())
    except Exception:
        return False


def ffi_cpu_blas_backend() -> str | None:
    """Return the native CPU FFI BLAS backend selected at build time."""
    try:
        return str(_import_ffi_cpu_impl().blas_backend())
    except Exception:
        return None


def is_ffi_cpu_native_tuning_enabled() -> bool:
    """Return whether the native CPU FFI handler was built for the local CPU."""
    try:
        return bool(_import_ffi_cpu_impl().native_tuning_enabled())
    except Exception:
        return False


@cache
def _import_ffi_cpu_impl() -> Any:
    # The extension is generated at build time and is optional in source-only
    # installations, so keep it outside the static import graph.
    return import_module("linear_dag.core.jaxlinarg.kernels._ffi_cpu_impl")


@cache
def _load_ffi_cpu_impl() -> Any:
    _ffi_cpu_impl = _import_ffi_cpu_impl()
    # Registration mutates JAX's process-wide FFI registry, so the cached
    # loader performs it at most once per extension module instance.
    for name, capsule in _ffi_cpu_impl.registrations().items():
        jax.ffi.register_ffi_target(name, capsule, platform="cpu", api_version=1)
    return _ffi_cpu_impl


def ffi_cpu_solve_forward(
    indptr: Any,
    indices: Any,
    data: Any,
    nonunique_indices: Any,
    min_index_to_keep: int,
    b: Any,
) -> jax.Array:
    """Execute the native forward solve over a compressed node buffer.

    **Arguments:**

    - `indptr`: CSC edge offsets for each source node.
    - `indices`: Destination node for each edge.
    - `data`: Edge coefficients aligned to `indices`.
    - `nonunique_indices`: Mapping from graph nodes to reusable buffer rows.
    - `min_index_to_keep`: First graph node whose buffer row must remain live.
    - `b`: Rank-2 compressed-node-by-trait input buffer.

    **Returns:**

    - The propagated compressed buffer with the same shape and dtype as `b`.

    **Raises:**

    - `RuntimeError`: If the native CPU extension cannot be registered.
    - `ValueError`: If `b` is not `float32` or `float64`.
    """
    return _ffi_cpu_solve(
        FFI_CPU_SOLVE_FORWARD_F32,
        FFI_CPU_SOLVE_FORWARD_F64,
        indptr,
        indices,
        data,
        nonunique_indices,
        min_index_to_keep,
        b,
    )


def ffi_cpu_solve_backward(
    indptr: Any,
    indices: Any,
    data: Any,
    nonunique_indices: Any,
    min_index_to_keep: int,
    b: Any,
) -> jax.Array:
    """Execute the native transposed solve over a compressed node buffer.

    **Arguments:**

    - `indptr`: CSC edge offsets for each source node.
    - `indices`: Destination node for each edge.
    - `data`: Edge coefficients aligned to `indices`.
    - `nonunique_indices`: Mapping from graph nodes to reusable buffer rows.
    - `min_index_to_keep`: First graph node whose buffer row must remain live.
    - `b`: Rank-2 compressed-node-by-trait input buffer.

    **Returns:**

    - The propagated compressed buffer with the same shape and dtype as `b`.

    **Raises:**

    - `RuntimeError`: If the native CPU extension cannot be registered.
    - `ValueError`: If `b` is not `float32` or `float64`.
    """
    return _ffi_cpu_solve(
        FFI_CPU_SOLVE_BACKWARD_F32,
        FFI_CPU_SOLVE_BACKWARD_F64,
        indptr,
        indices,
        data,
        nonunique_indices,
        min_index_to_keep,
        b,
    )


def ffi_cpu_packed_solve_forward(
    indptr: Any,
    indices: Any,
    data: Any,
    nonunique_indices: Any,
    descriptors: Any,
    b: Any,
) -> jax.Array:
    """Execute the native forward solve over one packed local graph shard.

    Descriptor offsets are rebased into the device-local flat graph buffers.
    Each valid row describes one block; invalid fixed-capacity rows are inert.

    **Arguments:**

    - `indptr`: Flattened rebased CSC edge offsets.
    - `indices`: Flattened rebased destination-node indices.
    - `data`: Edge coefficients aligned to `indices`.
    - `nonunique_indices`: Rebased graph-node-to-compressed-row mapping.
    - `descriptors`: Versioned integer descriptor rows in
      [`PACKED_FFI_DESCRIPTOR_FIELDS`][] order.
    - `b`: Rank-2 aggregate compressed-node-by-trait work buffer.

    **Returns:**

    - The propagated aggregate work buffer with the same shape and dtype as
      `b`.

    **Raises:**

    - `RuntimeError`: If the native CPU extension cannot be registered.
    - `ValueError`: If the descriptor shape or work-buffer dtype is invalid.
    """
    return _ffi_cpu_packed_solve(
        FFI_CPU_PACKED_SOLVE_FORWARD_F32,
        FFI_CPU_PACKED_SOLVE_FORWARD_F64,
        indptr,
        indices,
        data,
        nonunique_indices,
        descriptors,
        b,
    )


def ffi_cpu_packed_solve_backward(
    indptr: Any,
    indices: Any,
    data: Any,
    nonunique_indices: Any,
    descriptors: Any,
    b: Any,
) -> jax.Array:
    """Execute the native transposed solve over one packed local graph shard.

    Descriptor offsets are rebased into the device-local flat graph buffers.
    Each valid row describes one block; invalid fixed-capacity rows are inert.

    **Arguments:**

    - `indptr`: Flattened rebased CSC edge offsets.
    - `indices`: Flattened rebased destination-node indices.
    - `data`: Edge coefficients aligned to `indices`.
    - `nonunique_indices`: Rebased graph-node-to-compressed-row mapping.
    - `descriptors`: Versioned integer descriptor rows in
      [`PACKED_FFI_DESCRIPTOR_FIELDS`][] order.
    - `b`: Rank-2 aggregate compressed-node-by-trait work buffer.

    **Returns:**

    - The propagated aggregate work buffer with the same shape and dtype as
      `b`.

    **Raises:**

    - `RuntimeError`: If the native CPU extension cannot be registered.
    - `ValueError`: If the descriptor shape or work-buffer dtype is invalid.
    """
    return _ffi_cpu_packed_solve(
        FFI_CPU_PACKED_SOLVE_BACKWARD_F32,
        FFI_CPU_PACKED_SOLVE_BACKWARD_F64,
        indptr,
        indices,
        data,
        nonunique_indices,
        descriptors,
        b,
    )


def _ffi_cpu_solve(
    target_f32: str,
    target_f64: str,
    indptr: Any,
    indices: Any,
    data: Any,
    nonunique_indices: Any,
    min_index_to_keep: int,
    b: Any,
) -> jax.Array:
    if not is_ffi_cpu_available():
        raise RuntimeError("FFI CPU backend is unavailable")
    b = jnp.asarray(b)
    target_name = _solve_target_name(b.dtype, target_f32=target_f32, target_f64=target_f64)
    result_shape = jax.ShapeDtypeStruct(b.shape, b.dtype)
    call = jax.ffi.ffi_call(
        target_name,
        result_shape,
        vmap_method="sequential",
        # The solve updates the state buffer in place. Alias the output to the
        # input state when XLA can legally reuse it; the native handler still
        # handles non-aliased buffers for portability.
        input_output_aliases={4: 0},
    )
    return call(
        indptr,
        indices,
        data,
        nonunique_indices,
        b,
        min_index_to_keep=int(min_index_to_keep),
    )


def _ffi_cpu_packed_solve(
    target_f32: str,
    target_f64: str,
    indptr: Any,
    indices: Any,
    data: Any,
    nonunique_indices: Any,
    descriptors: Any,
    b: Any,
) -> jax.Array:
    if not is_ffi_cpu_available():
        detail = last_ffi_cpu_error()
        suffix = f": {detail}" if detail is not None else ""
        raise RuntimeError(f"packed CPU FFI backend is unavailable{suffix}")
    descriptors = jnp.asarray(descriptors)
    if descriptors.ndim != 2 or descriptors.shape[1] != len(PACKED_FFI_DESCRIPTOR_FIELDS):
        raise ValueError(
            f"packed CPU FFI descriptor buffer must be rank two with {len(PACKED_FFI_DESCRIPTOR_FIELDS)} columns"
        )
    if descriptors.dtype != jnp.int32:
        raise ValueError(f"packed CPU FFI descriptors must have int32 dtype, got {descriptors.dtype}")
    b = jnp.asarray(b)
    target_name = _solve_target_name(b.dtype, target_f32=target_f32, target_f64=target_f64)
    result_shape = jax.ShapeDtypeStruct(b.shape, b.dtype)
    call = jax.ffi.ffi_call(
        target_name,
        result_shape,
        vmap_method="sequential",
        # Only the mutable aggregate state aliases its result. Graph arrays and
        # descriptors remain read-only operands at the native boundary.
        input_output_aliases={5: 0},
    )
    return call(indptr, indices, data, nonunique_indices, descriptors, b)


def _solve_target_name(dtype: Any, *, target_f32: str, target_f64: str) -> str:
    dtype = jnp.dtype(dtype)
    if dtype == jnp.float32:
        return target_f32
    if dtype == jnp.float64:
        return target_f64
    raise ValueError(f"FFI CPU solve supports float32 and float64 buffers, got {dtype}")
