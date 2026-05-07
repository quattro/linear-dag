# pattern: Imperative Shell

from __future__ import annotations

from functools import cache
from typing import Any

import jax
import jax.numpy as jnp

FFI_CPU_SOLVE_FORWARD_F32 = "linear_dag_jaxlinarg_solve_forward_f32"
FFI_CPU_SOLVE_BACKWARD_F32 = "linear_dag_jaxlinarg_solve_backward_f32"
FFI_CPU_SOLVE_FORWARD_F64 = "linear_dag_jaxlinarg_solve_forward_f64"
FFI_CPU_SOLVE_BACKWARD_F64 = "linear_dag_jaxlinarg_solve_backward_f64"


@cache
def is_ffi_cpu_available() -> bool:
    """Return whether the native CPU FFI handler can be imported."""
    try:
        from linear_dag.core.jaxlinarg.kernels import _ffi_cpu_impl  # noqa: F401
    except ImportError:
        return False
    return True


def ffi_cpu_solve_forward(
    indptr: Any,
    indices: Any,
    data: Any,
    src_of_edge: Any,
    nonunique_indices: Any,
    min_index_to_keep: int,
    b: Any,
) -> jax.Array:
    """Execute the native CPU FFI forward compressed solve."""
    return _ffi_cpu_solve(
        FFI_CPU_SOLVE_FORWARD_F32,
        FFI_CPU_SOLVE_FORWARD_F64,
        indptr,
        indices,
        data,
        src_of_edge,
        nonunique_indices,
        min_index_to_keep,
        b,
    )


def ffi_cpu_solve_backward(
    indptr: Any,
    indices: Any,
    data: Any,
    src_of_edge: Any,
    nonunique_indices: Any,
    min_index_to_keep: int,
    b: Any,
) -> jax.Array:
    """Execute the native CPU FFI backward compressed solve."""
    return _ffi_cpu_solve(
        FFI_CPU_SOLVE_BACKWARD_F32,
        FFI_CPU_SOLVE_BACKWARD_F64,
        indptr,
        indices,
        data,
        src_of_edge,
        nonunique_indices,
        min_index_to_keep,
        b,
    )


def _ffi_cpu_solve(
    target_f32: str,
    target_f64: str,
    indptr: Any,
    indices: Any,
    data: Any,
    src_of_edge: Any,
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
    )
    return call(
        indptr,
        indices,
        data,
        src_of_edge,
        nonunique_indices,
        b,
        min_index_to_keep=int(min_index_to_keep),
    )


def _solve_target_name(dtype: Any, *, target_f32: str, target_f64: str) -> str:
    dtype = jnp.dtype(dtype)
    if dtype == jnp.float32:
        return target_f32
    if dtype == jnp.float64:
        return target_f64
    raise ValueError(f"FFI CPU solve supports float32 and float64 buffers, got {dtype}")
