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
FFI_CPU_MATMAT_F32 = "linear_dag_jaxlinarg_matmat_f32"
FFI_CPU_MATMAT_F64 = "linear_dag_jaxlinarg_matmat_f64"
FFI_CPU_RMATMAT_F32 = "linear_dag_jaxlinarg_rmatmat_f32"
FFI_CPU_RMATMAT_F64 = "linear_dag_jaxlinarg_rmatmat_f64"
FFI_CPU_TARGET_NAMES = (
    FFI_CPU_SOLVE_FORWARD_F32,
    FFI_CPU_SOLVE_BACKWARD_F32,
    FFI_CPU_SOLVE_FORWARD_F64,
    FFI_CPU_SOLVE_BACKWARD_F64,
    FFI_CPU_MATMAT_F32,
    FFI_CPU_MATMAT_F64,
    FFI_CPU_RMATMAT_F32,
    FFI_CPU_RMATMAT_F64,
)

_last_ffi_cpu_error: Exception | None = None


@cache
def is_ffi_cpu_available() -> bool:
    """Return whether the native CPU FFI handler can be imported and registered."""
    global _last_ffi_cpu_error
    try:
        implementation = _load_ffi_cpu_impl()
        missing = set(FFI_CPU_TARGET_NAMES) - set(implementation.registrations())
        if missing:
            raise RuntimeError(f"native extension is missing targets: {sorted(missing)}")
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
    from linear_dag.core.jaxlinarg.kernels import _ffi_cpu_impl

    return _ffi_cpu_impl


@cache
def _load_ffi_cpu_impl() -> Any:
    _ffi_cpu_impl = _import_ffi_cpu_impl()
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
    """Execute the native CPU FFI forward compressed solve."""
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
    """Execute the native CPU FFI backward compressed solve."""
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


def ffi_cpu_matmat(
    indptr: Any,
    indices: Any,
    data: Any,
    nonunique_indices: Any,
    variant_indices: Any,
    sample_indices: Any,
    flip: Any,
    min_index_to_keep: int,
    n_nonunique_indices: int,
    values: Any,
) -> jax.Array:
    """Execute one complete exact native CPU forward product."""
    return _ffi_cpu_product(
        FFI_CPU_MATMAT_F32,
        FFI_CPU_MATMAT_F64,
        indptr,
        indices,
        data,
        nonunique_indices,
        variant_indices,
        sample_indices,
        flip,
        min_index_to_keep,
        n_nonunique_indices,
        values,
        result_rows=jnp.asarray(sample_indices).shape[0],
    )


def ffi_cpu_rmatmat(
    indptr: Any,
    indices: Any,
    data: Any,
    nonunique_indices: Any,
    variant_indices: Any,
    sample_indices: Any,
    flip: Any,
    min_index_to_keep: int,
    n_nonunique_indices: int,
    values: Any,
) -> jax.Array:
    """Execute one complete exact native CPU backward product."""
    return _ffi_cpu_product(
        FFI_CPU_RMATMAT_F32,
        FFI_CPU_RMATMAT_F64,
        indptr,
        indices,
        data,
        nonunique_indices,
        variant_indices,
        sample_indices,
        flip,
        min_index_to_keep,
        n_nonunique_indices,
        values,
        result_rows=jnp.asarray(variant_indices).shape[0],
    )


def _ffi_cpu_product(
    target_f32: str,
    target_f64: str,
    indptr: Any,
    indices: Any,
    data: Any,
    nonunique_indices: Any,
    variant_indices: Any,
    sample_indices: Any,
    flip: Any,
    min_index_to_keep: int,
    n_nonunique_indices: int,
    values: Any,
    *,
    result_rows: int,
) -> jax.Array:
    if not is_ffi_cpu_available():
        raise RuntimeError("FFI CPU backend is unavailable")
    values = jnp.asarray(values)
    target_name = _solve_target_name(values.dtype, target_f32=target_f32, target_f64=target_f64)
    result_shape = jax.ShapeDtypeStruct((int(result_rows), values.shape[1]), values.dtype)
    workspace_shape = jax.ShapeDtypeStruct((int(n_nonunique_indices) + 1, values.shape[1]), values.dtype)
    call = jax.ffi.ffi_call(
        target_name,
        (result_shape, workspace_shape),
        vmap_method="sequential",
    )
    result, _workspace = call(
        indptr,
        indices,
        data,
        nonunique_indices,
        variant_indices,
        sample_indices,
        flip,
        values,
        min_index_to_keep=int(min_index_to_keep),
    )
    return result


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


def _solve_target_name(dtype: Any, *, target_f32: str, target_f64: str) -> str:
    dtype = jnp.dtype(dtype)
    if dtype == jnp.float32:
        return target_f32
    if dtype == jnp.float64:
        return target_f64
    raise ValueError(f"FFI CPU solve supports float32 and float64 buffers, got {dtype}")
