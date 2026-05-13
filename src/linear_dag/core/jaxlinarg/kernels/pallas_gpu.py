# pattern: Functional Core

from __future__ import annotations

import warnings

from typing import Any, Literal, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .pure_jax import pure_jax_solve_backward_compressed, pure_jax_solve_forward_compressed

try:
    import jax.experimental.pallas as pl

    from jax.experimental.pallas import mosaic_gpu
except Exception:  # pragma: no cover - depends on the installed JAX build.
    pl = None
    mosaic_gpu = None


class LevelSchedule(NamedTuple):
    """Edge ordering grouped into dependency-safe graph levels."""

    edge_order: np.ndarray
    level_offsets: np.ndarray


class PallasGpuKernelSupport(NamedTuple):
    """Whether a Mosaic GPU kernel supports a shape."""

    supported: bool
    reason: str


class PallasGpuResourceEstimate(NamedTuple):
    """Approximate per-program/block resources for a Pallas GPU kernel."""

    kernel_kind: str
    n_edges: int
    n_rows: int
    n_cols: int
    dtype: np.dtype
    index_dtype: np.dtype
    estimated_smem_bytes: int
    estimated_work_items: int
    reason: str


_PALLAS_GPU_FALLBACK_COUNT = 0
_PALLAS_GPU_FALLBACK_WARNED = False
_WARP_GROUP_TRANSFER_BYTES = 128


def is_pallas_import_available() -> bool:
    """Return whether the Pallas module imported successfully."""
    return pl is not None


def is_pallas_gpu_available() -> bool:
    """Return whether the Pallas GPU backend can be used by this process."""
    return jax.default_backend() in {"gpu", "cuda", "rocm"} and is_pallas_import_available()


def pallas_gpu_fallback_count() -> int:
    """Return how many Pallas GPU solves fell back to pure JAX in this process."""
    return _PALLAS_GPU_FALLBACK_COUNT


def reset_pallas_gpu_fallback_count(*, clear_warnings: bool = True) -> None:
    """Reset Pallas GPU fallback diagnostics for tests and benchmarks."""
    global _PALLAS_GPU_FALLBACK_COUNT, _PALLAS_GPU_FALLBACK_WARNED
    _PALLAS_GPU_FALLBACK_COUNT = 0
    if clear_warnings:
        _PALLAS_GPU_FALLBACK_WARNED = False


def check_pallas_gpu_kernel_support(
    *,
    indptr: Any,
    indices: Any,
    data: Any,
    src_of_edge: Any,
    nonunique_indices: Any,
    b: Any,
    kernel_kind: Literal["serial", "scheduled"] = "serial",
    max_shared_memory_bytes: int | None = None,
) -> PallasGpuKernelSupport:
    """Return whether the current Mosaic GPU kernels support this static shape."""
    common_support = _check_common_pallas_inputs(
        indptr=indptr,
        indices=indices,
        data=data,
        src_of_edge=src_of_edge,
        nonunique_indices=nonunique_indices,
        b=b,
        kernel_kind=kernel_kind,
    )
    if not common_support.supported:
        return common_support

    refs = _mosaic_visible_refs(
        indptr=indptr,
        indices=indices,
        data=data,
        src_of_edge=src_of_edge,
        nonunique_indices=nonunique_indices,
        b=b,
    )
    copy_support = _check_mosaic_gpu_copy_constraints(refs)
    if not copy_support.supported:
        return copy_support

    if kernel_kind == "serial":
        logical_estimate = _estimate_serial_kernel_resources(
            indptr=indptr,
            indices=indices,
            data=data,
            src_of_edge=src_of_edge,
            nonunique_indices=nonunique_indices,
            b=b,
        )
    elif kernel_kind == "scheduled":
        logical_estimate = _estimate_scheduled_kernel_resources(
            indptr=indptr,
            indices=indices,
            data=data,
            src_of_edge=src_of_edge,
            nonunique_indices=nonunique_indices,
            b=b,
        )
    else:
        raise ValueError(f"unknown Pallas GPU kernel kind: {kernel_kind}")

    lowering_smem_bytes = _estimate_mosaic_lowering_smem_bytes(
        indptr=indptr,
        indices=indices,
        data=data,
        src_of_edge=src_of_edge,
        nonunique_indices=nonunique_indices,
        b=b,
        kernel_kind=kernel_kind,
    )
    effective_smem_bytes = max(logical_estimate.estimated_smem_bytes, lowering_smem_bytes)
    max_shared_memory_bytes = _max_shared_memory_bytes() if max_shared_memory_bytes is None else max_shared_memory_bytes
    if effective_smem_bytes > max_shared_memory_bytes:
        return PallasGpuKernelSupport(
            False,
            f"{kernel_kind} kernel estimated Mosaic lowering shared memory {effective_smem_bytes} bytes exceeds "
            f"available {max_shared_memory_bytes} bytes",
        )
    return PallasGpuKernelSupport(True, "")


def compute_level_schedule(indptr: Any, indices: Any) -> LevelSchedule:
    """Compute a host-side edge schedule grouped by source node wavefront."""
    indptr = np.asarray(indptr, dtype=np.int32)
    indices = np.asarray(indices, dtype=np.int32)
    if indptr.ndim != 1:
        raise ValueError("indptr must be rank 1")
    if indices.ndim != 1:
        raise ValueError("indices must be rank 1")
    if indptr.shape[0] == 0:
        raise ValueError("indptr must contain at least one entry")
    if int(indptr[0]) != 0:
        raise ValueError("indptr must start at 0")
    if np.any(np.diff(indptr) < 0):
        raise ValueError("indptr must be monotonic")
    n_edges = int(indices.shape[0])
    if int(indptr[-1]) != n_edges:
        raise ValueError("final indptr entry must match the edge count")

    node_count = indptr.shape[0] - 1
    if indices.shape[0] and (np.any(indices < 0) or np.any(indices >= node_count)):
        raise ValueError("indices contains an out-of-range node index")

    # Edges from the same level read sources whose incoming updates have already
    # completed. Their writes may collide, so GPU execution still uses atomics.
    node_levels = np.zeros(node_count, dtype=np.int32)
    edge_levels = np.zeros(n_edges, dtype=np.int32)
    for src in range(node_count):
        start = int(indptr[src])
        stop = int(indptr[src + 1])
        src_level = int(node_levels[src])
        edge_levels[start:stop] = src_level
        for dst in indices[start:stop]:
            node_levels[int(dst)] = max(int(node_levels[int(dst)]), src_level + 1)

    edge_order = np.argsort(edge_levels, kind="stable").astype(np.int32)
    if n_edges == 0:
        return LevelSchedule(edge_order=edge_order, level_offsets=np.asarray([0], dtype=np.int32))

    level_counts = np.bincount(edge_levels, minlength=int(edge_levels.max()) + 1).astype(np.int32)
    level_offsets = np.concatenate([np.asarray([0], dtype=np.int32), np.cumsum(level_counts, dtype=np.int32)])
    return LevelSchedule(edge_order=edge_order, level_offsets=level_offsets)


def pallas_gpu_solve_forward(
    indptr: Any,
    indices: Any,
    data: Any,
    src_of_edge: Any,
    nonunique_indices: Any,
    min_index_to_keep: int,
    b: Any,
) -> jax.Array:
    """Solve a compressed LinearARG node buffer with a serial Pallas edge scan."""
    _require_pallas_gpu()
    support = check_pallas_gpu_kernel_support(
        indptr=indptr,
        indices=indices,
        data=data,
        src_of_edge=src_of_edge,
        nonunique_indices=nonunique_indices,
        b=b,
        kernel_kind="serial",
    )
    if not support.supported:
        return _fallback_forward(
            support.reason,
            indptr,
            indices,
            data,
            src_of_edge,
            nonunique_indices,
            min_index_to_keep,
            b,
        )
    n_edges = int(indices.shape[0])
    n_rows, n_cols = b.shape

    def kernel(
        indptr_ref: Any,
        indices_ref: Any,
        data_ref: Any,
        src_of_edge_ref: Any,
        nonunique_indices_ref: Any,
        b_ref: Any,
        out_ref: Any,
    ) -> None:
        _copy_buffer(b_ref, out_ref, n_rows=n_rows, n_cols=n_cols)

        @pl.loop(0, n_edges)
        def edge_loop(edge_index: Any) -> None:
            src = src_of_edge_ref[edge_index]
            dst = indices_ref[edge_index]
            src_row = nonunique_indices_ref[src]
            dst_row = nonunique_indices_ref[dst]
            weight = data_ref[edge_index]

            @pl.loop(0, n_cols)
            def col_loop(col: Any) -> None:
                out_ref[dst_row, col] = out_ref[dst_row, col] + out_ref[src_row, col] * weight

            should_zero = (edge_index == indptr_ref[src + 1] - 1) & (src < min_index_to_keep)

            @pl.when(should_zero)
            def zero_source() -> None:
                _zero_row(out_ref, src_row, n_cols=n_cols)

    return _call_kernel(
        kernel,
        indptr,
        indices,
        data,
        src_of_edge,
        nonunique_indices,
        b,
        name="linear_dag_jaxlinarg_pallas_gpu_solve_forward",
    )


def pallas_gpu_solve_forward_level_scheduled(
    indptr: Any,
    indices: Any,
    data: Any,
    src_of_edge: Any,
    nonunique_indices: Any,
    min_index_to_keep: int,
    schedule: LevelSchedule,
    b: Any,
    *,
    interpret: bool = False,
) -> jax.Array:
    """Solve by applying scheduled edge wavefronts with per-element atomics."""
    if not interpret:
        _require_pallas_gpu()
    schedule = _check_level_schedule(schedule, n_edges=int(indices.shape[0]))
    if not interpret:
        support = check_pallas_gpu_kernel_support(
            indptr=indptr,
            indices=indices,
            data=data,
            src_of_edge=src_of_edge,
            nonunique_indices=nonunique_indices,
            b=b,
            kernel_kind="scheduled",
        )
        if not support.supported:
            return _fallback_forward(
                support.reason,
                indptr,
                indices,
                data,
                src_of_edge,
                nonunique_indices,
                min_index_to_keep,
                b,
            )
    return _scheduled_solve_forward(
        indptr,
        indices,
        data,
        src_of_edge,
        nonunique_indices,
        min_index_to_keep,
        schedule,
        b,
        interpret=interpret,
    )


def pallas_gpu_solve_backward(
    indptr: Any,
    indices: Any,
    data: Any,
    src_of_edge: Any,
    nonunique_indices: Any,
    min_index_to_keep: int,
    b: Any,
) -> jax.Array:
    """Solve a compressed transposed LinearARG buffer with a serial Pallas edge scan."""
    _require_pallas_gpu()
    support = check_pallas_gpu_kernel_support(
        indptr=indptr,
        indices=indices,
        data=data,
        src_of_edge=src_of_edge,
        nonunique_indices=nonunique_indices,
        b=b,
        kernel_kind="serial",
    )
    if not support.supported:
        return _fallback_backward(
            support.reason,
            indptr,
            indices,
            data,
            src_of_edge,
            nonunique_indices,
            min_index_to_keep,
            b,
        )
    n_edges = int(indices.shape[0])
    n_rows, n_cols = b.shape

    def kernel(
        indptr_ref: Any,
        indices_ref: Any,
        data_ref: Any,
        src_of_edge_ref: Any,
        nonunique_indices_ref: Any,
        b_ref: Any,
        out_ref: Any,
    ) -> None:
        _copy_buffer(b_ref, out_ref, n_rows=n_rows, n_cols=n_cols)

        @pl.loop(0, n_edges)
        def edge_loop(scan_index: Any) -> None:
            edge_index = n_edges - scan_index - 1
            src = src_of_edge_ref[edge_index]
            dst = indices_ref[edge_index]
            src_row = nonunique_indices_ref[src]
            dst_row = nonunique_indices_ref[dst]
            weight = data_ref[edge_index]
            should_zero = (edge_index == indptr_ref[src + 1] - 1) & (src < min_index_to_keep)

            @pl.when(should_zero)
            def zero_source() -> None:
                _zero_row(out_ref, src_row, n_cols=n_cols)

            @pl.loop(0, n_cols)
            def col_loop(col: Any) -> None:
                out_ref[src_row, col] = out_ref[src_row, col] + out_ref[dst_row, col] * weight

    return _call_kernel(
        kernel,
        indptr,
        indices,
        data,
        src_of_edge,
        nonunique_indices,
        b,
        name="linear_dag_jaxlinarg_pallas_gpu_solve_backward",
    )


def pallas_gpu_solve_backward_level_scheduled(
    indptr: Any,
    indices: Any,
    data: Any,
    src_of_edge: Any,
    nonunique_indices: Any,
    min_index_to_keep: int,
    schedule: LevelSchedule,
    b: Any,
    *,
    interpret: bool = False,
) -> jax.Array:
    """Solve transpose by applying scheduled edge wavefronts in reverse."""
    if not interpret:
        _require_pallas_gpu()
    schedule = _check_level_schedule(schedule, n_edges=int(indices.shape[0]))
    if not interpret:
        support = check_pallas_gpu_kernel_support(
            indptr=indptr,
            indices=indices,
            data=data,
            src_of_edge=src_of_edge,
            nonunique_indices=nonunique_indices,
            b=b,
            kernel_kind="scheduled",
        )
        if not support.supported:
            return _fallback_backward(
                support.reason,
                indptr,
                indices,
                data,
                src_of_edge,
                nonunique_indices,
                min_index_to_keep,
                b,
            )
    return _scheduled_solve_backward(
        indptr,
        indices,
        data,
        src_of_edge,
        nonunique_indices,
        min_index_to_keep,
        schedule,
        b,
        interpret=interpret,
    )


def _require_pallas_gpu() -> None:
    if not is_pallas_gpu_available():
        raise RuntimeError(
            "Pallas GPU backend is unavailable; expected jax.default_backend() == 'gpu' "
            "and importable jax.experimental.pallas."
        )


def _fallback_forward(
    reason: str,
    indptr: Any,
    indices: Any,
    data: Any,
    src_of_edge: Any,
    nonunique_indices: Any,
    min_index_to_keep: int,
    b: Any,
) -> jax.Array:
    _record_pallas_gpu_fallback(reason)
    return pure_jax_solve_forward_compressed(
        indptr,
        indices,
        data,
        src_of_edge,
        nonunique_indices,
        min_index_to_keep,
        b,
    )


def _fallback_backward(
    reason: str,
    indptr: Any,
    indices: Any,
    data: Any,
    src_of_edge: Any,
    nonunique_indices: Any,
    min_index_to_keep: int,
    b: Any,
) -> jax.Array:
    _record_pallas_gpu_fallback(reason)
    return pure_jax_solve_backward_compressed(
        indptr,
        indices,
        data,
        src_of_edge,
        nonunique_indices,
        min_index_to_keep,
        b,
    )


def _record_pallas_gpu_fallback(reason: str) -> None:
    global _PALLAS_GPU_FALLBACK_COUNT, _PALLAS_GPU_FALLBACK_WARNED
    _PALLAS_GPU_FALLBACK_COUNT += 1
    if _PALLAS_GPU_FALLBACK_WARNED:
        return
    _PALLAS_GPU_FALLBACK_WARNED = True
    warnings.warn(
        f"Pallas GPU kernel unsupported for this shape/device ({reason}); falling back to pure JAX.",
        UserWarning,
        stacklevel=3,
    )


def _check_common_pallas_inputs(
    *,
    indptr: Any,
    indices: Any,
    data: Any,
    src_of_edge: Any,
    nonunique_indices: Any,
    b: Any,
    kernel_kind: str,
) -> PallasGpuKernelSupport:
    if kernel_kind not in {"serial", "scheduled"}:
        raise ValueError(f"unknown Pallas GPU kernel kind: {kernel_kind}")

    arrays = {
        "indptr": indptr,
        "indices": indices,
        "data": data,
        "src_of_edge": src_of_edge,
        "nonunique_indices": nonunique_indices,
    }
    for name, value in arrays.items():
        if value.ndim != 1:
            return PallasGpuKernelSupport(False, f"{name} must be rank 1")
    if b.ndim != 2:
        return PallasGpuKernelSupport(False, "b must be rank 2")

    n_edges = int(indices.shape[0])
    if data.shape[0] != n_edges:
        return PallasGpuKernelSupport(False, "data must have the same length as indices")
    if src_of_edge.shape[0] != n_edges:
        return PallasGpuKernelSupport(False, "src_of_edge must have the same length as indices")
    if indptr.shape[0] == 0:
        return PallasGpuKernelSupport(False, "indptr must contain at least one entry")

    index_dtype = np.dtype(indices.dtype)
    for name in ("indptr", "indices", "src_of_edge", "nonunique_indices"):
        if np.dtype(arrays[name].dtype) != index_dtype:
            return PallasGpuKernelSupport(False, "index arrays must have matching dtypes")
    if index_dtype != np.dtype(np.int32):
        return PallasGpuKernelSupport(False, "Pallas GPU kernels currently require int32 indices")

    dtype = np.dtype(data.dtype)
    if dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        return PallasGpuKernelSupport(False, "Pallas GPU kernels currently require float32 or float64 data")
    if np.dtype(b.dtype) != dtype:
        return PallasGpuKernelSupport(False, "b dtype must match data dtype")

    return PallasGpuKernelSupport(True, "")


def _estimate_serial_kernel_resources(
    *,
    indptr: Any,
    indices: Any,
    data: Any,
    src_of_edge: Any,
    nonunique_indices: Any,
    b: Any,
) -> PallasGpuResourceEstimate:
    del indptr, src_of_edge, nonunique_indices
    n_edges = int(indices.shape[0])
    n_rows, n_cols = (int(b.shape[0]), int(b.shape[1]))
    dtype = np.dtype(data.dtype)
    index_dtype = np.dtype(indices.dtype)

    # Pallas refs address full input/output arrays in global memory. This
    # estimate intentionally counts only per-program working state used by the
    # serial grid=(1,) kernel: loop counters, scalar indices, the edge weight,
    # and one scalar value per active column operation. It does not count the
    # total bytes of graph arrays or the node buffer.
    scalar_state_bytes = 8 * index_dtype.itemsize + 2 * dtype.itemsize
    column_loop_bytes = max(1, n_cols) * dtype.itemsize
    estimated_smem_bytes = _align_bytes(scalar_state_bytes + column_loop_bytes, 128)
    return PallasGpuResourceEstimate(
        kernel_kind="serial",
        n_edges=n_edges,
        n_rows=n_rows,
        n_cols=n_cols,
        dtype=dtype,
        index_dtype=index_dtype,
        estimated_smem_bytes=estimated_smem_bytes,
        estimated_work_items=max(1, n_edges) * max(1, n_cols),
        reason="serial kernel uses scalar loop state; full arrays remain in global memory",
    )


def _estimate_scheduled_kernel_resources(
    *,
    indptr: Any,
    indices: Any,
    data: Any,
    src_of_edge: Any,
    nonunique_indices: Any,
    b: Any,
) -> PallasGpuResourceEstimate:
    del indptr, src_of_edge, nonunique_indices
    n_edges = int(indices.shape[0])
    n_rows, n_cols = (int(b.shape[0]), int(b.shape[1]))
    dtype = np.dtype(data.dtype)
    index_dtype = np.dtype(indices.dtype)

    # Scheduled kernels launch one program for one (edge, col) pair. Their
    # per-program state is scalar: edge position, column, source/destination
    # rows, one weight, and one increment. The total graph and state arrays are
    # global-memory refs and should not be charged as per-program shared memory.
    scalar_state_bytes = 7 * index_dtype.itemsize + 2 * dtype.itemsize
    estimated_smem_bytes = _align_bytes(scalar_state_bytes, 128)
    return PallasGpuResourceEstimate(
        kernel_kind="scheduled",
        n_edges=n_edges,
        n_rows=n_rows,
        n_cols=n_cols,
        dtype=dtype,
        index_dtype=index_dtype,
        estimated_smem_bytes=estimated_smem_bytes,
        estimated_work_items=max(1, n_edges) * max(1, n_cols),
        reason="scheduled kernel work item is one edge-column scalar update",
    )


def _align_bytes(value: int, alignment: int) -> int:
    return ((int(value) + alignment - 1) // alignment) * alignment


def _mosaic_visible_refs(
    *,
    indptr: Any,
    indices: Any,
    data: Any,
    src_of_edge: Any,
    nonunique_indices: Any,
    b: Any,
) -> tuple[tuple[str, Any], ...]:
    return (
        ("indptr", indptr),
        ("indices", indices),
        ("data", data),
        ("src_of_edge", src_of_edge),
        ("nonunique_indices", nonunique_indices),
        ("b", b),
        ("out", b),
    )


def _check_mosaic_gpu_copy_constraints(refs: tuple[tuple[str, Any], ...]) -> PallasGpuKernelSupport:
    for name, value in refs:
        nbytes = _array_nbytes(value)
        if nbytes % _WARP_GROUP_TRANSFER_BYTES != 0:
            return PallasGpuKernelSupport(
                False,
                f"{name} transfer is {nbytes} bytes; Mosaic GPU requires 128-byte aligned transfers",
            )
    return PallasGpuKernelSupport(True, "")


def _estimate_mosaic_lowering_smem_bytes(
    *,
    indptr: Any,
    indices: Any,
    data: Any,
    src_of_edge: Any,
    nonunique_indices: Any,
    b: Any,
    kernel_kind: str,
) -> int:
    refs = _mosaic_visible_refs(
        indptr=indptr,
        indices=indices,
        data=data,
        src_of_edge=src_of_edge,
        nonunique_indices=nonunique_indices,
        b=b,
    )
    if kernel_kind not in {"serial", "scheduled"}:
        raise ValueError(f"unknown Pallas GPU kernel kind: {kernel_kind}")
    # Pallas refs point at global memory, so full graph/state arrays are not the
    # logical per-program scratch used by the algorithm. However, Mosaic GPU
    # lowering for both current serial and scheduled kernels has been observed
    # to reserve shared memory proportional to the whole visible ref set. Keep
    # this as a backend-lowering guard so large blocks fall back before
    # pallas_call raises `Mosaic GPU kernel exceeds available shared memory`.
    return _align_bytes(sum(_array_nbytes(value) for _name, value in refs), _WARP_GROUP_TRANSFER_BYTES)


def _array_nbytes(value: Any) -> int:
    shape = tuple(int(dim) for dim in value.shape)
    return int(np.prod(shape, dtype=np.int64)) * np.dtype(value.dtype).itemsize


def _max_shared_memory_bytes() -> int:
    try:
        device = jax.local_devices()[0]
    except (RuntimeError, IndexError):
        return 48 * 1024
    return int(getattr(device, "shared_memory_per_block_optin", 48 * 1024))


def _call_kernel(
    kernel: Any,
    indptr: Any,
    indices: Any,
    data: Any,
    src_of_edge: Any,
    nonunique_indices: Any,
    b: Any,
    *,
    name: str,
) -> jax.Array:
    if pl is None:  # pragma: no cover - guarded by _require_pallas_gpu.
        raise RuntimeError("Pallas is unavailable.")
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(b.shape, b.dtype),
        grid=(1,),
        name=name,
    )(indptr, indices, data, src_of_edge, nonunique_indices, b)


def _check_level_schedule(schedule: LevelSchedule, *, n_edges: int) -> LevelSchedule:
    edge_order = np.asarray(schedule.edge_order, dtype=np.int32)
    level_offsets = np.asarray(schedule.level_offsets, dtype=np.int32)
    if edge_order.ndim != 1:
        raise ValueError("level schedule edge_order must be rank 1")
    if level_offsets.ndim != 1:
        raise ValueError("level schedule level_offsets must be rank 1")
    if int(edge_order.shape[0]) != n_edges:
        raise ValueError("level schedule edge_order length must match the edge count")
    if level_offsets.shape[0] == 0:
        raise ValueError("level schedule level_offsets must contain at least one entry")
    if int(level_offsets[0]) != 0 or int(level_offsets[-1]) != n_edges:
        raise ValueError("level schedule offsets must span the edge count")
    if np.any(np.diff(level_offsets) < 0):
        raise ValueError("level schedule offsets must be monotonic")
    if edge_order.size and not np.array_equal(np.sort(edge_order), np.arange(n_edges, dtype=np.int32)):
        raise ValueError("level schedule edge_order must be a permutation of edge indices")
    return LevelSchedule(edge_order=edge_order, level_offsets=level_offsets)


def _scheduled_solve_forward(
    indptr: Any,
    indices: Any,
    data: Any,
    src_of_edge: Any,
    nonunique_indices: Any,
    min_index_to_keep: int,
    schedule: LevelSchedule,
    b: Any,
    *,
    interpret: bool,
) -> jax.Array:
    n_cols = b.shape[1]
    edge_order = jnp.asarray(schedule.edge_order, dtype=jnp.int32)
    level_offsets = tuple(int(value) for value in schedule.level_offsets)
    state = b

    for level_start, level_stop in zip(level_offsets[:-1], level_offsets[1:], strict=True):
        edge_count = level_stop - level_start
        if edge_count == 0:
            continue
        # Forward evaluation reads each source before that source is retired.
        # Keep update and zeroing in separate kernels so a zero cannot race a
        # same-level read from another edge.
        state = _call_scheduled_update_kernel(
            _forward_level_update_kernel(
                level_start=level_start,
                interpret=interpret,
            ),
            state,
            indptr,
            indices,
            data,
            src_of_edge,
            nonunique_indices,
            edge_order,
            grid=(edge_count, n_cols),
            name="linear_dag_jaxlinarg_pallas_gpu_scheduled_forward_update",
            interpret=interpret,
        )
        state = _call_scheduled_update_kernel(
            _forward_level_zero_kernel(
                level_start=level_start,
                min_index_to_keep=min_index_to_keep,
            ),
            state,
            indptr,
            indices,
            data,
            src_of_edge,
            nonunique_indices,
            edge_order,
            grid=(edge_count, n_cols),
            name="linear_dag_jaxlinarg_pallas_gpu_scheduled_forward_zero",
            interpret=interpret,
        )
    return state


def _scheduled_solve_backward(
    indptr: Any,
    indices: Any,
    data: Any,
    src_of_edge: Any,
    nonunique_indices: Any,
    min_index_to_keep: int,
    schedule: LevelSchedule,
    b: Any,
    *,
    interpret: bool,
) -> jax.Array:
    n_cols = b.shape[1]
    edge_order = jnp.asarray(schedule.edge_order, dtype=jnp.int32)
    level_offsets = tuple(int(value) for value in schedule.level_offsets)
    state = b

    for level_start, level_stop in reversed(tuple(zip(level_offsets[:-1], level_offsets[1:], strict=True))):
        edge_count = level_stop - level_start
        if edge_count == 0:
            continue
        # Backward evaluation mirrors the serial reverse scan: a source row is
        # cleared before receiving transpose contributions from its descendants.
        state = _call_scheduled_update_kernel(
            _backward_level_zero_kernel(
                level_start=level_start,
                min_index_to_keep=min_index_to_keep,
            ),
            state,
            indptr,
            indices,
            data,
            src_of_edge,
            nonunique_indices,
            edge_order,
            grid=(edge_count, n_cols),
            name="linear_dag_jaxlinarg_pallas_gpu_scheduled_backward_zero",
            interpret=interpret,
        )
        state = _call_scheduled_update_kernel(
            _backward_level_update_kernel(
                level_start=level_start,
                interpret=interpret,
            ),
            state,
            indptr,
            indices,
            data,
            src_of_edge,
            nonunique_indices,
            edge_order,
            grid=(edge_count, n_cols),
            name="linear_dag_jaxlinarg_pallas_gpu_scheduled_backward_update",
            interpret=interpret,
        )
    return state


def _forward_level_update_kernel(
    *,
    level_start: int,
    interpret: bool,
) -> Any:
    def kernel(
        state_ref: Any,
        indptr_ref: Any,
        indices_ref: Any,
        data_ref: Any,
        src_of_edge_ref: Any,
        nonunique_indices_ref: Any,
        edge_order_ref: Any,
        out_ref: Any,
    ) -> None:
        del indptr_ref
        edge_position = pl.program_id(0)
        col = pl.program_id(1)
        edge_index = edge_order_ref[level_start + edge_position]
        src = src_of_edge_ref[edge_index]
        dst = indices_ref[edge_index]
        src_row = nonunique_indices_ref[src]
        dst_row = nonunique_indices_ref[dst]
        increment = state_ref[src_row, col] * data_ref[edge_index]
        _add_to_ref(out_ref, dst_row, col, increment, interpret=interpret)

    return kernel


def _forward_level_zero_kernel(*, level_start: int, min_index_to_keep: int) -> Any:
    def kernel(
        state_ref: Any,
        indptr_ref: Any,
        indices_ref: Any,
        data_ref: Any,
        src_of_edge_ref: Any,
        nonunique_indices_ref: Any,
        edge_order_ref: Any,
        out_ref: Any,
    ) -> None:
        del state_ref, indices_ref, data_ref
        edge_position = pl.program_id(0)
        col = pl.program_id(1)
        edge_index = edge_order_ref[level_start + edge_position]
        src = src_of_edge_ref[edge_index]
        src_row = nonunique_indices_ref[src]
        should_zero = (edge_index == indptr_ref[src + 1] - 1) & (src < min_index_to_keep)

        @pl.when(should_zero)
        def zero_source() -> None:
            out_ref[src_row, col] = 0.0

    return kernel


def _backward_level_zero_kernel(*, level_start: int, min_index_to_keep: int) -> Any:
    def kernel(
        state_ref: Any,
        indptr_ref: Any,
        indices_ref: Any,
        data_ref: Any,
        src_of_edge_ref: Any,
        nonunique_indices_ref: Any,
        edge_order_ref: Any,
        out_ref: Any,
    ) -> None:
        del state_ref, indices_ref, data_ref
        edge_position = pl.program_id(0)
        col = pl.program_id(1)
        edge_index = edge_order_ref[level_start + edge_position]
        src = src_of_edge_ref[edge_index]
        src_row = nonunique_indices_ref[src]
        should_zero = (edge_index == indptr_ref[src + 1] - 1) & (src < min_index_to_keep)

        @pl.when(should_zero)
        def zero_source() -> None:
            out_ref[src_row, col] = 0.0

    return kernel


def _backward_level_update_kernel(
    *,
    level_start: int,
    interpret: bool,
) -> Any:
    def kernel(
        state_ref: Any,
        indptr_ref: Any,
        indices_ref: Any,
        data_ref: Any,
        src_of_edge_ref: Any,
        nonunique_indices_ref: Any,
        edge_order_ref: Any,
        out_ref: Any,
    ) -> None:
        del indptr_ref
        edge_position = pl.program_id(0)
        col = pl.program_id(1)
        edge_index = edge_order_ref[level_start + edge_position]
        src = src_of_edge_ref[edge_index]
        dst = indices_ref[edge_index]
        src_row = nonunique_indices_ref[src]
        dst_row = nonunique_indices_ref[dst]
        increment = state_ref[dst_row, col] * data_ref[edge_index]
        _add_to_ref(out_ref, src_row, col, increment, interpret=interpret)

    return kernel


def _call_scheduled_update_kernel(
    kernel: Any,
    state: Any,
    indptr: Any,
    indices: Any,
    data: Any,
    src_of_edge: Any,
    nonunique_indices: Any,
    edge_order: Any,
    *,
    grid: tuple[int, int],
    name: str,
    interpret: bool,
) -> jax.Array:
    if pl is None:
        raise RuntimeError("Pallas is unavailable.")
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(state.shape, state.dtype),
        grid=grid,
        name=name,
        input_output_aliases={0: 0},
        interpret=interpret,
    )(state, indptr, indices, data, src_of_edge, nonunique_indices, edge_order)


def _add_to_ref(out_ref: Any, row: Any, col: Any, increment: Any, *, interpret: bool) -> None:
    if interpret:
        # CPU interpretation runs one program instance at a time, so ordinary
        # addition is deterministic and keeps tests independent of GPU support.
        out_ref[row, col] = out_ref[row, col] + increment
        return
    if mosaic_gpu is None:  # pragma: no cover - guarded by import availability.
        raise RuntimeError("Pallas Mosaic GPU helpers are unavailable.")
    mosaic_gpu.atomic_add(out_ref.at[row, col], increment)


def _copy_buffer(b_ref: Any, out_ref: Any, *, n_rows: int, n_cols: int) -> None:
    @pl.loop(0, n_rows)
    def row_loop(row: Any) -> None:
        @pl.loop(0, n_cols)
        def col_loop(col: Any) -> None:
            out_ref[row, col] = b_ref[row, col]


def _zero_row(out_ref: Any, row: Any, *, n_cols: int) -> None:
    @pl.loop(0, n_cols)
    def col_loop(col: Any) -> None:
        out_ref[row, col] = 0.0
