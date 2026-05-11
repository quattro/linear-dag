# pattern: Functional Core

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import numpy as np

try:
    import jax.experimental.pallas as pl
except Exception:  # pragma: no cover - depends on the installed JAX build.
    pl = None


class LevelSchedule(NamedTuple):
    edge_order: np.ndarray
    level_offsets: np.ndarray


def is_pallas_gpu_available() -> bool:
    """Return whether the Pallas GPU backend can be used by this process."""
    return jax.default_backend() == "gpu" and pl is not None


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
) -> jax.Array:
    """Solve with the level-scheduling branch, currently backed by the serial scan."""
    _check_level_schedule(schedule, n_edges=int(indices.shape[0]))
    return pallas_gpu_solve_forward(
        indptr,
        indices,
        data,
        src_of_edge,
        nonunique_indices,
        min_index_to_keep,
        b,
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
) -> jax.Array:
    """Solve transpose with the level-scheduling branch, currently serial."""
    _check_level_schedule(schedule, n_edges=int(indices.shape[0]))
    return pallas_gpu_solve_backward(
        indptr,
        indices,
        data,
        src_of_edge,
        nonunique_indices,
        min_index_to_keep,
        b,
    )


def _require_pallas_gpu() -> None:
    if not is_pallas_gpu_available():
        raise RuntimeError(
            "Pallas GPU backend is unavailable; expected jax.default_backend() == 'gpu' "
            "and importable jax.experimental.pallas."
        )


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


def _check_level_schedule(schedule: LevelSchedule, *, n_edges: int) -> None:
    if schedule.edge_order.ndim != 1:
        raise ValueError("level schedule edge_order must be rank 1")
    if schedule.level_offsets.ndim != 1:
        raise ValueError("level schedule level_offsets must be rank 1")
    if int(schedule.edge_order.shape[0]) != n_edges:
        raise ValueError("level schedule edge_order length must match the edge count")
    if schedule.level_offsets.shape[0] == 0:
        raise ValueError("level schedule level_offsets must contain at least one entry")
    if int(schedule.level_offsets[0]) != 0 or int(schedule.level_offsets[-1]) != n_edges:
        raise ValueError("level schedule offsets must span the edge count")


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
