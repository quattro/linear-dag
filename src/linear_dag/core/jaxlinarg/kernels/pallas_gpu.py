# pattern: Functional Core

from __future__ import annotations

from typing import Any

import jax

try:
    import jax.experimental.pallas as pl
except Exception:  # pragma: no cover - depends on the installed JAX build.
    pl = None


def is_pallas_gpu_available() -> bool:
    """Return whether the Pallas GPU backend can be used by this process."""
    return jax.default_backend() == "gpu" and pl is not None


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
