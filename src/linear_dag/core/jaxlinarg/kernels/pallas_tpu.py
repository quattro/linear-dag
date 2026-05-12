# pattern: Functional Core

from __future__ import annotations

from typing import Any

import jax

try:
    import jax.experimental.pallas as pl

    from jax.experimental.pallas import tpu as pltpu
except Exception:  # pragma: no cover - depends on the installed JAX build.
    pl = None
    pltpu = None


def is_pallas_import_available() -> bool:
    """Return whether the Pallas module imported successfully."""
    return pl is not None


def is_pallas_tpu_available() -> bool:
    """Return whether the Pallas TPU backend can be used by this process."""
    return jax.default_backend() == "tpu" and is_pallas_import_available()


def pallas_tpu_solve_forward(
    indptr: Any,
    indices: Any,
    data: Any,
    src_of_edge: Any,
    nonunique_indices: Any,
    min_index_to_keep: int,
    b: Any,
    *,
    interpret: bool = False,
) -> jax.Array:
    """Solve a compressed LinearARG node buffer with a serial TPU Pallas edge scan."""
    if not interpret:
        _require_pallas_tpu()
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
        name="linear_dag_jaxlinarg_pallas_tpu_solve_forward",
        interpret=interpret,
    )


def pallas_tpu_solve_backward(
    indptr: Any,
    indices: Any,
    data: Any,
    src_of_edge: Any,
    nonunique_indices: Any,
    min_index_to_keep: int,
    b: Any,
    *,
    interpret: bool = False,
) -> jax.Array:
    """Solve a compressed transposed LinearARG buffer with a serial TPU Pallas edge scan."""
    if not interpret:
        _require_pallas_tpu()
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
        name="linear_dag_jaxlinarg_pallas_tpu_solve_backward",
        interpret=interpret,
    )


def _require_pallas_tpu() -> None:
    if not is_pallas_tpu_available():
        raise RuntimeError(
            "Pallas TPU backend is unavailable; expected jax.default_backend() == 'tpu' "
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
    interpret: bool,
) -> jax.Array:
    if pl is None:
        raise RuntimeError("Pallas is unavailable.")
    compiler_params = None
    if not interpret and pltpu is not None:
        compiler_params = pltpu.CompilerParams(dimension_semantics=["arbitrary"])
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(b.shape, b.dtype),
        grid=(1,),
        name=name,
        interpret=interpret,
        compiler_params=compiler_params,
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
