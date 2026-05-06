# pattern: Functional Core

from __future__ import annotations

from typing import Any

import jax

from jax import lax


def pure_jax_solve_forward(
    indptr: Any,
    indices: Any,
    data: Any,
    src_of_edge: Any,
    b: Any,
    *,
    n_edges: int,
) -> jax.Array:
    """Solve a lower-triangular LinearARG node buffer by forward edge scan."""
    del indptr

    def step(carry: jax.Array, edge_index: jax.Array) -> tuple[jax.Array, None]:
        src = src_of_edge[edge_index]
        dst = indices[edge_index]
        weight = data[edge_index]
        return carry.at[dst, :].add(carry[src, :] * weight), None

    edge_indices = jax.numpy.arange(n_edges, dtype=indices.dtype)
    result, _ = lax.scan(step, b, edge_indices)
    return result


def pure_jax_solve_backward(
    indptr: Any,
    indices: Any,
    data: Any,
    src_of_edge: Any,
    b: Any,
    *,
    n_edges: int,
) -> jax.Array:
    """Solve a transposed LinearARG node buffer by backward edge scan."""
    del indptr

    def step(carry: jax.Array, edge_index: jax.Array) -> tuple[jax.Array, None]:
        src = src_of_edge[edge_index]
        dst = indices[edge_index]
        weight = data[edge_index]
        return carry.at[src, :].add(carry[dst, :] * weight), None

    edge_indices = jax.numpy.arange(n_edges - 1, -1, -1, dtype=indices.dtype)
    result, _ = lax.scan(step, b, edge_indices)
    return result
