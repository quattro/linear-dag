# pattern: Functional Core

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

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


def pure_jax_solve_forward_compressed(
    indptr: Any,
    indices: Any,
    data: Any,
    src_of_edge: Any,
    nonunique_indices: Any,
    b: Any,
    *,
    min_index_to_keep: int,
    n_edges: int,
) -> jax.Array:
    """Solve a compressed LinearARG node buffer by forward edge scan."""

    def step(carry: jax.Array, edge_index: jax.Array) -> tuple[jax.Array, None]:
        src = src_of_edge[edge_index]
        dst = indices[edge_index]
        src_col = nonunique_indices[src]
        dst_col = nonunique_indices[dst]
        source = carry[src_col, :]
        updated = carry.at[dst_col, :].add(source * data[edge_index])
        should_zero = (edge_index == indptr[src + 1] - 1) & (src < min_index_to_keep)
        updated = lax.cond(
            should_zero,
            lambda values: values.at[src_col, :].set(jnp.zeros_like(source)),
            lambda values: values,
            updated,
        )
        return updated, None

    edge_indices = jnp.arange(n_edges, dtype=indices.dtype)
    result, _ = lax.scan(step, b, edge_indices)
    return result


def pure_jax_solve_backward_compressed(
    indptr: Any,
    indices: Any,
    data: Any,
    src_of_edge: Any,
    nonunique_indices: Any,
    b: Any,
    *,
    min_index_to_keep: int,
    n_edges: int,
) -> jax.Array:
    """Solve a compressed transposed LinearARG node buffer by backward edge scan."""

    def step(carry: jax.Array, edge_index: jax.Array) -> tuple[jax.Array, None]:
        src = src_of_edge[edge_index]
        dst = indices[edge_index]
        src_col = nonunique_indices[src]
        dst_col = nonunique_indices[dst]
        should_zero = (edge_index == indptr[src + 1] - 1) & (src < min_index_to_keep)
        carry = lax.cond(
            should_zero,
            lambda values: values.at[src_col, :].set(jnp.zeros_like(values[src_col, :])),
            lambda values: values,
            carry,
        )
        return carry.at[src_col, :].add(carry[dst_col, :] * data[edge_index]), None

    edge_indices = jnp.arange(n_edges - 1, -1, -1, dtype=indices.dtype)
    result, _ = lax.scan(step, b, edge_indices)
    return result
