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
    b: Any,
) -> jax.Array:
    """Solve a lower-triangular LinearARG node buffer by forward node scan."""
    n_nodes = indptr.shape[0] - 1

    def node_step(carry: jax.Array, node: jax.Array) -> tuple[jax.Array, None]:
        edge_start = indptr[node]
        edge_stop = indptr[node + 1]

        def edge_step(edge: jax.Array, values: jax.Array) -> jax.Array:
            dst = indices[edge]
            return values.at[dst, :].add(values[node, :] * data[edge])

        return lax.fori_loop(edge_start, edge_stop, edge_step, carry), None

    nodes = jnp.arange(n_nodes, dtype=indices.dtype)
    result, _ = lax.scan(node_step, b, nodes)
    return result


def pure_jax_solve_backward(
    indptr: Any,
    indices: Any,
    data: Any,
    b: Any,
) -> jax.Array:
    """Solve a transposed LinearARG node buffer by backward node scan."""
    n_nodes = indptr.shape[0] - 1

    def node_step(carry: jax.Array, node: jax.Array) -> tuple[jax.Array, None]:
        edge_start = indptr[node]
        edge_stop = indptr[node + 1]

        def edge_step(edge: jax.Array, values: jax.Array) -> jax.Array:
            dst = indices[edge]
            return values.at[node, :].add(values[dst, :] * data[edge])

        return lax.fori_loop(edge_start, edge_stop, edge_step, carry), None

    nodes = jnp.arange(n_nodes - 1, -1, -1, dtype=indices.dtype)
    result, _ = lax.scan(node_step, b, nodes)
    return result


def pure_jax_solve_forward_compressed(
    indptr: Any,
    indices: Any,
    data: Any,
    nonunique_indices: Any,
    min_index_to_keep: int,
    b: Any,
) -> jax.Array:
    """Solve a compressed LinearARG node buffer by forward node scan."""
    n_nodes = indptr.shape[0] - 1

    def node_step(carry: jax.Array, node: jax.Array) -> tuple[jax.Array, None]:
        edge_start = indptr[node]
        edge_stop = indptr[node + 1]
        src_col = nonunique_indices[node]

        def edge_step(edge: jax.Array, values: jax.Array) -> jax.Array:
            dst_col = nonunique_indices[indices[edge]]
            return values.at[dst_col, :].add(values[src_col, :] * data[edge])

        updated = lax.fori_loop(edge_start, edge_stop, edge_step, carry)
        should_zero = (edge_start != edge_stop) & (node < min_index_to_keep)
        updated = lax.cond(
            should_zero,
            lambda values: values.at[src_col, :].set(jnp.zeros_like(values[src_col, :])),
            lambda values: values,
            updated,
        )
        return updated, None

    nodes = jnp.arange(n_nodes, dtype=indices.dtype)
    result, _ = lax.scan(node_step, b, nodes)
    return result


def pure_jax_solve_backward_compressed(
    indptr: Any,
    indices: Any,
    data: Any,
    nonunique_indices: Any,
    min_index_to_keep: int,
    b: Any,
) -> jax.Array:
    """Solve a compressed transposed LinearARG node buffer by backward node scan."""
    n_nodes = indptr.shape[0] - 1

    def node_step(carry: jax.Array, node: jax.Array) -> tuple[jax.Array, None]:
        edge_start = indptr[node]
        edge_stop = indptr[node + 1]
        dst_col = nonunique_indices[node]
        should_zero = (edge_start != edge_stop) & (node < min_index_to_keep)
        carry = lax.cond(
            should_zero,
            lambda values: values.at[dst_col, :].set(jnp.zeros_like(values[dst_col, :])),
            lambda values: values,
            carry,
        )

        def edge_step(edge: jax.Array, values: jax.Array) -> jax.Array:
            src_col = nonunique_indices[indices[edge]]
            return values.at[dst_col, :].add(values[src_col, :] * data[edge])

        updated = lax.fori_loop(edge_start, edge_stop, edge_step, carry)
        return updated, None

    nodes = jnp.arange(n_nodes - 1, -1, -1, dtype=indices.dtype)
    result, _ = lax.scan(node_step, b, nodes)
    return result
