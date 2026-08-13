# pattern: Functional Core

"""Portable JAX kernels for forward and transposed LinearARG solves."""

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
    """Propagate a node buffer through a LinearARG in topological order.

    **Arguments:**

    - `indptr`: CSC edge offsets for each source node.
    - `indices`: Destination node for each edge.
    - `data`: Edge coefficients aligned to `indices`.
    - `b`: Rank-2 node-by-trait input buffer.

    **Returns:**

    - The propagated node buffer with the same shape and dtype as `b`.
    """
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
    """Propagate a node buffer through the transposed LinearARG.

    **Arguments:**

    - `indptr`: CSC edge offsets for each source node.
    - `indices`: Destination node for each edge.
    - `data`: Edge coefficients aligned to `indices`.
    - `b`: Rank-2 node-by-trait input buffer.

    **Returns:**

    - The propagated node buffer with the same shape and dtype as `b`.
    """
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
    """Propagate a compressed node buffer in topological order.

    **Arguments:**

    - `indptr`: CSC edge offsets for each source node.
    - `indices`: Destination node for each edge.
    - `data`: Edge coefficients aligned to `indices`.
    - `nonunique_indices`: Mapping from graph nodes to reusable buffer rows.
    - `min_index_to_keep`: First graph node whose buffer row must remain live.
    - `b`: Rank-2 compressed-node-by-trait input buffer.

    **Returns:**

    - The propagated compressed buffer with the same shape and dtype as `b`.
    """
    n_nodes = indptr.shape[0] - 1

    def node_step(carry: jax.Array, node: jax.Array) -> tuple[jax.Array, None]:
        edge_start = indptr[node]
        edge_stop = indptr[node + 1]
        src_col = nonunique_indices[node]

        def edge_step(edge: jax.Array, values: jax.Array) -> jax.Array:
            dst_col = nonunique_indices[indices[edge]]
            return values.at[dst_col, :].add(values[src_col, :] * data[edge])

        updated = lax.fori_loop(edge_start, edge_stop, edge_step, carry)
        # Internal node rows can be reused after all outgoing edges have read
        # them. Sample rows at or above `min_index_to_keep` remain live outputs.
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
    """Propagate a compressed node buffer through the transposed LinearARG.

    **Arguments:**

    - `indptr`: CSC edge offsets for each source node.
    - `indices`: Destination node for each edge.
    - `data`: Edge coefficients aligned to `indices`.
    - `nonunique_indices`: Mapping from graph nodes to reusable buffer rows.
    - `min_index_to_keep`: First graph node whose buffer row must remain live.
    - `b`: Rank-2 compressed-node-by-trait input buffer.

    **Returns:**

    - The propagated compressed buffer with the same shape and dtype as `b`.
    """
    n_nodes = indptr.shape[0] - 1

    def node_step(carry: jax.Array, node: jax.Array) -> tuple[jax.Array, None]:
        edge_start = indptr[node]
        edge_stop = indptr[node + 1]
        dst_col = nonunique_indices[node]
        # Reverse traversal must clear a reusable internal row before earlier
        # nodes accumulate into it. Sample rows remain live as output seeds.
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
