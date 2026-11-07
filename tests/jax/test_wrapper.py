# pattern: Functional Core

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import polars as pl
import pytest

from scipy import sparse

from linear_dag.core.jaxlinarg import (
    Backend,
    JaxLinearARG,
    JaxParallelOperator,
    variant_offsets_from_metadata,
)
from linear_dag.core.lineararg import LinearARG


def _fixture_operator(linarg_h5_path, linarg_block_metadata) -> JaxParallelOperator:
    return JaxParallelOperator.from_hdf5(
        linarg_h5_path,
        block_metadata=linarg_block_metadata,
        backend=Backend.PURE_JAX,
    )


def _tiny_block(*, n_samples: int = 1, n_variants: int = 1, dtype=jnp.float32) -> JaxLinearARG:
    return JaxLinearARG.from_lineararg_arrays(
        indptr=np.array([0, 1, 1], dtype=np.int32),
        indices=np.array([1], dtype=np.int32),
        data=np.ones(1, dtype=np.dtype(dtype)),
        variant_indices=np.array([0], dtype=np.int32),
        flip=np.array([False]),
        sample_indices=np.zeros(n_samples, dtype=np.int32),
        nonunique_indices=np.array([0, 1], dtype=np.int32),
        n_variants=n_variants,
        n_samples=n_samples,
        backend=Backend.PURE_JAX,
        dtype=dtype,
    )


def _lineararg_with_graph_shape(*, n_nodes: int, n_edges: int) -> LinearARG:
    rows = np.arange(1, n_edges + 1, dtype=np.int32)
    cols = np.zeros(n_edges, dtype=np.int32)
    return LinearARG(
        sparse.csc_matrix((np.ones(n_edges, dtype=np.float32), (rows, cols)), shape=(n_nodes, n_nodes)),
        variant_indices=np.array([0], dtype=np.int32),
        flip=np.array([False]),
        n_samples=np.int32(1),
        nonunique_indices=np.arange(n_nodes, dtype=np.int32),
    )


def test_jax_parallel_operator_construction_from_hdf5_blocks(
    linarg_h5_path,
    linarg_block_metadata,
):
    op = JaxParallelOperator.from_hdf5(
        linarg_h5_path,
        block_metadata=linarg_block_metadata,
        backend=Backend.PURE_JAX,
    )

    assert len(op.blocks) == linarg_block_metadata.height
    assert op.shape == (
        linarg_block_metadata.get_column("n_samples")[0],
        linarg_block_metadata.get_column("n_variants").sum(),
    )
    np.testing.assert_array_equal(np.asarray(op.variant_offsets), variant_offsets_from_metadata(linarg_block_metadata))


def test_jax_parallel_operator_construction_rejects_mismatched_sample_counts():
    first = _tiny_block(n_samples=1)
    second = _tiny_block(n_samples=2)

    with pytest.raises(ValueError, match="same n_samples"):
        JaxParallelOperator(
            blocks=(first, second),
            variant_offsets=(0, 1, 2),
            backend=Backend.PURE_JAX,
        )


@pytest.mark.skipif(
    not jax.config.read("jax_enable_x64"),
    reason="requires JAX_ENABLE_X64=1 to preserve float64 block dtype",
)
def test_jax_parallel_operator_construction_rejects_mixed_block_dtypes():
    first = _tiny_block(dtype=jnp.float32)
    second = _tiny_block(dtype=jnp.float64)

    with pytest.raises(ValueError, match="same dtype"):
        JaxParallelOperator(
            blocks=(first, second),
            variant_offsets=(0, 1, 2),
            backend=Backend.PURE_JAX,
        )


def test_jax_parallel_operator_construction_rejects_bad_variant_offsets():
    block = _tiny_block()

    with pytest.raises(ValueError, match="variant_offsets"):
        JaxParallelOperator(
            blocks=(block,),
            variant_offsets=(1, 0),
            backend=Backend.PURE_JAX,
        )


def test_jax_parallel_operator_construction_rejects_offsets_that_mismatch_blocks():
    first = _tiny_block(n_variants=1)
    second = _tiny_block(n_variants=1)

    with pytest.raises(ValueError, match="block n_variants"):
        JaxParallelOperator(
            blocks=(first, second),
            variant_offsets=(0, 2, 2),
            backend=Backend.PURE_JAX,
        )


def test_jax_parallel_operator_from_hdf5_rejects_block_metadata_variant_count_mismatch(
    linarg_h5_path,
    linarg_block_metadata,
):
    n_variants = linarg_block_metadata.get_column("n_variants").to_numpy().copy()
    n_variants[0] += 1
    bad_metadata = linarg_block_metadata.with_columns(
        pl.Series("n_variants", n_variants),
    )

    with pytest.raises(ValueError, match="block n_variants"):
        JaxParallelOperator.from_hdf5(
            linarg_h5_path,
            block_metadata=bad_metadata,
            backend=Backend.PURE_JAX,
        )


def test_jax_parallel_operator_from_linearargs_auto_accepts_consistent_prebuilt_backend(monkeypatch):
    block = JaxLinearARG.from_lineararg(
        _lineararg_with_graph_shape(n_nodes=3, n_edges=2),
        backend=Backend.PURE_JAX,
    )

    def resolve_backend(requested):
        requested = Backend(requested)
        return Backend.FFI_CPU if requested is Backend.AUTO else requested

    monkeypatch.setattr("linear_dag.core.jaxlinarg.wrapper.resolve_backend", resolve_backend)

    op = JaxParallelOperator.from_linearargs(
        (block,),
    )

    assert op.backend is Backend.PURE_JAX
    assert op.blocks == (block,)


def test_jax_parallel_operator_from_linearargs_rejects_explicit_prebuilt_backend_mismatch(monkeypatch):
    block = JaxLinearARG.from_lineararg(
        _lineararg_with_graph_shape(n_nodes=3, n_edges=2),
        backend=Backend.PURE_JAX,
    )

    def resolve_backend(requested):
        requested = Backend(requested)
        return Backend.FFI_CPU if requested is Backend.FFI_CPU else requested

    monkeypatch.setattr("linear_dag.core.jaxlinarg.wrapper.resolve_backend", resolve_backend)

    with pytest.raises(ValueError, match="backend"):
        JaxParallelOperator.from_linearargs(
            (block,),
            backend=Backend.FFI_CPU,
        )


def test_jax_parallel_operator_matmat_matches_sum_of_block_products(
    linarg_h5_path,
    linarg_block_metadata,
):
    op = _fixture_operator(linarg_h5_path, linarg_block_metadata)
    w = jnp.arange(op.shape[1] * 3, dtype=jnp.float32).reshape(op.shape[1], 3) / 100.0

    actual = op.matmat(w)
    expected = sum(
        block.matmat(w[start:end])
        for block, start, end in zip(
            op.blocks,
            op.variant_offsets[:-1],
            op.variant_offsets[1:],
            strict=True,
        )
    )

    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=1e-5, atol=1e-5)


def test_jax_parallel_operator_rmatmat_matches_concatenated_block_products(
    linarg_h5_path,
    linarg_block_metadata,
):
    op = _fixture_operator(linarg_h5_path, linarg_block_metadata)
    y = jnp.arange(op.shape[0] * 3, dtype=jnp.float32).reshape(op.shape[0], 3) / 100.0

    actual = op.rmatmat(y)
    expected = jnp.concatenate([block.rmatmat(y) for block in op.blocks], axis=0)

    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=1e-5, atol=1e-5)


def test_jax_parallel_operator_autodiff_matches_concatenated_block_gradients(
    linarg_h5_path,
    linarg_block_metadata,
):
    op = _fixture_operator(linarg_h5_path, linarg_block_metadata)
    w = jnp.arange(op.shape[1] * 2, dtype=jnp.float32).reshape(op.shape[1], 2) / 100.0
    target = jnp.linspace(-1.0, 1.0, op.shape[0] * 2, dtype=jnp.float32).reshape(op.shape[0], 2)

    @jax.jit
    def loss(values):
        residual = op.matmat(values) - target
        return 0.5 * jnp.sum(residual**2)

    residual = op.matmat(w) - target
    expected_blocks = []
    for block, start, end in zip(
        op.blocks,
        op.variant_offsets[:-1],
        op.variant_offsets[1:],
        strict=True,
    ):

        @jax.jit
        def block_pullback(values, *, block=block):
            return jnp.sum(block.matmat(values) * residual)

        expected_blocks.append(jax.grad(block_pullback)(w[start:end]))

    actual = jax.jit(jax.grad(loss))(w)
    expected = jnp.concatenate(expected_blocks, axis=0)

    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("backend", [Backend.PURE_JAX, Backend.FFI_CPU])
@pytest.mark.parametrize("setting", ["num_processes", "max_num_traits", "workspace_bytes"])
def test_execution_controls_require_integers(backend, setting):
    with pytest.raises(TypeError):
        JaxParallelOperator(
            blocks=(_tiny_block(),),
            variant_offsets=(0, 1),
            backend=backend,
            **{setting: 1.5},
        )


@pytest.mark.parametrize("backend", [Backend.PURE_JAX, Backend.FFI_CPU])
@pytest.mark.parametrize("transpose", [False, True])
def test_ragged_products_jit_vmap_and_grad_match_dense(linarg_h5_path, linarg_block_metadata, backend, transpose):
    # Build the reference through the independent Cython graph implementation.
    blocks = tuple(
        LinearARG.read(linarg_h5_path, block=name) for name in linarg_block_metadata.head(2).get_column("block_name")
    )
    assert len({block.shape[1] for block in blocks}) == 2
    dense = np.concatenate([block @ np.eye(block.shape[1], dtype=np.float32) for block in blocks], axis=1)
    op = JaxParallelOperator.from_linearargs(blocks, backend=backend, num_processes=2)
    if transpose:
        dense = dense.T
    product = op.rmatvec if transpose else op.matvec
    values = jnp.asarray(np.random.default_rng(42).normal(size=(3, dense.shape[1])), dtype=jnp.float32)
    expected = np.asarray(values) @ dense.T
    np.testing.assert_allclose(jax.jit(jax.vmap(product))(values), expected, rtol=2e-5, atol=2e-5)
    # A quadratic loss exercises the transpose of the batched product as well.
    gradient = jax.jit(jax.grad(lambda x: 0.5 * jnp.sum(jax.vmap(product)(x) ** 2)))(values)
    np.testing.assert_allclose(gradient, expected @ dense, rtol=2e-5, atol=2e-4)


def test_variant_offsets_from_metadata_returns_leading_zero_cumulative_offsets():
    metadata = pl.DataFrame({"n_variants": [2, 3, 5]})

    offsets = variant_offsets_from_metadata(metadata)

    np.testing.assert_array_equal(offsets, np.array([0, 2, 5, 10], dtype=np.int64))


def test_variant_offsets_from_metadata_requires_n_variants_column():
    metadata = pl.DataFrame({"n_entries": [2, 3]})

    with pytest.raises(ValueError, match='metadata must contain columns "n_variants"'):
        variant_offsets_from_metadata(metadata)
