# pattern: Functional Core

from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import polars as pl
import pytest

from jax.sharding import AbstractMesh, Mesh
from scipy import sparse

from linear_dag.core.jaxlinarg import (
    Backend,
    JaxLinearARG,
    JaxParallelOperator,
    split_blocks_by_n_entries,
    variant_offsets_from_metadata,
)
from linear_dag.core.lineararg import LinearARG


def _mesh(axis_name: str = "blocks") -> Mesh:
    return Mesh(np.asarray(jax.devices()[:1]), (axis_name,))


def _two_device_cpu_mesh_or_skip() -> Mesh:
    devices = jax.devices("cpu")
    if len(devices) < 2:
        pytest.skip(
            "requires at least two CPU devices; set JAX_NUM_CPU_DEVICES=2 "
            "or XLA_FLAGS=--xla_force_host_platform_device_count=2 before JAX import"
        )
    return Mesh(np.asarray(devices[:2]), ("blocks",))


def _fixture_operator(linarg_h5_path, linarg_block_metadata, *, mesh: Mesh | None = None) -> JaxParallelOperator:
    return JaxParallelOperator.from_hdf5(
        linarg_h5_path,
        mesh=_mesh() if mesh is None else mesh,
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
        mesh=_mesh(),
        block_metadata=linarg_block_metadata,
        backend=Backend.PURE_JAX,
    )

    assert len(op.blocks) == linarg_block_metadata.height
    assert op.shape == (
        linarg_block_metadata.get_column("n_samples")[0],
        linarg_block_metadata.get_column("n_variants").sum(),
    )
    np.testing.assert_array_equal(np.asarray(op.variant_offsets), variant_offsets_from_metadata(linarg_block_metadata))
    assert op.block_ranges == split_blocks_by_n_entries(linarg_block_metadata, 1)


def test_jax_parallel_operator_construction_rejects_empty_mesh():
    block = _tiny_block()
    empty_mesh = Mesh(np.asarray([]), ("blocks",))

    with pytest.raises(ValueError, match="mesh must contain at least one device"):
        JaxParallelOperator(
            blocks=(block,),
            variant_offsets=(0, 1),
            mesh=empty_mesh,
            backend=Backend.PURE_JAX,
            block_ranges=((0, 1),),
        )


def test_jax_parallel_operator_construction_rejects_fake_mesh():
    block = _tiny_block()
    fake_mesh = SimpleNamespace(devices=np.asarray(jax.devices()[:1]), axis_names=("blocks",))

    with pytest.raises(TypeError, match="jax.sharding.Mesh"):
        JaxParallelOperator(
            blocks=(block,),
            variant_offsets=(0, 1),
            mesh=fake_mesh,
            backend=Backend.PURE_JAX,
            block_ranges=((0, 1),),
        )


def test_jax_parallel_operator_construction_rejects_mesh_without_blocks_axis():
    block = _tiny_block()

    with pytest.raises(ValueError, match="blocks"):
        JaxParallelOperator(
            blocks=(block,),
            variant_offsets=(0, 1),
            mesh=_mesh("devices"),
            backend=Backend.PURE_JAX,
            block_ranges=((0, 1),),
        )


def test_jax_parallel_operator_construction_rejects_mismatched_sample_counts():
    first = _tiny_block(n_samples=1)
    second = _tiny_block(n_samples=2)

    with pytest.raises(ValueError, match="same n_samples"):
        JaxParallelOperator(
            blocks=(first, second),
            variant_offsets=(0, 1, 2),
            mesh=_mesh(),
            backend=Backend.PURE_JAX,
            block_ranges=((0, 2),),
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
            mesh=_mesh(),
            backend=Backend.PURE_JAX,
            block_ranges=((0, 2),),
        )


def test_jax_parallel_operator_construction_rejects_bad_variant_offsets():
    block = _tiny_block()

    with pytest.raises(ValueError, match="variant_offsets"):
        JaxParallelOperator(
            blocks=(block,),
            variant_offsets=(1, 0),
            mesh=_mesh(),
            backend=Backend.PURE_JAX,
            block_ranges=((0, 1),),
        )


def test_jax_parallel_operator_construction_rejects_offsets_that_mismatch_blocks():
    first = _tiny_block(n_variants=1)
    second = _tiny_block(n_variants=1)

    with pytest.raises(ValueError, match="block n_variants"):
        JaxParallelOperator(
            blocks=(first, second),
            variant_offsets=(0, 2, 2),
            mesh=_mesh(),
            backend=Backend.PURE_JAX,
            block_ranges=((0, 2),),
        )


def test_jax_parallel_operator_construction_rejects_block_ranges_that_do_not_cover_mesh_axis():
    first = _tiny_block(n_variants=1)
    second = _tiny_block(n_variants=1)
    mesh = AbstractMesh((2,), ("blocks",))

    with pytest.raises(ValueError, match="block_ranges"):
        JaxParallelOperator(
            blocks=(first, second),
            variant_offsets=(0, 1, 2),
            mesh=mesh,
            backend=Backend.PURE_JAX,
            block_ranges=((0, 1),),
        )


def test_jax_parallel_operator_construction_rejects_noncontiguous_block_ranges():
    first = _tiny_block(n_variants=1)
    second = _tiny_block(n_variants=1)
    mesh = AbstractMesh((2,), ("blocks",))

    with pytest.raises(ValueError, match="contiguous"):
        JaxParallelOperator(
            blocks=(first, second),
            variant_offsets=(0, 1, 2),
            mesh=mesh,
            backend=Backend.PURE_JAX,
            block_ranges=((0, 1), (2, 2)),
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
            mesh=_mesh(),
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
        mesh=_mesh(),
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
            mesh=_mesh(),
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


def test_jax_parallel_operator_rmatmat_uses_concatenated_cached_path(monkeypatch):
    def reject_shard_map(*args, **kwargs):
        raise AssertionError("rmatmat should not use shard_map")

    monkeypatch.setattr("linear_dag.core.jaxlinarg.wrapper.jax.shard_map", reject_shard_map)
    op = JaxParallelOperator(
        blocks=(_tiny_block(), _tiny_block()),
        variant_offsets=(0, 1, 2),
        mesh=AbstractMesh((2,), ("blocks",)),
        backend=Backend.PURE_JAX,
        block_ranges=((0, 1), (1, 2)),
    )
    y = jnp.ones((op.shape[0], 1), dtype=jnp.float32)

    actual = op.rmatmat(y)
    expected = jnp.concatenate([block.rmatmat(y) for block in op.blocks], axis=0)

    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=1e-5, atol=1e-5)


def test_jax_parallel_operator_products_on_two_device_cpu_mesh(
    linarg_h5_path,
    linarg_block_metadata,
    monkeypatch,
):
    shard_map_calls = []

    def recording_shard_map(*args, **kwargs):
        shard_map_calls.append(kwargs)
        return original_shard_map(*args, **kwargs)

    original_shard_map = jax.shard_map
    monkeypatch.setattr("linear_dag.core.jaxlinarg.wrapper.jax.shard_map", recording_shard_map)
    op = _fixture_operator(linarg_h5_path, linarg_block_metadata, mesh=_two_device_cpu_mesh_or_skip())
    w = jnp.ones((op.shape[1], 2), dtype=jnp.float32)
    y = jnp.ones((op.shape[0], 2), dtype=jnp.float32)

    expected_matmat = sum(
        block.matmat(w[start:end])
        for block, start, end in zip(
            op.blocks,
            op.variant_offsets[:-1],
            op.variant_offsets[1:],
            strict=True,
        )
    )
    expected_rmatmat = jnp.concatenate([block.rmatmat(y) for block in op.blocks], axis=0)

    np.testing.assert_allclose(np.asarray(op.matmat(w)), np.asarray(expected_matmat), rtol=1e-5, atol=1e-5)
    assert len(shard_map_calls) == 1

    np.testing.assert_allclose(np.asarray(op.rmatmat(y)), np.asarray(expected_rmatmat), rtol=1e-5, atol=1e-5)
    assert len(shard_map_calls) == 1
    assert all(call["axis_names"] == {"blocks"} for call in shard_map_calls)


def test_jax_parallel_operator_autodiff_matches_concatenated_block_gradients(
    linarg_h5_path,
    linarg_block_metadata,
    monkeypatch,
):
    shard_map_calls = []

    def recording_shard_map(*args, **kwargs):
        shard_map_calls.append(kwargs)
        return original_shard_map(*args, **kwargs)

    original_shard_map = jax.shard_map
    monkeypatch.setattr("linear_dag.core.jaxlinarg.wrapper.jax.shard_map", recording_shard_map)
    op = _fixture_operator(linarg_h5_path, linarg_block_metadata, mesh=_two_device_cpu_mesh_or_skip())
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
    assert shard_map_calls
