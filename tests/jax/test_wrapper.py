# pattern: Functional Core

from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax.sharding import Mesh

from linear_dag.core.jaxlinarg import (
    Backend,
    JaxLinearARG,
    JaxParallelOperator,
    split_blocks_by_n_entries,
    variant_offsets_from_metadata,
)


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


def _tiny_block(*, n_samples: int = 1, n_variants: int = 1) -> JaxLinearARG:
    return JaxLinearARG.from_lineararg_arrays(
        indptr=np.array([0, 1, 1], dtype=np.int32),
        indices=np.array([1], dtype=np.int32),
        data=np.ones(1, dtype=np.float32),
        src_of_edge=np.array([0], dtype=np.int32),
        variant_indices=np.array([0], dtype=np.int32),
        flip=np.array([False]),
        sample_indices=np.zeros(n_samples, dtype=np.int32),
        nonunique_indices=np.array([0, 1], dtype=np.int32),
        n_variants=n_variants,
        n_samples=n_samples,
        backend=Backend.PURE_JAX,
        dtype=jnp.float32,
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
    empty_mesh = SimpleNamespace(devices=np.asarray([]), axis_names=("blocks",))

    with pytest.raises(ValueError, match="mesh must contain at least one device"):
        JaxParallelOperator(
            blocks=(block,),
            variant_offsets=(0, 1),
            mesh=empty_mesh,
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


def test_jax_parallel_operator_products_on_two_device_cpu_mesh(
    linarg_h5_path,
    linarg_block_metadata,
):
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
    np.testing.assert_allclose(np.asarray(op.rmatmat(y)), np.asarray(expected_rmatmat), rtol=1e-5, atol=1e-5)
