# pattern: Imperative Shell

from __future__ import annotations

import shutil

import h5py
import jax.numpy as jnp
import numpy as np

from linear_dag.core.jaxlinarg import Backend, JaxLinearARG
from linear_dag.core.jaxlinarg.ingress import (
    from_block_arrays,
    from_lineararg,
    read_hdf5_block_arrays,
    read_hdf5_blocks,
)
from linear_dag.core.lineararg import LinearARG


def test_from_lineararg_matches_in_memory_lineararg_products(oracle_case) -> None:
    op = from_lineararg(oracle_case.linarg, backend=Backend.PURE_JAX)

    assert op.shape == oracle_case.linarg.shape
    np.testing.assert_allclose(np.asarray(op.matmat(oracle_case.w)), oracle_case.Xw, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.asarray(op.rmatmat(oracle_case.y)), oracle_case.XTy, rtol=1e-5, atol=1e-5)


def test_from_lineararg_canonicalizes_arrays_and_preserves_cached_allele_counts(oracle_case) -> None:
    linarg = oracle_case.linarg
    op = from_lineararg(linarg, backend=Backend.PURE_JAX)

    assert op.indptr.dtype == jnp.int32
    assert op.indices.dtype == jnp.int32
    assert op.variant_indices.dtype == jnp.int32
    assert op.sample_indices.dtype == jnp.int32
    assert op.nonunique_indices.dtype == jnp.int32
    assert op.data.dtype == jnp.float32
    assert op.flip.dtype == jnp.bool_
    np.testing.assert_array_equal(np.asarray(op.allele_counts), linarg.allele_counts.astype(np.int32))


def test_jax_lineararg_from_hdf5_block_matches_lineararg_read(
    linarg_h5_path,
    first_block_name,
) -> None:
    linarg = LinearARG.read(linarg_h5_path, block=first_block_name)
    op = JaxLinearARG.from_hdf5_block(linarg_h5_path, first_block_name, backend=Backend.PURE_JAX)
    rng = np.random.default_rng(20260507)
    w = rng.normal(size=(linarg.shape[1], 3)).astype(np.float32)
    y = rng.normal(size=(linarg.shape[0], 3)).astype(np.float32)

    assert op.shape == linarg.shape
    np.testing.assert_allclose(np.asarray(op.matmat(w)), np.asarray(linarg @ w), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.asarray(op.rmatmat(y)), np.asarray(linarg.T @ y), rtol=1e-5, atol=1e-5)


def test_jax_lineararg_from_hdf5_block_uses_native_array_ingress(
    linarg_h5_path,
    first_block_name,
    monkeypatch,
) -> None:
    def fail_read(*args, **kwargs):
        raise AssertionError("JAX HDF5 ingress should not materialize LinearARG")

    monkeypatch.setattr(LinearARG, "read", fail_read)

    op = JaxLinearARG.from_hdf5_block(linarg_h5_path, first_block_name, backend=Backend.PURE_JAX)

    assert op.shape[0] > 0
    assert op.shape[1] > 0


def test_read_hdf5_block_arrays_canonicalizes_optional_arrays(
    linarg_h5_path,
    first_block_name,
    tmp_path,
) -> None:
    path = tmp_path / "missing_optional.h5"
    shutil.copyfile(linarg_h5_path, path)
    with h5py.File(path, "a") as h5f:
        group = h5f[first_block_name]
        del group["nonunique_indices"]
        del group["allele_counts"]

    arrays = read_hdf5_block_arrays(path, first_block_name)
    op = from_block_arrays(arrays, backend=Backend.PURE_JAX)

    np.testing.assert_array_equal(np.asarray(op.nonunique_indices), np.arange(op.n_nonunique_indices, dtype=np.int32))
    np.testing.assert_array_equal(np.asarray(op.allele_counts), np.full(op.n_variants, -1, dtype=np.int32))


def test_read_hdf5_blocks_returns_jax_block_tuple(
    linarg_h5_path,
    linarg_block_metadata,
) -> None:
    block_names = tuple(linarg_block_metadata.get_column("block_name").to_list()[:2])

    blocks = read_hdf5_blocks(linarg_h5_path, block_names, backend=Backend.PURE_JAX)

    assert isinstance(blocks, tuple)
    assert len(blocks) == len(block_names)
    for block, block_name in zip(blocks, block_names, strict=True):
        expected = JaxLinearARG.from_hdf5_block(linarg_h5_path, block_name, backend=Backend.PURE_JAX)
        np.testing.assert_array_equal(np.asarray(block.indptr), np.asarray(expected.indptr))
        np.testing.assert_array_equal(np.asarray(block.indices), np.asarray(expected.indices))
        np.testing.assert_array_equal(np.asarray(block.variant_indices), np.asarray(expected.variant_indices))
        assert block.shape == expected.shape
