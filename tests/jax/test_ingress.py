# pattern: Imperative Shell

from __future__ import annotations

import gc
import shutil
import weakref

from dataclasses import fields

import equinox as eqx
import h5py
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np

from jax.sharding import Mesh, NamedSharding

import linear_dag.core.jaxlinarg.ingress as ingress_module

from linear_dag.core.jaxlinarg import Backend, JaxLinearARG
from linear_dag.core.jaxlinarg.ingress import (
    _packed_from_block_arrays,
    _packed_from_group_reader,
    _packed_from_hdf5,
    _PackedJaxLinearARG,
    from_block_arrays,
    from_lineararg,
    read_hdf5_block_arrays,
    read_hdf5_blocks,
)
from linear_dag.core.jaxlinarg.packing import pack_blocks, PACKED_COMPONENT_NAMES
from linear_dag.core.lineararg import LinearARG


def _graph_mesh(num_devices: int = 1) -> Mesh:
    return Mesh(np.asarray(jax.devices("cpu")[:num_devices]), ("graph",))


def _assert_packed_arrays_match_host(op: _PackedJaxLinearARG, expected) -> None:
    for name in PACKED_COMPONENT_NAMES:
        np.testing.assert_array_equal(np.asarray(getattr(op, name)), getattr(expected.buffers, name), err_msg=name)


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


def test_private_packed_constructor_preserves_fixed_components_from_canonical_arrays(
    linarg_h5_path,
    first_block_name,
) -> None:
    arrays = read_hdf5_block_arrays(linarg_h5_path, first_block_name)
    mesh = _graph_mesh()

    result = _packed_from_block_arrays((arrays,), mesh=mesh)
    op = result.operator
    expected = pack_blocks((arrays,), num_devices=1)

    assert isinstance(op, _PackedJaxLinearARG)
    assert isinstance(op, eqx.Module)
    assert op.shape == (arrays.n_samples, arrays.n_variants)
    assert op.capacities == tuple(expected.plan.capacities.values())
    assert len(jtu.tree_leaves(op)) == 1
    _assert_packed_arrays_match_host(op, expected)
    for name in PACKED_COMPONENT_NAMES:
        array = getattr(op, name)
        assert isinstance(array.sharding, NamedSharding)
        assert array.sharding.mesh.axis_names == ("graph",)


def test_private_packed_hdf5_and_generic_group_reader_match_host_packing(
    linarg_h5_path,
    linarg_block_metadata,
) -> None:
    block_names = tuple(linarg_block_metadata.get_column("block_name").to_list())
    mesh = _graph_mesh()
    blocks = tuple(read_hdf5_block_arrays(linarg_h5_path, block_name) for block_name in block_names)
    expected = pack_blocks(blocks, num_devices=1)

    from_hdf5 = _packed_from_hdf5(linarg_h5_path, block_names, mesh=mesh)
    with h5py.File(linarg_h5_path, "r") as file:
        reader = type("GroupReader", (), {"root": {"blocks": file}})()
        from_groups = _packed_from_group_reader(reader, block_names, mesh=mesh)

    _assert_packed_arrays_match_host(from_hdf5.operator, expected)
    _assert_packed_arrays_match_host(from_groups.operator, expected)
    assert from_hdf5.diagnostics.staging_block_owners == (0,) * len(block_names)
    assert from_groups.diagnostics.staging_block_owners == (0,) * len(block_names)


def test_packed_hdf5_reads_each_block_once_with_one_live_source_block(
    linarg_h5_path,
    linarg_block_metadata,
    monkeypatch,
) -> None:
    block_names = tuple(linarg_block_metadata.get_column("block_name").to_list())
    reads: list[str] = []
    live_blocks: set[int] = set()
    peak_live_blocks = 0
    original = ingress_module._read_block_arrays_from_group

    def tracked_read(group, *, dtype=None):
        nonlocal peak_live_blocks
        block = original(group, dtype=dtype)
        block_id = len(reads)
        reads.append(group.name.rsplit("/", maxsplit=1)[-1])
        live_blocks.add(block_id)
        peak_live_blocks = max(peak_live_blocks, len(live_blocks))
        weakref.finalize(block, live_blocks.discard, block_id)
        return block

    monkeypatch.setattr(ingress_module, "_read_block_arrays_from_group", tracked_read)

    _packed_from_hdf5(linarg_h5_path, block_names, mesh=_graph_mesh())
    gc.collect()

    assert reads == list(block_names)
    assert peak_live_blocks == 1
    assert not live_blocks


def test_packed_carrier_static_definition_excludes_dataset_diagnostics(
    linarg_h5_path,
    linarg_block_metadata,
) -> None:
    mesh = _graph_mesh()
    block_names = tuple(linarg_block_metadata.get_column("block_name").to_list())
    first = _packed_from_block_arrays(
        (read_hdf5_block_arrays(linarg_h5_path, block_names[0]),),
        mesh=mesh,
    )
    second = _packed_from_block_arrays(
        (read_hdf5_block_arrays(linarg_h5_path, block_names[1]),),
        mesh=mesh,
    )

    expected_static_fields = ("n_samples", "n_variants", "capacities", "graph_mesh")
    for result in (first, second):
        static_fields = tuple(field.name for field in fields(result.operator) if field.metadata.get("static"))
        assert static_fields == expected_static_fields
        assert result.operator.graph_mesh == mesh
        assert not hasattr(result.operator, "diagnostics")
        assert len(jtu.tree_leaves(result.operator)) == 1
        assert type(result.operator) is _PackedJaxLinearARG
