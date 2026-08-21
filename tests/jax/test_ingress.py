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
import polars as pl
import pytest

from jax.sharding import Mesh, NamedSharding

import linear_dag.core.jaxlinarg.ingress as ingress_module

from linear_dag.core.jaxlinarg import Backend, JaxLinearARG, JaxParallelOperator
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
from linear_dag.core.jaxlinarg.packing import BLOCK_DESCRIPTOR_FIELDS, pack_blocks, PACKED_COMPONENT_NAMES
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


def test_packed_block_arrays_resolves_explicit_backend_before_source_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unavailable(_backend: Backend) -> Backend:
        raise RuntimeError("explicit FFI backend unavailable")

    monkeypatch.setattr(ingress_module, "_resolve_packed_backend", unavailable)

    with pytest.raises(RuntimeError, match="explicit FFI backend unavailable"):
        _packed_from_block_arrays(
            (block for block in ()),
            mesh=_graph_mesh(),
            backend=Backend.FFI_CPU,
        )


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

    expected_static_fields = ("n_samples", "n_variants", "capacities", "graph_mesh", "backend", "iids")
    for result in (first, second):
        static_fields = tuple(field.name for field in fields(result.operator) if field.metadata.get("static"))
        assert static_fields == expected_static_fields
        assert result.operator.graph_mesh == mesh
        assert not hasattr(result.operator, "diagnostics")
        assert len(jtu.tree_leaves(result.operator)) == 1
        assert type(result.operator) is _PackedJaxLinearARG


def test_private_packed_target_in_memory_constructors_preserve_contract(oracle_case) -> None:
    block = oracle_case.linarg
    arrays = ingress_module._lineararg_block_arrays(block, dtype=np.float64)

    from_arrays = _PackedJaxLinearARG.from_lineararg_arrays(
        indptr=arrays.indptr,
        indices=arrays.indices,
        data=arrays.data,
        variant_indices=arrays.variant_indices,
        flip=arrays.flip,
        sample_indices=arrays.sample_indices,
        nonunique_indices=arrays.nonunique_indices,
        allele_counts=arrays.allele_counts,
        n_variants=arrays.n_variants,
        n_samples=arrays.n_samples,
        iids=block.iids,
        dtype=np.float64,
    )
    from_lineararg = _PackedJaxLinearARG.from_lineararg(block, dtype=np.float64)
    from_plural = _PackedJaxLinearARG.from_linearargs((block,), mesh=_graph_mesh(), dtype=np.float64)

    expected_iids = tuple(block.iids.to_list())
    expected_dtype = jax.dtypes.canonicalize_dtype(np.dtype(np.float64))
    for operator in (from_arrays, from_lineararg, from_plural):
        assert operator.shape == block.shape
        assert operator.dtype == expected_dtype
        assert operator.iids == expected_iids
        np.testing.assert_allclose(np.asarray(operator.matmat(oracle_case.w)), oracle_case.Xw, rtol=1e-5, atol=1e-5)


def test_private_packed_hdf5_constructor_validates_metadata_and_preserves_subset_order(
    linarg_h5_path,
    linarg_block_metadata,
) -> None:
    metadata = linarg_block_metadata.select("block_name", "n", "n_entries", "n_variants", "n_samples")
    selected = metadata.head(1)

    operator = _PackedJaxLinearARG.from_hdf5(
        linarg_h5_path,
        mesh=_graph_mesh(),
        block_metadata=selected,
    )
    expected = LinearARG.read(linarg_h5_path, block=selected["block_name"][0])

    assert operator.shape == expected.shape
    assert expected.iids is not None
    assert operator.iids == tuple(expected.iids.to_list())

    reordered = metadata.reverse()
    with pytest.raises(ValueError, match="block order"):
        _PackedJaxLinearARG.from_hdf5(
            linarg_h5_path,
            mesh=_graph_mesh(),
            block_metadata=reordered,
            max_padding_ratio=None,
        )

    mismatched = selected.with_columns((pl.col("n_entries") + 1).alias("n_entries"))
    with pytest.raises(ValueError, match="n_entries"):
        _PackedJaxLinearARG.from_hdf5(
            linarg_h5_path,
            mesh=_graph_mesh(),
            block_metadata=mismatched,
        )


def test_private_packed_root_hdf5_matches_lineararg_and_rejects_mixed_layout(
    linarg_h5_path,
    first_block_name,
    tmp_path,
) -> None:
    source = LinearARG.read(linarg_h5_path, block=first_block_name)
    root_path = tmp_path / "root_lineararg.h5"
    source.write(root_path)

    operator = _PackedJaxLinearARG.from_hdf5(root_path, mesh=_graph_mesh(), dtype=np.float32)
    by_block = _PackedJaxLinearARG.from_hdf5_block(root_path, None, dtype=np.float32)
    rng = np.random.default_rng(20260817)
    weights = rng.normal(size=(source.shape[1], 2)).astype(np.float32)

    assert source.iids is not None
    assert operator.iids == tuple(source.iids.to_list())
    assert by_block.shape == source.shape
    np.testing.assert_allclose(np.asarray(operator.matmat(weights)), np.asarray(source @ weights), rtol=1e-5, atol=1e-5)

    with h5py.File(root_path, "a") as file:
        file.create_group("chr1:0-1")
    with pytest.raises(ValueError, match="mixed.*root.*block"):
        _PackedJaxLinearARG.from_hdf5(root_path, mesh=_graph_mesh())


def test_private_packed_hdf5_products_match_exact_ragged(
    linarg_h5_path,
    linarg_block_metadata,
) -> None:
    metadata = linarg_block_metadata.select("block_name", "n", "n_entries", "n_variants", "n_samples")
    packed = _PackedJaxLinearARG.from_hdf5(
        linarg_h5_path,
        mesh=_graph_mesh(),
        block_metadata=metadata,
    )
    exact_mesh = Mesh(np.asarray(jax.devices("cpu")[:1]), ("blocks",))
    exact = JaxParallelOperator.from_hdf5(
        linarg_h5_path,
        mesh=exact_mesh,
        block_metadata=linarg_block_metadata,
        backend=Backend.PURE_JAX,
    )
    rng = np.random.default_rng(20260817)
    weights = rng.normal(size=(packed.shape[1], 2)).astype(np.float32)
    samples = rng.normal(size=(packed.shape[0], 2)).astype(np.float32)

    np.testing.assert_allclose(
        np.asarray(packed.matmat(weights)),
        np.asarray(exact.matmat(weights)),
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        np.asarray(packed.rmatmat(samples)),
        np.asarray(exact.rmatmat(samples)),
        rtol=1e-5,
        atol=1e-5,
    )


def test_private_packed_plural_ingress_converts_and_canonicalizes_each_block_once(
    linarg_h5_path,
    linarg_block_metadata,
    monkeypatch,
) -> None:
    block_names = linarg_block_metadata.get_column("block_name").to_list()
    lineargs = tuple(LinearARG.read(linarg_h5_path, block=name) for name in block_names)
    conversions: list[int] = []
    canonicalizations: list[int] = []
    original_convert = ingress_module._lineararg_block_arrays
    original_canonicalize = ingress_module.canonicalize_block_arrays

    def tracked_convert(linarg, *, dtype=None):
        conversions.append(id(linarg))
        return original_convert(linarg, dtype=dtype)

    def tracked_canonicalize(arrays, *, dtype=None):
        canonicalizations.append(id(arrays))
        return original_canonicalize(arrays, dtype=dtype)

    monkeypatch.setattr(ingress_module, "_lineararg_block_arrays", tracked_convert)
    monkeypatch.setattr(ingress_module, "canonicalize_block_arrays", tracked_canonicalize)

    operator = _PackedJaxLinearARG.from_linearargs(lineargs, mesh=_graph_mesh())

    assert operator.shape[1] == sum(linarg.shape[1] for linarg in lineargs)
    assert conversions == [id(linarg) for linarg in lineargs]
    assert len(canonicalizations) == len(lineargs)


def test_private_packed_plural_ingress_plans_then_stages_one_live_transfer_block(
    linarg_h5_path,
    linarg_block_metadata,
    monkeypatch,
) -> None:
    block_names = linarg_block_metadata.get_column("block_name").to_list()
    lineargs = tuple(LinearARG.read(linarg_h5_path, block=name) for name in block_names)
    expected_staging = tuple(
        ingress_module._block_metrics(
            ingress_module.canonicalize_block_arrays(
                ingress_module._lineararg_block_arrays(linarg),
                dtype=np.float32,
            )
        ).canonical_bytes
        for linarg in lineargs
    )
    events: list[str] = []
    live_blocks: set[int] = set()
    peak_live_blocks = 0
    captured_diagnostics = []
    original_convert = ingress_module._lineararg_block_arrays
    original_canonicalize = ingress_module.canonicalize_block_arrays
    original_plan = ingress_module._plan_packing_from_summaries
    original_packed_from_plan = ingress_module._packed_from_plan

    def tracked_convert(linarg, *, dtype=None):
        nonlocal peak_live_blocks
        block = original_convert(linarg, dtype=dtype)
        block_id = id(block)
        events.append("convert")
        live_blocks.add(block_id)
        peak_live_blocks = max(peak_live_blocks, len(live_blocks))
        weakref.finalize(block, live_blocks.discard, block_id)
        return block

    def tracked_canonicalize(block, *, dtype=None):
        events.append("canonicalize")
        return original_canonicalize(block, dtype=dtype)

    def tracked_plan(summaries, **kwargs):
        plan = original_plan(summaries, **kwargs)
        events.append("plan")
        return plan

    def tracked_packed_from_plan(*args, **kwargs):
        result = original_packed_from_plan(*args, **kwargs)
        captured_diagnostics.append(result.diagnostics)
        return result

    monkeypatch.setattr(ingress_module, "_lineararg_block_arrays", tracked_convert)
    monkeypatch.setattr(ingress_module, "canonicalize_block_arrays", tracked_canonicalize)
    monkeypatch.setattr(ingress_module, "_plan_packing_from_summaries", tracked_plan)
    monkeypatch.setattr(ingress_module, "_packed_from_plan", tracked_packed_from_plan)

    _PackedJaxLinearARG.from_linearargs(lineargs, mesh=_graph_mesh())
    gc.collect()

    assert events == ["plan", *(event for _ in lineargs for event in ("convert", "canonicalize"))]
    assert peak_live_blocks == 1
    assert not live_blocks
    assert len(captured_diagnostics) == 1
    assert captured_diagnostics[0].staging_bytes == max(expected_staging)
    assert captured_diagnostics[0].staging_bytes_by_device == (max(expected_staging),)


def test_private_packed_plural_ingress_rejects_inconsistent_iids(
    linarg_h5_path,
    linarg_block_metadata,
) -> None:
    block_names = linarg_block_metadata.get_column("block_name").to_list()
    lineargs = [LinearARG.read(linarg_h5_path, block=name) for name in block_names]
    second_iids = lineargs[1].iids
    assert second_iids is not None
    lineargs[1].iids = pl.Series([*second_iids[:-1], "different-iid"])

    with pytest.raises(ValueError, match="identical IID metadata"):
        _PackedJaxLinearARG.from_linearargs(lineargs, mesh=_graph_mesh())


def test_private_packed_hdf5_opens_once_and_never_uses_eager_block_reader(
    linarg_h5_path,
    linarg_block_metadata,
    monkeypatch,
) -> None:
    opens = 0
    original_file = h5py.File

    def tracked_file(*args, **kwargs):
        nonlocal opens
        opens += 1
        return original_file(*args, **kwargs)

    def fail_eager_reader(*args, **kwargs):
        raise AssertionError("packed HDF5 ingress must not construct eager exact blocks")

    monkeypatch.setattr(h5py, "File", tracked_file)
    monkeypatch.setattr(ingress_module, "read_hdf5_blocks", fail_eager_reader)

    _PackedJaxLinearARG.from_hdf5(
        linarg_h5_path,
        mesh=_graph_mesh(),
        block_metadata=linarg_block_metadata,
    )

    assert opens == 1


def test_private_packed_hdf5_rejects_empty_and_partial_root_sources(tmp_path) -> None:
    empty_path = tmp_path / "empty.h5"
    with h5py.File(empty_path, "w"):
        pass

    with pytest.raises(ValueError, match="empty or corrupt"):
        _PackedJaxLinearARG.from_hdf5(empty_path, mesh=_graph_mesh())

    with h5py.File(empty_path, "a") as file:
        file.create_dataset("indptr", data=np.array([0, 0], dtype=np.int32))
        file.attrs["n"] = 1
    with pytest.raises(ValueError, match="missing root graph fields"):
        _PackedJaxLinearARG.from_hdf5(empty_path, mesh=_graph_mesh())


def _write_packed_hdf5_schema_variant(
    path,
    linarg,
    *,
    layout: str,
    include_optional_arrays: bool,
    include_n_individuals: bool,
) -> tuple[str | None, int]:
    graph = linarg.A.tocsc(copy=False)
    n_nodes = int(graph.shape[0])
    n_samples, n_variants = map(int, linarg.shape)
    block_name = None if layout == "root" else "chr1:0-1"
    with h5py.File(path, "w") as file:
        group = file if block_name is None else file.create_group(block_name)
        group.attrs.update(
            {
                "n": n_nodes,
                "n_entries": int(graph.nnz),
                "n_samples": n_samples,
                "n_variants": n_variants,
            }
        )
        if include_n_individuals:
            group.attrs["n_individuals"] = 1
        group.create_dataset("indptr", data=np.asarray(graph.indptr, dtype=np.int32))
        group.create_dataset("indices", data=np.asarray(graph.indices, dtype=np.int32))
        group.create_dataset("data", data=np.asarray(graph.data, dtype=np.float32))
        group.create_dataset("variant_indices", data=np.asarray(linarg.variant_indices, dtype=np.int32))
        group.create_dataset("flip", data=np.asarray(linarg.flip, dtype=np.bool_))
        if include_optional_arrays:
            group.create_dataset("nonunique_indices", data=np.arange(n_nodes, dtype=np.int32))
            group.create_dataset("allele_counts", data=np.arange(n_variants, dtype=np.int32))
        file.create_dataset("iids", data=np.asarray([f"iid-{index}" for index in range(n_samples)], dtype="S"))
    return block_name, n_nodes


@pytest.mark.parametrize("layout", ["root", "group"])
@pytest.mark.parametrize("include_optional_arrays", [False, True])
@pytest.mark.parametrize("include_n_individuals", [False, True])
def test_private_packed_hdf5_accepts_schema_optional_arrays_and_individual_metadata(
    oracle_case,
    tmp_path,
    layout,
    include_optional_arrays,
    include_n_individuals,
) -> None:
    path = tmp_path / f"{layout}-{include_optional_arrays}-{include_n_individuals}.h5"
    block_name, n_nodes = _write_packed_hdf5_schema_variant(
        path,
        oracle_case.linarg,
        layout=layout,
        include_optional_arrays=include_optional_arrays,
        include_n_individuals=include_n_individuals,
    )

    operator = _PackedJaxLinearARG.from_hdf5(path, mesh=_graph_mesh(), max_padding_ratio=None)
    descriptor = np.asarray(operator.block_descriptors)[0, 0]
    node_start = int(descriptor[BLOCK_DESCRIPTOR_FIELDS.index("node_start")])
    node_length = int(descriptor[BLOCK_DESCRIPTOR_FIELDS.index("node_length")])
    variant_start = int(descriptor[BLOCK_DESCRIPTOR_FIELDS.index("variant_start")])
    variant_length = int(descriptor[BLOCK_DESCRIPTOR_FIELDS.index("variant_length")])
    sample_start = int(descriptor[BLOCK_DESCRIPTOR_FIELDS.index("sample_start")])
    sample_length = int(descriptor[BLOCK_DESCRIPTOR_FIELDS.index("sample_length")])
    expected_nonunique = np.arange(n_nodes, dtype=np.int32)
    expected_allele_counts = (
        np.arange(operator.n_variants, dtype=np.int32)
        if include_optional_arrays
        else np.full(operator.n_variants, -1, dtype=np.int32)
    )
    expected_samples = ingress_module._sample_indices(
        n_nodes,
        operator.n_samples,
        1 if include_n_individuals else None,
    )

    assert block_name is None or block_name == "chr1:0-1"
    np.testing.assert_array_equal(
        np.asarray(operator.nonunique_indices)[0, node_start : node_start + node_length],
        expected_nonunique,
    )
    np.testing.assert_array_equal(
        np.asarray(operator.allele_counts)[0, variant_start : variant_start + variant_length],
        expected_allele_counts,
    )
    np.testing.assert_array_equal(
        np.asarray(operator.sample_indices)[0, sample_start : sample_start + sample_length],
        expected_samples,
    )


@pytest.mark.parametrize("column", ["n", "n_variants", "n_samples"])
def test_private_packed_hdf5_rejects_each_shape_metadata_mismatch(
    linarg_h5_path,
    linarg_block_metadata,
    column,
) -> None:
    selected = linarg_block_metadata.select(
        "block_name",
        "n",
        "n_entries",
        "n_variants",
        "n_samples",
    ).head(1)
    mismatched = selected.with_columns((pl.col(column) + 1).alias(column))

    with pytest.raises(ValueError, match=rf"block_metadata {column} mismatch"):
        _PackedJaxLinearARG.from_hdf5(
            linarg_h5_path,
            mesh=_graph_mesh(),
            block_metadata=mismatched,
        )
